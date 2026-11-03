"""Transport-Curvature Aggregation (TCA) method utilities."""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import cvxpy as cp
import numpy as np
import ot
import torch
from scipy.spatial.distance import pdist, squareform

from ..core.train_utils import flatten_split_params, weighted_average


def estimate_q(
    norm_v1: Union[float, List[float], np.ndarray],
    norm_v2: Union[float, List[float], np.ndarray],
    eps: float = 1e-12,
    return_raw: bool = False,
) -> Union[float, Tuple[float, float], Tuple[np.ndarray, float], Tuple[np.ndarray, float, np.ndarray]]:
    """Estimate round contraction rate q from half-round norms:
        ratio = n2 / n1
        q = clip(ratio, 0.0, 0.999) ** 2

    If scalar floats are provided, returns `q` (or `(q, q_half_raw)` if `return_raw=True`).
    If 1D arrays/lists of per-client norms are provided, returns `(q_per_client, median_q)`
    (or `(q_per_client, median_q, q_half_raw)` if `return_raw=True`).
    """
    if np.isscalar(norm_v1) and np.isscalar(norm_v2):
        n1 = float(norm_v1)
        n2 = float(norm_v2)
        ratio = (n2 / n1) if n1 > eps else 1.0
        q = float(np.clip(ratio, 0.0, 0.999) ** 2)
        if return_raw:
            return q, float(ratio)
        return q

    arr1 = np.asarray(norm_v1, dtype=np.float64)
    arr2 = np.asarray(norm_v2, dtype=np.float64)
    if arr1.ndim > 1:
        arr1 = np.linalg.norm(arr1, axis=1)
    if arr2.ndim > 1:
        arr2 = np.linalg.norm(arr2, axis=1)

    q_half_raw = np.where(arr1 > eps, arr2 / np.maximum(arr1, eps), 1.0)
    q_vals = np.clip(q_half_raw, 0.0, 0.999) ** 2
    median_q = float(np.median(q_vals)) if q_vals.size > 0 else 1.0
    if return_raw:
        return q_vals, median_q, q_half_raw
    return q_vals, median_q


def build_variance_vector(
    sample_counts: Union[List[int], List[float], np.ndarray],
    var_scale: float = 1.0,
    reg: Optional[float] = None,
) -> np.ndarray:
    """Compute per-client variance weights v_k = var_scale * mean(n) / n_k."""
    if reg is not None:
        warnings.warn("`reg` is deprecated; use `var_scale` instead.", DeprecationWarning, stacklevel=2)
        var_scale = float(reg)
    n = np.asarray(sample_counts, dtype=np.float64)
    if n.size == 0:
        raise ValueError("sample_counts must be non-empty.")
    if np.any(n <= 0):
        raise ValueError("All sample_counts must be strictly positive.")
    return float(var_scale) * (float(np.mean(n)) / n)


def build_cost_matrix(
    cost_mode: str = "update",
    deltas: Optional[List[np.ndarray]] = None,
    emd_matrix: Optional[np.ndarray] = None,
    D: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """Build median-normalized m x m pairwise distance/cost matrix D (`cost_mode` in {'update', 'emd'})."""
    cost_mode = cost_mode.strip().lower()
    if D is not None:
        raw_d = np.asarray(D, dtype=np.float64).copy()
    elif cost_mode == "update":
        if deltas is None or len(deltas) == 0:
            raise ValueError("build_cost_matrix requires non-empty `deltas` when cost_mode='update'.")
        stacked = np.stack([np.asarray(d, dtype=np.float64).reshape(-1) for d in deltas], axis=0)
        if stacked.shape[0] == 1:
            return np.zeros((1, 1), dtype=np.float64)
        raw_d = squareform(pdist(stacked, metric="euclidean"))
    elif cost_mode == "emd":
        if emd_matrix is None:
            raise ValueError("build_cost_matrix requires `emd_matrix` when cost_mode='emd'.")
        raw_d = np.asarray(emd_matrix, dtype=np.float64).copy()
    else:
        raise ValueError(f"Unsupported TCA cost_mode: {cost_mode}. Expected 'update' or 'emd'.")

    raw_d = 0.5 * (raw_d + raw_d.T)
    np.fill_diagonal(raw_d, 0.0)
    off_diag = raw_d[np.triu_indices(raw_d.shape[0], k=1)]
    pos_vals = off_diag[off_diag > eps]
    median_d = float(np.median(pos_vals)) if pos_vals.size > 0 else 1.0
    D_norm = raw_d / max(median_d, eps)
    np.fill_diagonal(D_norm, 0.0)
    return D_norm



def solve_tca_plan(
    C: np.ndarray,
    v: Optional[Union[float, List[float], np.ndarray]] = None,
    rho: Optional[Union[List[float], np.ndarray]] = None,
    marginal: str = "hard",
    tau: float = 1.0,
    var_scale: Optional[float] = None,
    sample_counts: Optional[Union[List[int], np.ndarray]] = None,
    reg: Optional[float] = None,
) -> np.ndarray:
    """Solve for the m x m row-stochastic aggregation plan A using CVXPY + Clarabel.

    Per-client variance weights satisfy:
        v_k = var_scale * mean(n) / n_k
    Data shares satisfy:
        rho = np.asarray(rho, float); rho = rho / rho.sum(); col_target = m * rho
    Objective:
        min_{A >= 0} sum_{i,k} C_{ik} A_{ik} + sum_{i,k} v_k A_{ik}^2
                     [+ tau * sum_k rel_entr(sum_i A_{ik}, col_target_k) if marginal == 'unbalanced']
    Constraints:
        - A >= 0, sum_k A_{ik} == 1          (all modes)
        - sum_i A_{ik} == col_target_k       (when marginal == 'hard')
    """
    if reg is not None:
        warnings.warn("`reg` is deprecated; use `var_scale` instead.", DeprecationWarning, stacklevel=2)
        if var_scale is None:
            var_scale = float(reg)

    C = np.asarray(C, dtype=np.float64)
    m = C.shape[0]
    if m == 1:
        return np.ones((1, 1), dtype=np.float64)

    if sample_counts is not None:
        scale = 1.0 if var_scale is None else float(var_scale)
        v_vec = build_variance_vector(sample_counts, var_scale=scale)
        if rho is None:
            rho = np.asarray(sample_counts, dtype=float)
    elif v is not None:
        if np.isscalar(v):
            v_vec = np.full(m, float(v), dtype=np.float64)
        else:
            v_vec = np.asarray(v, dtype=np.float64).reshape(-1)
        if var_scale is not None:
            v_vec = v_vec * float(var_scale)
    else:
        scale = 1.0 if var_scale is None else float(var_scale)
        v_vec = np.full(m, scale, dtype=np.float64)

    if v_vec.shape[0] != m:
        raise ValueError(f"Length of v ({v_vec.shape[0]}) must match C dimension ({m}).")

    if rho is None:
        rho = np.ones(m, dtype=float) / float(m)
    rho = np.asarray(rho, float)
    rho = rho / rho.sum()
    col_target = m * rho

    marginal = marginal.strip().lower()
    A = cp.Variable((m, m), nonneg=True)

    bias_term = cp.sum(cp.multiply(C, A))
    var_term = cp.sum(cp.multiply(cp.square(A), v_vec[None, :]))
    constraints = [A >= 0, cp.sum(A, axis=1) == 1]

    if marginal == "hard":
        constraints.append(cp.sum(A, axis=0) == col_target)
        objective = cp.Minimize(bias_term + var_term)
    elif marginal == "unbalanced":
        if tau <= 0:
            raise ValueError("`tau` must be strictly positive when marginal='unbalanced'.")
        kl_term = float(tau) * cp.sum(cp.rel_entr(cp.sum(A, axis=0), col_target))
        objective = cp.Minimize(bias_term + var_term + kl_term)
    elif marginal == "none":
        objective = cp.Minimize(bias_term + var_term)
    else:
        raise ValueError(
            f"Unsupported TCA marginal mode: {marginal}. Expected 'hard', 'unbalanced', or 'none'."
        )

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.CLARABEL)
    except Exception:
        prob.solve(solver=cp.SCS)

    if A.value is None:
        prob.solve(solver=cp.SCS)

    A_val = np.asarray(A.value, dtype=np.float64)
    A_val = np.maximum(A_val, 0.0)
    A_val[A_val < 1e-7] = 0.0
    row_sums = A_val.sum(axis=1, keepdims=True)
    A_val = np.where(row_sums > 1e-12, A_val / np.maximum(row_sums, 1e-12), rho[None, :])
    return A_val



def plan_curvature(
    A: np.ndarray,
    D: Optional[np.ndarray] = None,
    C: Optional[np.ndarray] = None,
    zero_tol: float = 1e-6,
    violation_tol: float = 1e-7,
) -> Dict[str, Union[float, int, np.ndarray]]:
    """Compute pairwise Ollivier-Ricci curvature of plan A under ground metric D."""
    A = np.asarray(A, dtype=np.float64)
    metric_d = D if D is not None else C
    if metric_d is None:
        raise ValueError("plan_curvature requires ground metric `D`.")
    D_mat = np.asarray(metric_d, dtype=np.float64)
    m = A.shape[0]

    row_sums = np.maximum(A.sum(axis=1, keepdims=True), 1e-12)
    A_norm = A / row_sums
    b = np.sum(A_norm * D_mat, axis=1)

    sparsity = float(np.mean(np.abs(A) <= zero_tol))
    if m <= 1:
        return {
            "kappa_matrix": np.zeros((m, m), dtype=np.float64),
            "kappa_min": 1.0,
            "kappa_mean": 1.0,
            "b": b,
            "sparsity": sparsity,
            "inequality_violations": 0,
            "metric_violations": 0,
        }

    kappa_mat = np.zeros((m, m), dtype=np.float64)
    np.fill_diagonal(kappa_mat, 1.0)
    pair_kappas: List[float] = []
    inequality_violations = 0
    metric_violations = 0

    for i in range(m):
        for j in range(i + 1, m):
            for k in range(m):
                if k != i and k != j and D_mat[i, j] > D_mat[i, k] + D_mat[k, j] + violation_tol:
                    metric_violations += 1

    for i in range(m):
        ai = A_norm[i]
        for j in range(i + 1, m):
            dij = float(D_mat[i, j])
            if dij <= 1e-12:
                kij = 1.0
            else:
                aj = A_norm[j]
                w1 = float(ot.emd2(ai, aj, D_mat))
                kij = 1.0 - (w1 / dij)
                upper_bound = (float(b[i]) + float(b[j])) / dij
                if kij > upper_bound + 1e-7:
                    inequality_violations += 1
            kappa_mat[i, j] = kappa_mat[j, i] = kij
            pair_kappas.append(kij)

    kappa_min = float(np.min(pair_kappas)) if pair_kappas else 1.0
    kappa_mean = float(np.mean(pair_kappas)) if pair_kappas else 1.0

    return {
        "kappa_matrix": kappa_mat,
        "kappa_min": kappa_min,
        "kappa_mean": kappa_mean,
        "b": b,
        "sparsity": sparsity,
        "inequality_violations": int(inequality_violations),
        "metric_violations": int(metric_violations),
    }


def lazy_calibrate(
    A: np.ndarray,
    median_q: float,
    kappa_target: Union[str, float] = "auto",
    D: Optional[np.ndarray] = None,
    C: Optional[np.ndarray] = None,
    curv_stats: Optional[Dict[str, Union[float, int, np.ndarray]]] = None,
) -> Tuple[np.ndarray, float, float]:
    """Calibrate plan A by mixing toward the identity matrix I."""
    A = np.asarray(A, dtype=np.float64)
    m = A.shape[0]
    metric_d = D if D is not None else C
    if metric_d is None and curv_stats is None:
        metric_d = np.ones((m, m), dtype=np.float64) - np.eye(m, dtype=np.float64)

    if curv_stats is None:
        curv_stats = plan_curvature(A, metric_d)

    q = float(np.clip(median_q, 0.0, 0.999))
    kappa_A = float(curv_stats["kappa_mean"])
    if isinstance(kappa_target, str):
        target = kappa_A if kappa_target.strip().lower() == "auto" else float(kappa_target)
    elif kappa_target is None:
        target = kappa_A
    else:
        target = float(kappa_target)

    kappa_round = target * (1.0 - q) / max(1.0 - q * target, 1e-12)
    if kappa_A <= 1e-6:
        warnings.warn("kappa_A <= 1e-6: calibration skipped (beta=1.0).", RuntimeWarning, stacklevel=2)
        beta = 1.0
    else:
        beta = float(np.clip(kappa_round / kappa_A, 0.0, 1.0))

    A_round = (1.0 - beta) * np.eye(m, dtype=np.float64) + beta * A
    return A_round, beta, target


def tca_aggregate_split_models(
    w_locals_client: List[dict],
    w_locals_server: List[dict],
    A: np.ndarray,
    rho: np.ndarray,
    active_users: List[int],
) -> Tuple[dict, dict, Dict[int, dict], Dict[int, dict]]:
    """Aggregate client-half and server-half state_dicts; global model is sum_k rho_k * theta_k."""
    A = np.asarray(A, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    rho = rho / max(float(rho.sum()), 1e-12)

    pers_client_by_id: Dict[int, dict] = {}
    pers_server_by_id: Dict[int, dict] = {}

    for row_idx, cid in enumerate(active_users):
        cid_int = int(cid)
        row_weights = np.maximum(A[row_idx], 0.0)
        if float(row_weights.sum()) <= 0.0:
            row_weights = rho.copy()
        pers_client_by_id[cid_int] = weighted_average(w_locals_client, row_weights)
        pers_server_by_id[cid_int] = weighted_average(w_locals_server, row_weights)

    w_glob_client = weighted_average(w_locals_client, rho)
    w_glob_server = weighted_average(w_locals_server, rho)

    return w_glob_client, w_glob_server, pers_client_by_id, pers_server_by_id


