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
    clip_max: float = 10.0,
) -> Union[float, Tuple[np.ndarray, float]]:
    """Estimate local step contraction rate q = ||v2|| / ||v1|| from midpoint norms.

    If scalar floats are provided, returns a single float `q`.
    If 1D arrays/lists of per-client norms are provided, returns `(q_per_client, median_q)`.
    """
    if np.isscalar(norm_v1) and np.isscalar(norm_v2):
        n1 = float(norm_v1)
        n2 = float(norm_v2)
        if n1 <= eps:
            return 1.0
        return float(np.clip(n2 / n1, 0.0, clip_max))

    arr1 = np.asarray(norm_v1, dtype=np.float64)
    arr2 = np.asarray(norm_v2, dtype=np.float64)
    if arr1.ndim > 1:
        arr1 = np.linalg.norm(arr1, axis=1)
    if arr2.ndim > 1:
        arr2 = np.linalg.norm(arr2, axis=1)

    q_vals = np.where(arr1 > eps, np.clip(arr2 / np.maximum(arr1, eps), 0.0, clip_max), 1.0)
    median_q = float(np.median(q_vals)) if q_vals.size > 0 else 1.0
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
    v: Optional[np.ndarray] = None,
    marginal: str = "hard",
    tau: float = 1.0,
    var_scale: Optional[float] = None,
    sample_counts: Optional[Union[List[int], np.ndarray]] = None,
    reg: Optional[float] = None,
) -> np.ndarray:
    """Solve for the m x m row-stochastic aggregation plan A using CVXPY + Clarabel.

    Per-client variance weights satisfy:
        v_k = var_scale * mean(n) / n_k
    Objective:
        min_{A >= 0} sum_{i,k} C_{ik} A_{ik} + sum_{i,k} v_k A_{ik}^2
                     [+ tau * sum_k rel_entr(col_marginal_k, 1/m) if marginal == 'unbalanced']
    Constraints:
        - A >= 0, sum_k A_{ik} == 1  (all modes)
        - sum_i A_{ik} == 1          (when marginal == 'hard')
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
    elif v is not None:
        v_vec = np.asarray(v, dtype=np.float64).reshape(-1)
        if var_scale is not None:
            v_vec = v_vec * float(var_scale)
    else:
        scale = 1.0 if var_scale is None else float(var_scale)
        v_vec = np.full(m, scale, dtype=np.float64)

    if v_vec.shape[0] != m:
        raise ValueError(f"Length of v ({v_vec.shape[0]}) must match C dimension ({m}).")

    marginal = marginal.strip().lower()
    A = cp.Variable((m, m), nonneg=True)

    bias_term = cp.sum(cp.multiply(C, A))
    var_term = cp.sum(cp.multiply(cp.square(A), v_vec[None, :]))
    constraints = [A >= 0, cp.sum(A, axis=1) == 1]

    if marginal == "hard":
        constraints.append(cp.sum(A, axis=0) == 1)
        objective = cp.Minimize(bias_term + var_term)
    elif marginal == "unbalanced":
        if tau <= 0:
            raise ValueError("`tau` must be strictly positive when marginal='unbalanced'.")
        col_sums = cp.sum(A, axis=0)
        ones = np.ones(m, dtype=np.float64)
        kl_term = float(tau) * cp.sum(cp.rel_entr(col_sums, ones))
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
    uniform_row = np.full((1, m), 1.0 / float(m), dtype=np.float64)
    A_val = np.where(row_sums > 1e-12, A_val / np.maximum(row_sums, 1e-12), uniform_row)
    return A_val



def plan_curvature(
    A: np.ndarray,
    D: Optional[np.ndarray] = None,
    C: Optional[np.ndarray] = None,
    zero_tol: float = 1e-6,
    violation_tol: float = 1e-7,
) -> Dict[str, Union[float, int, np.ndarray]]:
    """Compute pairwise Ollivier-Ricci curvature of plan A under ground metric D, sparsity, and violations."""
    A = np.asarray(A, dtype=np.float64)
    metric_d = D if D is not None else C
    if metric_d is None:
        raise ValueError("plan_curvature requires ground metric `D`.")
    D_mat = np.asarray(metric_d, dtype=np.float64)
    m = A.shape[0]

    sparsity = float(np.mean(np.abs(A) <= zero_tol))
    if m <= 1:
        return {
            "kappa_matrix": np.zeros((m, m), dtype=np.float64),
            "kappa_min": 1.0,
            "kappa_mean": 1.0,
            "sparsity": sparsity,
            "inequality_violations": 0,
        }

    kappa_mat = np.zeros((m, m), dtype=np.float64)
    np.fill_diagonal(kappa_mat, 1.0)
    pair_kappas: List[float] = []
    violations = 0

    for i in range(m):
        for j in range(i + 1, m):
            for k in range(m):
                if k != i and k != j and D_mat[i, j] > D_mat[i, k] + D_mat[k, j] + violation_tol:
                    violations += 1

    for i in range(m):
        ai = A[i] / max(float(A[i].sum()), 1e-12)
        for j in range(i + 1, m):
            dij = float(D_mat[i, j])
            if dij <= 1e-12:
                kij = 1.0
            else:
                aj = A[j] / max(float(A[j].sum()), 1e-12)
                w1 = float(ot.emd2(ai, aj, D_mat))
                if w1 > dij + violation_tol:
                    violations += 1
                kij = 1.0 - (w1 / dij)
            kappa_mat[i, j] = kappa_mat[j, i] = kij
            pair_kappas.append(kij)

    kappa_min = float(np.min(pair_kappas)) if pair_kappas else 1.0
    kappa_mean = float(np.mean(pair_kappas)) if pair_kappas else 1.0

    return {
        "kappa_matrix": kappa_mat,
        "kappa_min": kappa_min,
        "kappa_mean": kappa_mean,
        "sparsity": sparsity,
        "inequality_violations": int(violations),
    }


def lazy_calibrate(
    A: np.ndarray,
    median_q: float,
    kappa_target: Union[str, float] = "auto",
    D: Optional[np.ndarray] = None,
    C: Optional[np.ndarray] = None,
    curv_stats: Optional[Dict[str, Union[float, int, np.ndarray]]] = None,
) -> Tuple[np.ndarray, float, float]:
    """Calibrate plan A via lazy consensus mixing A_cal = (1 - beta) * A + beta * (1 1^T / m)."""
    A = np.asarray(A, dtype=np.float64)
    m = A.shape[0]
    metric_d = D if D is not None else C

    if curv_stats is None and metric_d is not None:
        curv_stats = plan_curvature(A, metric_d)

    if curv_stats is not None:
        kappa_mean = float(curv_stats["kappa_mean"])
    else:
        tv_vals = []
        for i in range(m):
            for j in range(i + 1, m):
                tv_vals.append(1.0 - 0.5 * float(np.sum(np.abs(A[i] - A[j]))))
        kappa_mean = float(np.mean(tv_vals)) if tv_vals else 0.5

    if isinstance(kappa_target, str):
        target = kappa_mean if kappa_target.strip().lower() == "auto" else float(kappa_target)
    elif kappa_target is None:
        target = kappa_mean
    else:
        target = float(kappa_target)

    q_eff = max(float(median_q), 1e-6)
    denom = q_eff * max(1.0 - kappa_mean, 1e-6)
    numer = max(0.0, 1.0 - target)
    beta = float(np.clip(1.0 - (numer / denom), 0.0, 1.0))

    u = np.full((m, m), 1.0 / max(float(m), 1.0), dtype=np.float64)
    A_cal = (1.0 - beta) * A + beta * u
    A_cal[A_cal < 1e-10] = 0.0
    A_cal = A_cal / np.maximum(A_cal.sum(axis=1, keepdims=True), 1e-12)
    return A_cal, beta, target



def tca_aggregate_split_models(
    w_locals_client: List[dict],
    w_locals_server: List[dict],
    A: np.ndarray,
    v: np.ndarray,
    active_users: List[int],
) -> Tuple[dict, dict, Dict[int, dict], Dict[int, dict]]:
    """Aggregate both client-half and server-half state_dicts with the same plan A."""
    A = np.asarray(A, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    v = v / max(float(v.sum()), 1e-12)

    pers_client_by_id: Dict[int, dict] = {}
    pers_server_by_id: Dict[int, dict] = {}

    for row_idx, cid in enumerate(active_users):
        cid_int = int(cid)
        row_weights = np.maximum(A[row_idx], 0.0)
        if float(row_weights.sum()) <= 0.0:
            row_weights = v.copy()
        pers_client_by_id[cid_int] = weighted_average(w_locals_client, row_weights)
        pers_server_by_id[cid_int] = weighted_average(w_locals_server, row_weights)

    global_weights = np.maximum(v @ A, 0.0)
    if float(global_weights.sum()) <= 0.0:
        global_weights = v.copy()
    w_glob_client = weighted_average(w_locals_client, global_weights)
    w_glob_server = weighted_average(w_locals_server, global_weights)

    return w_glob_client, w_glob_server, pers_client_by_id, pers_server_by_id


