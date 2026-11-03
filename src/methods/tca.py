"""Transport-Curvature Aggregation (TCA) method utilities."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import ot
import torch
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

from ..core.train_utils import weighted_average


def flatten_split_params(
    client_net: torch.nn.Module, server_net: torch.nn.Module
) -> np.ndarray:
    """Flatten trainable floating-point parameters of both client-half and server-half models."""
    vecs: List[np.ndarray] = []
    for module in (client_net, server_net):
        for p in module.parameters():
            if p.is_floating_point():
                vecs.append(p.detach().cpu().reshape(-1).to(dtype=torch.float64).numpy())
    if not vecs:
        return np.zeros(1, dtype=np.float64)
    return np.concatenate(vecs, axis=0)


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


def build_cost_matrix(
    cost_mode: str,
    deltas: Optional[List[np.ndarray]] = None,
    emd_matrix: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """Build normalized m x m pairwise cost matrix C (`cost_mode` in {'update', 'emd'})."""
    cost_mode = cost_mode.strip().lower()
    if cost_mode == "update":
        if deltas is None or len(deltas) == 0:
            raise ValueError("build_cost_matrix requires non-empty `deltas` when cost_mode='update'.")
        stacked = np.stack([np.asarray(d, dtype=np.float64).reshape(-1) for d in deltas], axis=0)
        if stacked.shape[0] == 1:
            return np.zeros((1, 1), dtype=np.float64)
        raw_c = squareform(pdist(stacked, metric="euclidean"))
    elif cost_mode == "emd":
        if emd_matrix is None:
            raise ValueError("build_cost_matrix requires `emd_matrix` when cost_mode='emd'.")
        raw_c = np.asarray(emd_matrix, dtype=np.float64).copy()
    else:
        raise ValueError(f"Unsupported TCA cost_mode: {cost_mode}. Expected 'update' or 'emd'.")

    raw_c = 0.5 * (raw_c + raw_c.T)
    np.fill_diagonal(raw_c, 0.0)
    positives = raw_c[np.triu_indices(raw_c.shape[0], 1)]
    pos_vals = positives[positives > eps]
    tau = float(np.median(pos_vals)) if pos_vals.size > 0 else 1.0
    C = raw_c / max(tau, eps)
    np.fill_diagonal(C, 0.0)
    return C


def _solve_row_weighted_simplex(
    c_row: np.ndarray, v: np.ndarray, reg: float
) -> np.ndarray:
    """Solve min_{a >= 0, sum(a) = 1} <c_row, a> + (reg / 2) * sum_j (a_j - v_j)^2 / v_j."""
    m = len(v)
    if m == 1:
        return np.ones(1, dtype=np.float64)

    reg = max(float(reg), 1e-8)
    shift = c_row - reg

    def row_sum(mu: float) -> float:
        a = v * np.maximum(0.0, -(shift - mu) / reg)
        return float(a.sum())

    lo = float(np.min(shift)) - reg
    hi = float(np.max(shift)) + reg / max(float(np.min(v)), 1e-8)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if row_sum(mid) < 1.0:
            lo = mid
        else:
            hi = mid
    mu_star = 0.5 * (lo + hi)
    a = v * np.maximum(0.0, -(shift - mu_star) / reg)
    a[a < 1e-10] = 0.0
    total = float(a.sum())
    if total <= 0.0:
        return v.copy()
    return a / total


def solve_tca_plan(
    C: np.ndarray,
    v: np.ndarray,
    marginal: str = "hard",
    reg: float = 1.0,
) -> np.ndarray:
    """Solve for the m x m row-stochastic aggregation plan A given cost matrix C and weights v."""
    C = np.asarray(C, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    v = v / max(float(v.sum()), 1e-12)
    m = len(v)
    if m == 1:
        return np.ones((1, 1), dtype=np.float64)

    marginal = marginal.strip().lower()
    reg = max(float(reg), 1e-6)

    if marginal == "none":
        A = np.zeros((m, m), dtype=np.float64)
        for i in range(m):
            A[i] = _solve_row_weighted_simplex(C[i], v, reg=reg)
        return A

    if marginal != "hard":
        raise ValueError(f"Unsupported TCA marginal mode: {marginal}. Expected 'hard' or 'none'.")

    vv = np.outer(v, v)
    inv_vv = 1.0 / np.maximum(vv, 1e-12)
    c_flat = C.reshape(-1)
    vv_flat = vv.reshape(-1)
    inv_vv_flat = inv_vv.reshape(-1)

    def obj_and_grad(p_flat: np.ndarray) -> Tuple[float, np.ndarray]:
        diff = p_flat - vv_flat
        val = float(np.dot(c_flat, p_flat) + 0.5 * reg * np.dot(diff * diff, inv_vv_flat))
        grad = c_flat + reg * diff * inv_vv_flat
        return val, grad

    constraints = []
    for i in range(m):
        row_mask = np.zeros((m, m), dtype=np.float64)
        row_mask[i, :] = 1.0
        rm_flat = row_mask.reshape(-1)
        constraints.append(
            {
                "type": "eq",
                "fun": lambda p, r=rm_flat, vi=v[i]: float(np.dot(r, p) - vi),
                "jac": lambda p, r=rm_flat: r,
            }
        )
    for j in range(m - 1):
        col_mask = np.zeros((m, m), dtype=np.float64)
        col_mask[:, j] = 1.0
        cm_flat = col_mask.reshape(-1)
        constraints.append(
            {
                "type": "eq",
                "fun": lambda p, c=cm_flat, vj=v[j]: float(np.dot(c, p) - vj),
                "jac": lambda p, c=cm_flat: c,
            }
        )

    bounds = [(0.0, float(min(v[i], v[j]))) for i in range(m) for j in range(m)]
    res = minimize(
        obj_and_grad,
        vv_flat.copy(),
        method="SLSQP",
        jac=True,
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 500, "disp": False},
    )

    P = res.x.reshape(m, m) if res.success else vv.copy()
    P = np.maximum(P, 0.0)
    P[P < 1e-8 * vv] = 0.0

    for i in range(m):
        rs = float(P[i].sum())
        if rs > 1e-12:
            P[i] *= v[i] / rs
        else:
            P[i] = v[i] * v

    A = P / np.maximum(v[:, None], 1e-12)
    A[A < 1e-7] = 0.0
    row_sums = A.sum(axis=1, keepdims=True)
    A = np.where(row_sums > 1e-12, A / np.maximum(row_sums, 1e-12), v[None, :])
    return A


def plan_curvature(
    A: np.ndarray,
    C: np.ndarray,
    v: Optional[np.ndarray] = None,
    zero_tol: float = 1e-6,
    violation_tol: float = 1e-7,
) -> Dict[str, Union[float, int, np.ndarray]]:
    """Compute pairwise Ollivier-Ricci curvature of plan A under cost C, sparsity, and violations."""
    A = np.asarray(A, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
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
                if k != i and k != j and C[i, j] > C[i, k] + C[k, j] + violation_tol:
                    violations += 1

    for i in range(m):
        ai = A[i] / max(float(A[i].sum()), 1e-12)
        for j in range(i + 1, m):
            cij = float(C[i, j])
            if cij <= 1e-12:
                kij = 1.0
            else:
                aj = A[j] / max(float(A[j].sum()), 1e-12)
                w1 = float(ot.emd2(ai, aj, C))
                if w1 > cij + violation_tol:
                    violations += 1
                kij = 1.0 - (w1 / cij)
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
    C: Optional[np.ndarray] = None,
    v: Optional[np.ndarray] = None,
    curv_stats: Optional[Dict[str, Union[float, int, np.ndarray]]] = None,
) -> Tuple[np.ndarray, float, float]:
    """Calibrate plan A via lazy consensus mixing A_cal = (1 - beta) * A + beta * (1 v^T)."""
    A = np.asarray(A, dtype=np.float64)
    m = A.shape[0]
    if v is None:
        v_vec = np.full(m, 1.0 / max(m, 1), dtype=np.float64)
    else:
        v_vec = np.asarray(v, dtype=np.float64)
        v_vec = v_vec / max(float(v_vec.sum()), 1e-12)

    if curv_stats is None and C is not None:
        curv_stats = plan_curvature(A, C, v=v_vec)

    kappa_min = float(curv_stats["kappa_min"]) if curv_stats is not None else 0.0
    kappa_mean = float(curv_stats["kappa_mean"]) if curv_stats is not None else 0.5

    if isinstance(kappa_target, str):
        target = kappa_mean if kappa_target.strip().lower() == "auto" else float(kappa_target)
    elif kappa_target is None:
        target = kappa_mean
    else:
        target = float(kappa_target)

    q_eff = max(float(median_q), 1e-6)
    denom = q_eff * max(1.0 - kappa_min, 1e-6)
    numer = max(0.0, 1.0 - target)
    beta = float(np.clip(1.0 - (numer / denom), 0.0, 1.0))

    consensus = np.outer(np.ones(m, dtype=np.float64), v_vec)
    A_cal = (1.0 - beta) * A + beta * consensus
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


