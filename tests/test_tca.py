"""CPU unit tests for Phase 2a Transport-Curvature Aggregation (TCA)."""

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.methods.tca import (
    build_cost_matrix,
    build_variance_vector,
    estimate_q,
    lazy_calibrate,
    plan_curvature,
    solve_tca_plan,
)


def test_build_cost_matrix_median_normalization():
    """1. build_cost_matrix normalizes off-diagonal pairwise distances D by their median."""
    rng = np.random.default_rng(42)
    deltas = [rng.normal(size=32) * (i + 1) for i in range(6)]
    D = build_cost_matrix(cost_mode="update", deltas=deltas)

    assert D.shape == (6, 6)
    assert np.allclose(D, D.T, atol=1e-12)
    assert np.allclose(np.diag(D), 0.0, atol=1e-12)

    off_diag = D[np.triu_indices(6, k=1)]
    assert np.median(off_diag[off_diag > 0]) == pytest.approx(1.0, rel=1e-9, abs=1e-9)


def test_build_variance_vector_per_client():
    """2. Variance vector uses per-client v_k = var_scale * mean(n) / n_k (and supports reg alias)."""
    n = np.array([100, 200, 300, 600], dtype=np.float64)
    var_scale = 0.5
    v = build_variance_vector(n, var_scale=var_scale)
    expected = var_scale * (np.mean(n) / n)
    assert np.allclose(v, expected, atol=1e-12)

    with pytest.deprecated_call():
        v_alias = build_variance_vector(n, reg=var_scale)
    assert np.allclose(v_alias, expected, atol=1e-12)


def test_solve_tca_plan_hard_marginal():
    """3. solve_tca_plan with marginal='hard' produces a doubly stochastic matrix A."""
    rng = np.random.default_rng(7)
    deltas = [rng.normal(size=16) for _ in range(5)]
    D = build_cost_matrix("update", deltas=deltas)
    n = [100, 150, 200, 250, 300]
    v = build_variance_vector(n, var_scale=0.5)

    A = solve_tca_plan(D, v, marginal="hard")
    assert A.shape == (5, 5)
    assert np.all(A >= -1e-9)
    assert np.allclose(A.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(A.sum(axis=0), 1.0, atol=1e-4)


def test_solve_tca_plan_none_marginal_and_unequal_variance():
    """4. solve_tca_plan with marginal='none' and C=0 recovers sample-proportional weights A_{ik} = n_k / sum(n)."""
    n = np.array([100.0, 200.0, 300.0, 400.0])
    v = build_variance_vector(n, var_scale=1.0)
    C_zero = np.zeros((4, 4), dtype=np.float64)

    A = solve_tca_plan(C_zero, v, marginal="none")
    expected_row = n / n.sum()
    for i in range(4):
        assert np.allclose(A[i], expected_row, atol=1e-4)


def test_solve_tca_plan_unbalanced_marginal_cvxpy_clarabel():
    """5. solve_tca_plan with marginal='unbalanced' (rel_entr + Clarabel) interpolates between none and hard."""
    pts = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [5.0, 5.0]])
    D = build_cost_matrix("update", deltas=[pts[i] for i in range(4)])
    n = np.array([500.0, 500.0, 500.0, 50.0])
    v = build_variance_vector(n, var_scale=0.5)

    A_low_tau = solve_tca_plan(D, v, marginal="unbalanced", tau=0.01)
    A_high_tau = solve_tca_plan(D, v, marginal="unbalanced", tau=100.0)

    assert np.allclose(A_low_tau.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(A_high_tau.sum(axis=1), 1.0, atol=1e-5)

    u = np.full(4, 0.25)
    dev_low = np.linalg.norm(A_low_tau.sum(axis=0) / 4.0 - u, ord=1)
    dev_high = np.linalg.norm(A_high_tau.sum(axis=0) / 4.0 - u, ord=1)
    assert dev_high < dev_low
    assert dev_high < 1e-2


def test_plan_curvature_w1_ground_metric_D():
    """6. plan_curvature computes W1 under ground metric D (0 for I, 1 for uniform consensus, 0 violations)."""
    pts = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([2.0, 2.0])]
    D = build_cost_matrix("update", deltas=pts)

    I_plan = np.eye(4, dtype=np.float64)
    st_id = plan_curvature(I_plan, D)
    assert st_id["kappa_min"] == pytest.approx(0.0, abs=1e-7)
    assert st_id["kappa_mean"] == pytest.approx(0.0, abs=1e-7)
    assert st_id["inequality_violations"] == 0
    assert st_id["sparsity"] == pytest.approx(0.75, abs=1e-7)

    U_plan = np.full((4, 4), 0.25, dtype=np.float64)
    st_u = plan_curvature(U_plan, D)
    assert st_u["kappa_min"] == pytest.approx(1.0, abs=1e-7)
    assert st_u["kappa_mean"] == pytest.approx(1.0, abs=1e-7)
    assert st_u["inequality_violations"] == 0
    assert st_u["sparsity"] == pytest.approx(0.0, abs=1e-7)


def test_estimate_q_norms_and_median():
    """7. estimate_q computes per-client contraction q = ||v2|| / ||v1|| and median q."""
    q_scalar = estimate_q(2.0, 1.5)
    assert q_scalar == pytest.approx(0.75, abs=1e-9)

    norms_v1 = [2.0, 4.0, 1.0]
    norms_v2 = [1.0, 3.0, 1.2]
    q_arr, med_q = estimate_q(norms_v1, norms_v2)
    assert np.allclose(q_arr, [0.5, 0.75, 1.2], atol=1e-9)
    assert med_q == pytest.approx(0.75, abs=1e-9)


def test_lazy_calibrate_auto_and_numeric_target():
    """8. lazy_calibrate computes beta and shifts plan curvature to target."""
    pts = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([2.0, 2.0])]
    D = build_cost_matrix("update", deltas=pts)
    v = build_variance_vector([100, 100, 100, 100], var_scale=0.3)
    A = solve_tca_plan(D, v, marginal="hard")
    st0 = plan_curvature(A, D)

    A_auto, beta_auto, target_auto = lazy_calibrate(A, median_q=2.0, kappa_target="auto", D=D, curv_stats=st0)
    assert target_auto == pytest.approx(st0["kappa_mean"], abs=1e-9)
    assert beta_auto == pytest.approx(0.5, abs=1e-9)
    assert np.allclose(A_auto.sum(axis=1), 1.0, atol=1e-7)

    desired_kappa = min(0.95, float(st0["kappa_mean"]) + 0.3 * (1.0 - float(st0["kappa_mean"])))
    A_num, beta_num, _ = lazy_calibrate(A, median_q=1.0, kappa_target=desired_kappa, D=D, curv_stats=st0)
    st_cal = plan_curvature(A_num, D)
    assert 0.0 < beta_num < 1.0
    assert st_cal["kappa_mean"] == pytest.approx(desired_kappa, abs=1e-5)

