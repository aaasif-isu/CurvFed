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
    """7. estimate_q computes per-client contraction q = (clip(n2 / n1, 0, 0.999))^2 and median q."""
    q_scalar = estimate_q(2.0, 1.5)
    assert q_scalar == pytest.approx(0.75 ** 2, abs=1e-9)

    norms_v1 = [2.0, 4.0, 1.0]
    norms_v2 = [1.0, 3.0, 1.2]
    q_arr, med_q, raw_arr = estimate_q(norms_v1, norms_v2, return_raw=True)
    assert np.allclose(raw_arr, [0.5, 0.75, 1.2], atol=1e-9)
    assert np.allclose(q_arr, [0.25, 0.5625, 0.999 ** 2], atol=1e-9)
    assert med_q == pytest.approx(0.5625, abs=1e-9)


def test_lazy_calibrate_auto_and_numeric_target():
    """8. lazy_calibrate mixes toward identity I according to kappa_round / kappa_A."""
    pts = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([2.0, 2.0])]
    D = build_cost_matrix("update", deltas=pts)
    v = build_variance_vector([100, 100, 100, 100], var_scale=0.3)
    A = solve_tca_plan(D ** 2, v, marginal="hard")
    st0 = plan_curvature(A, D)
    kappa_A = float(st0["kappa_mean"])

    q = 0.5
    A_auto, beta_auto, target_auto = lazy_calibrate(A, median_q=q, kappa_target="auto", D=D, curv_stats=st0)
    expected_beta = (1.0 - q) / (1.0 - q * kappa_A)
    assert target_auto == pytest.approx(kappa_A, abs=1e-9)
    assert beta_auto == pytest.approx(expected_beta, abs=1e-9)
    assert np.allclose(A_auto, (1.0 - beta_auto) * np.eye(4) + beta_auto * A, atol=1e-9)
    assert np.allclose(A_auto.sum(axis=1), 1.0, atol=1e-7)

    desired_kappa = 0.5 * kappa_A
    A_num, beta_num, _ = lazy_calibrate(A, median_q=0.0, kappa_target=desired_kappa, D=D, curv_stats=st0)
    st_cal = plan_curvature(A_num, D)
    assert beta_num == pytest.approx(0.5, abs=1e-7)
    assert st_cal["kappa_mean"] == pytest.approx(desired_kappa, abs=1e-5)


def test_spec_c_zero_hard_nonuniform_rho():
    """9. C = 0, hard marginal, non-uniform rho, v_k = 1/rho_k -> every row of A equals rho (atol 1e-4)."""
    rho = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    v = 1.0 / rho
    C = np.zeros((4, 4), dtype=np.float64)

    A = solve_tca_plan(C, v=v, rho=rho, marginal="hard")
    for i in range(4):
        assert np.allclose(A[i], rho, atol=1e-4)


def test_spec_uniform_rho_small_var_scale_identity():
    """10. uniform rho, var_scale = 1e-6, positive off-diagonal C -> A ≈ I."""
    m = 4
    rho = np.full(m, 1.0 / m, dtype=np.float64)
    v = np.full(m, 1e-6, dtype=np.float64)
    C = np.ones((m, m), dtype=np.float64) - np.eye(m, dtype=np.float64)

    A = solve_tca_plan(C, v=v, rho=rho, marginal="hard")
    assert np.allclose(A, np.eye(m), atol=1e-4)


def test_spec_block_diagonal_cost():
    """11. block-diagonal C (off-block cost 100), uniform rho -> off-block mass < 1e-4."""
    C = np.array(
        [
            [0.0, 0.1, 100.0, 100.0],
            [0.1, 0.0, 100.0, 100.0],
            [100.0, 100.0, 0.0, 0.1],
            [100.0, 100.0, 0.1, 0.0],
        ],
        dtype=np.float64,
    )
    rho = np.full(4, 0.25, dtype=np.float64)
    v = np.full(4, 0.5, dtype=np.float64)

    A = solve_tca_plan(C, v=v, rho=rho, marginal="hard")
    off_block_mask = np.array(
        [
            [False, False, True, True],
            [False, False, True, True],
            [True, True, False, False],
            [True, True, False, False],
        ]
    )
    assert float(np.sum(A[off_block_mask])) < 1e-4


def test_spec_three_client_closed_form_and_curvature():
    """12. theta* = [0, 0.1, 1.0], D = |theta_i - theta_k|, C = D**2, v = 0.05, uniform rho, hard ->
    A ≈ [[0.55,0.45,0],[0.45,0.55,0],[0,0,1]] (atol 1e-3), kappa(0,1) ≈ 0.9, b = [0.045, 0.045, 0],
    inequality_violations == 0."""
    theta_star = np.array([0.0, 0.1, 1.0], dtype=np.float64)
    D = np.abs(theta_star[:, None] - theta_star[None, :])
    C = D ** 2
    rho = np.full(3, 1.0 / 3.0, dtype=np.float64)

    A = solve_tca_plan(C, v=0.05, rho=rho, marginal="hard")
    expected_A = np.array(
        [
            [0.55, 0.45, 0.0],
            [0.45, 0.55, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    assert np.allclose(A, expected_A, atol=1e-3)

    curv = plan_curvature(A, D)
    assert curv["kappa_matrix"][0, 1] == pytest.approx(0.9, abs=1e-3)
    assert np.allclose(curv["b"], [0.045, 0.045, 0.0], atol=1e-4)
    assert curv["inequality_violations"] == 0


def test_spec_estimate_q_0p81():
    """13. estimate_q(1.0, 0.9) == 0.81."""
    assert estimate_q(1.0, 0.9) == pytest.approx(0.81, abs=1e-12)


def test_spec_lazy_calibrate_cases():
    """14. lazy_calibrate: q=0 -> A_round == A; q=0.999 -> A_round ≈ I;
    2-client A=[[0.9,0.1],[0.1,0.9]], q=0.81, auto -> beta ≈ 0.19 / (1 - 0.81*0.2) = 0.2267."""
    A = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64)
    D = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)

    A_q0, beta_q0, _ = lazy_calibrate(A, median_q=0.0, kappa_target="auto", D=D)
    assert beta_q0 == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(A_q0, A, atol=1e-9)

    A_q999, beta_q999, _ = lazy_calibrate(A, median_q=0.999, kappa_target="auto", D=D)
    assert beta_q999 < 2e-3
    assert np.allclose(A_q999, np.eye(2), atol=2e-3)

    A_q81, beta_q81, target_q81 = lazy_calibrate(A, median_q=0.81, kappa_target="auto", D=D)
    expected_beta = 0.19 / (1.0 - 0.81 * 0.2)
    assert target_q81 == pytest.approx(0.2, abs=1e-9)
    assert beta_q81 == pytest.approx(expected_beta, abs=1e-6)
    assert beta_q81 == pytest.approx(0.2267, abs=1e-4)
    assert np.allclose(A_q81, (1.0 - expected_beta) * np.eye(2) + expected_beta * A, atol=1e-9)



