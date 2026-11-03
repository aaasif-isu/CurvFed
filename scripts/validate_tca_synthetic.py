"""Standalone CPU synthetic validation script for Transport-Curvature Aggregation (TCA)."""

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.methods.tca import (
    build_cost_matrix,
    build_variance_vector,
    estimate_q,
    lazy_calibrate,
    plan_curvature,
    solve_tca_plan,
)


def run_two_client_feedback_check(a: float = 0.25, q: float = 0.6, rounds: int = 100):
    """2-client feedback check verifying uncalibrated over-mixing (a' ≈ 0.357, kappa ≈ 0.714)
    vs lazy_calibrate restoring the target effective curvature (a'_cal = 0.250, kappa_cal = 0.500)."""
    A = np.array([[1.0 - a, a], [a, 1.0 - a]], dtype=np.float64)
    D = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    st_A = plan_curvature(A, D)
    kappa_A = float(st_A["kappa_mean"])

    # Theoretical uncalibrated effective feedback operator: A_eff = (1 - q) * (I - q * A)^{-1} @ A
    a_prime_expected = a / (1.0 - q * (1.0 - 2.0 * a))
    kappa_eff_expected = 2.0 * a_prime_expected

    # Iterative simulation of theta^{(t+1)} = A @ (q * theta^{(t)} + (1 - q) * theta^*)
    theta_star = np.eye(2, dtype=np.float64)
    theta_uncal = theta_star.copy()
    for _ in range(rounds):
        theta_uncal = A @ (q * theta_uncal + (1.0 - q) * theta_star)
    st_uncal = plan_curvature(theta_uncal, D)
    a_prime_printed = float(theta_uncal[0, 1])
    kappa_eff_printed = float(st_uncal["kappa_mean"])

    # Calibrated feedback loop via lazy_calibrate
    A_round, beta, target = lazy_calibrate(A, median_q=q, kappa_target="auto", D=D, curv_stats=st_A)
    theta_cal = theta_star.copy()
    for _ in range(rounds):
        theta_cal = A_round @ (q * theta_cal + (1.0 - q) * theta_star)
    st_cal = plan_curvature(theta_cal, D)
    a_cal_printed = float(theta_cal[0, 1])
    kappa_cal_printed = float(st_cal["kappa_mean"])

    print("================ 2-CLIENT FEEDBACK CHECK ================")
    print(f"Single-round plan: a = {a:.3f}, kappa_A = {kappa_A:.3f}, q = {q:.3f}")
    print(
        f"Uncalibrated feedback: expected a' = {a_prime_expected:.3f}, kappa = {kappa_eff_expected:.3f} | "
        f"printed a' = {a_prime_printed:.3f}, kappa = {kappa_eff_printed:.3f}"
    )
    print(
        f"Calibrated feedback (beta = {beta:.4f}): expected a' = {a:.3f}, kappa = {target:.3f} | "
        f"printed a' = {a_cal_printed:.3f}, kappa = {kappa_cal_printed:.3f}"
    )
    print("=========================================================")


if __name__ == "__main__":
    run_two_client_feedback_check()

