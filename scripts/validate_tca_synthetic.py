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


def run_synthetic_tca_benchmark(
    num_clients: int = 12,
    dim: int = 32,
    num_trials: int = 50,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    n_counts = np.array([400, 350, 300, 250] * (num_clients // 4), dtype=np.float64)
    cluster_ids = np.repeat([0, 1, 2], num_clients // 3)
    centers = rng.normal(0.0, 3.0, size=(3, dim))
    true_client_means = np.array([centers[c] for c in cluster_ids], dtype=np.float64)

    methods = [
        ("fedavg", None),
        ("tca_hard", ("hard", 0.5, 1.0, False)),
        ("tca_unbalanced", ("unbalanced", 0.5, 1.0, False)),
        ("tca_none", ("none", 0.5, 1.0, False)),
        ("tca_calibrated", ("hard", 0.5, 1.0, True)),
    ]
    errors = {name: [] for name, _ in methods}

    for _ in range(num_trials):
        noise_scale = 4.0 / np.sqrt(n_counts)[:, None]
        observed = true_client_means + rng.normal(size=(num_clients, dim)) * noise_scale
        D = build_cost_matrix("update", deltas=[observed[i] for i in range(num_clients)])

        p_weights = n_counts / n_counts.sum()
        fedavg_vec = p_weights @ observed
        errors["fedavg"].append(
            float(np.mean(np.linalg.norm(fedavg_vec[None, :] - true_client_means, axis=1)))
        )

        for name, cfg in methods[1:]:
            marg, var_scale, tau, do_cal = cfg
            v = build_variance_vector(n_counts, var_scale=var_scale)
            A = solve_tca_plan(D, v, marginal=marg, tau=tau)
            if do_cal:
                A, _, _ = lazy_calibrate(A, median_q=1.25, kappa_target="auto", D=D)
            est = A @ observed
            errors[name].append(float(np.mean(np.linalg.norm(est - true_client_means, axis=1))))

    print("================ TCA SYNTHETIC BENCHMARK ================")
    for name, _ in methods:
        arr = np.asarray(errors[name], dtype=np.float64)
        print(f"{name:<18} | Mean L2 Error: {arr.mean():.4f} ± {arr.std(ddof=1):.4f}")
    print("=========================================================")


if __name__ == "__main__":
    run_synthetic_tca_benchmark()
