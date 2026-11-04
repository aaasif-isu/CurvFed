"""Collects Phase 2 experiment CSVs into a single summary table (mean ± std over seeds)."""

import argparse
import glob
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

RUN_ORDER = [
    "fedavg",
    "tca_decoupled_update_hard",
    "tca_decoupled_update_none",
    "tca_feedback_uncalibrated",
    "tca_feedback_calibrated",
]




def _load_run_df(run_dir: str) -> pd.DataFrame:
    combined_csv = os.path.join(run_dir, "round_metrics.csv")
    if os.path.exists(combined_csv):
        return pd.read_csv(combined_csv)

    round_files = sorted(glob.glob(os.path.join(run_dir, "rounds", "round_*.csv")))
    if not round_files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in round_files], ignore_index=True)


def _extract_seed_metrics(df: pd.DataFrame) -> Optional[Dict[str, float]]:
    if df.empty:
        return None
    df = df.sort_values("round").reset_index(drop=True)

    def _col_mean(col_name: str) -> float:
        if col_name not in df.columns:
            return float("nan")
        s = pd.to_numeric(df[col_name], errors="coerce")
        return float(s.mean(skipna=True)) if not s.isna().all() else float("nan")

    global_acc = pd.to_numeric(df["global_acc_test"], errors="coerce")
    pers_mean = pd.to_numeric(df.get("personalized_mean_acc", pd.Series(dtype=float)), errors="coerce")
    pers_min = pd.to_numeric(df.get("personalized_min_acc", pd.Series(dtype=float)), errors="coerce")

    return {
        "final_global_acc": float(global_acc.iloc[-1]) if not global_acc.empty else float("nan"),
        "best_global_acc": float(global_acc.max()) if not global_acc.empty else float("nan"),
        "personalized_mean_acc": float(pers_mean.iloc[-1]) if not pers_mean.empty else float("nan"),
        "personalized_min_acc": float(pers_min.iloc[-1]) if not pers_min.empty else float("nan"),
        "mean_kappa": _col_mean("kappa_mean"),
        "sparsity": _col_mean("sparsity"),
        "median_q": _col_mean("median_q"),
    }


def _fmt_mean_std(vals: List[float], decimals: int = 2) -> str:
    clean = [v for v in vals if not np.isnan(v)]
    if not clean:
        return "N/A"
    mean_v = float(np.mean(clean))
    std_v = float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0
    return f"{mean_v:.{decimals}f} ± {std_v:.{decimals}f}"


def summarize_phase2(results_root: str = "results/phase2") -> pd.DataFrame:
    if not os.path.exists(results_root):
        print(f"No results directory found at {results_root}.")
        return pd.DataFrame()

    alpha_dirs = sorted(
        d
        for d in os.listdir(results_root)
        if os.path.isdir(os.path.join(results_root, d)) and d.startswith("alpha_")
    )
    if not alpha_dirs:
        alpha_dirs = ["."]

    rows: List[dict] = []
    for alpha_dir in alpha_dirs:
        alpha_path = results_root if alpha_dir == "." else os.path.join(results_root, alpha_dir)
        alpha_label = alpha_dir.replace("alpha_", "") if alpha_dir != "." else "all"

        existing_runs = [
            d for d in os.listdir(alpha_path) if os.path.isdir(os.path.join(alpha_path, d))
        ]
        ordered_runs = [r for r in RUN_ORDER if r in existing_runs] + sorted(
            r for r in existing_runs if r not in RUN_ORDER
        )

        for run_name in ordered_runs:
            run_root = os.path.join(alpha_path, run_name)
            seed_dirs = sorted(
                os.path.join(run_root, d)
                for d in os.listdir(run_root)
                if os.path.isdir(os.path.join(run_root, d)) and d.startswith("seed_")
            )
            if not seed_dirs:
                seed_dirs = [run_root]

            seed_stats: List[Dict[str, float]] = []
            for sdir in seed_dirs:
                df = _load_run_df(sdir)
                st = _extract_seed_metrics(df)
                if st is not None:
                    seed_stats.append(st)

            if not seed_stats:
                continue

            rows.append(
                {
                    "alpha": alpha_label,
                    "run_name": run_name,
                    "num_seeds": len(seed_stats),
                    "final_global_acc": _fmt_mean_std([s["final_global_acc"] for s in seed_stats], 2),
                    "best_global_acc": _fmt_mean_std([s["best_global_acc"] for s in seed_stats], 2),
                    "personalized_mean_acc": _fmt_mean_std(
                        [s["personalized_mean_acc"] for s in seed_stats], 2
                    ),
                    "personalized_min_acc": _fmt_mean_std(
                        [s["personalized_min_acc"] for s in seed_stats], 2
                    ),
                    "mean_kappa": _fmt_mean_std([s["mean_kappa"] for s in seed_stats], 4),
                    "sparsity": _fmt_mean_std([s["sparsity"] for s in seed_stats], 4),
                    "median_q": _fmt_mean_std([s["median_q"] for s in seed_stats], 4),
                }
            )

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        os.makedirs(results_root, exist_ok=True)
        summary_path = os.path.join(results_root, "phase2_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print("\n================ PHASE 2 SUMMARY TABLE (mean ± std over seeds) ================")
        print(summary_df.to_string(index=False))
        print(f"===============================================================================\nSaved to: {summary_path}")
    else:
        print(f"No completed Phase 2 runs found under {results_root}.")

    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Summarize Phase 2 TCA runs across seeds.")
    parser.add_argument("--results-root", type=str, default="results/phase2")
    args = parser.parse_args()
    summarize_phase2(args.results_root)


if __name__ == "__main__":
    main()
