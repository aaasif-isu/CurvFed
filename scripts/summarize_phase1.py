"""Collects Phase 1 experiment CSVs into a single summary table."""

import argparse
import glob
import os
from typing import List

import numpy as np
import pandas as pd


def _load_run_df(run_dir: str) -> pd.DataFrame:
    combined_csv = os.path.join(run_dir, "round_metrics.csv")
    if os.path.exists(combined_csv):
        return pd.read_csv(combined_csv)

    round_files = sorted(glob.glob(os.path.join(run_dir, "rounds", "round_*.csv")))
    if not round_files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in round_files], ignore_index=True)


def summarize_phase1(results_root: str = "results/phase1") -> pd.DataFrame:
    run_names = [
        "baseline",
        "emd_orc_geometry_per_client",
        "emd_orc_geometry_pooled",
        "emd_orc_random_pooled",
    ]
    existing_subdirs = sorted(
        d
        for d in os.listdir(results_root)
        if os.path.isdir(os.path.join(results_root, d)) and d not in run_names
    ) if os.path.exists(results_root) else []

    rows: List[dict] = []
    for name in run_names + existing_subdirs:
        run_dir = os.path.join(results_root, name)
        if not os.path.isdir(run_dir):
            continue
        df = _load_run_df(run_dir)
        if df.empty:
            continue

        df = df.sort_values("round").reset_index(drop=True)
        ari_series = pd.to_numeric(df.get("ari_prev_round", pd.Series(dtype=float)), errors="coerce")
        snr_series = pd.to_numeric(df.get("emd_snr", pd.Series(dtype=float)), errors="coerce")
        w_emd_series = pd.to_numeric(df.get("within_emd_mean", pd.Series(dtype=float)), errors="coerce")
        c_emd_series = pd.to_numeric(df.get("cross_emd_mean", pd.Series(dtype=float)), errors="coerce")
        pers_min_series = pd.to_numeric(df.get("personalized_min_acc", pd.Series(dtype=float)), errors="coerce")
        pers_mean_series = pd.to_numeric(df.get("personalized_mean_acc", pd.Series(dtype=float)), errors="coerce")

        rows.append(
            {
                "run_name": name,
                "method": str(df["method"].iloc[-1]),
                "cluster_mode": str(df["cluster_mode"].iloc[-1]) if "cluster_mode" in df else "N/A",
                "zscore_mode": str(df["zscore_mode"].iloc[-1]) if "zscore_mode" in df else "N/A",
                "rounds_completed": int(len(df)),
                "final_global_acc": float(df["global_acc_test"].iloc[-1]),
                "best_global_acc": float(df["global_acc_test"].max()),
                "mean_within_emd": float(w_emd_series.mean(skipna=True)) if not w_emd_series.isna().all() else float("nan"),
                "mean_cross_emd": float(c_emd_series.mean(skipna=True)) if not c_emd_series.isna().all() else float("nan"),
                "mean_emd_snr": float(snr_series.mean(skipna=True)) if not snr_series.isna().all() else float("nan"),
                "mean_ari_prev_round": float(ari_series.mean(skipna=True)) if not ari_series.isna().all() else float("nan"),
                "final_personalized_mean_acc": float(pers_mean_series.iloc[-1]) if not pers_mean_series.empty else float("nan"),
                "final_personalized_min_acc": float(pers_min_series.iloc[-1]) if not pers_min_series.empty else float("nan"),
                "best_personalized_min_acc": float(pers_min_series.max()) if not pers_min_series.empty else float("nan"),
            }
        )

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        os.makedirs(results_root, exist_ok=True)
        summary_path = os.path.join(results_root, "phase1_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print("\n================ PHASE 1 SUMMARY TABLE ================")
        print(summary_df.to_string(index=False))
        print(f"=======================================================\nSaved to: {summary_path}")
    else:
        print(f"No completed runs found under {results_root}.")

    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Summarize Phase 1 CurvFed runs.")
    parser.add_argument("--results-root", type=str, default="results/phase1")
    args = parser.parse_args()
    summarize_phase1(args.results_root)


if __name__ == "__main__":
    main()
