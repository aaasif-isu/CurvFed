"""Reads all `results/*/round_metrics.csv` and prints a summary table."""

import argparse
import glob
import os
from typing import List

import pandas as pd


def _col_mean(df: pd.DataFrame, col_name: str) -> float:
    if col_name not in df.columns:
        return float("nan")
    s = pd.to_numeric(df[col_name], errors="coerce")
    return float(s.mean(skipna=True)) if not s.isna().all() else float("nan")


def summarize_results(results_root: str = "results") -> pd.DataFrame:
    csv_files = sorted(glob.glob(os.path.join(results_root, "*", "round_metrics.csv")))
    rows: List[dict] = []

    for csv_path in csv_files:
        run_dir = os.path.dirname(csv_path)
        run_name = os.path.basename(run_dir)
        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        df = df.sort_values("round").reset_index(drop=True)
        global_acc = pd.to_numeric(df["global_acc_test"], errors="coerce")
        pers_mean = pd.to_numeric(df.get("personalized_mean_acc", pd.Series(dtype=float)), errors="coerce")
        pers_min = pd.to_numeric(df.get("personalized_min_acc", pd.Series(dtype=float)), errors="coerce")

        rows.append(
            {
                "run_name": run_name,
                "final_global_acc": float(global_acc.iloc[-1]) if not global_acc.empty else float("nan"),
                "best_global_acc": float(global_acc.max()) if not global_acc.empty else float("nan"),
                "personalized_mean_acc": float(pers_mean.iloc[-1]) if not pers_mean.empty else float("nan"),
                "personalized_min_acc": float(pers_min.iloc[-1]) if not pers_min.empty else float("nan"),
                "mean_kappa": _col_mean(df, "kappa_mean"),
                "median_q": _col_mean(df, "median_q"),
            }
        )

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        os.makedirs(results_root, exist_ok=True)
        summary_path = os.path.join(results_root, "summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print("\n================ RESULTS SUMMARY TABLE ================")
        print(summary_df.to_string(index=False))
        print(f"=======================================================\nSaved to: {summary_path}")
    else:
        print(f"No `results/*/round_metrics.csv` files found under {results_root}.")

    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Summarize all results/*/round_metrics.csv runs.")
    parser.add_argument("--results-root", type=str, default="results")
    args = parser.parse_args()
    summarize_results(args.results_root)


if __name__ == "__main__":
    main()
