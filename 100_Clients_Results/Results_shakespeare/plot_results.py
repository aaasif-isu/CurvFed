#!/usr/bin/env python3
"""Plot Shakespeare + CharLSTM results for 100 clients, 10 active/round."""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


# Keep this script inside 100_Clients_Results/Results_shakespeare.
ROOT = Path(__file__).resolve().parent

METHODS = [
    (("Baseline",), "Baseline", "#1f77b4", 2.5),
    (("EMD", "EMD_only"), "EMD-only", "#ff7f0e", 2.4),
    (("EMD+ORC",), "EMD + ORC", "#2ca02c", 2.4),
    (("EMD+ORC_weighted", "EMD+ORC_Weighted"),
     "Weighted EMD + ORC", "#d62728", 3.0),
    (("FedWaD",), "FedWaD", "#9467bd", 2.4),
]


def find_method_folder(candidates):
    for name in candidates:
        folder = ROOT / name
        if folder.is_dir():
            return folder
    raise FileNotFoundError(
        f"Missing method folder. Expected one of: {', '.join(candidates)}"
    )


def find_single_excel(folder):
    # Recursive search supports Excel files inside nested experiment folders.
    files = sorted(
        file for file in folder.rglob("*")
        if file.is_file()
        and file.suffix.lower() in {".xlsx", ".xls"}
        and not file.name.startswith("~$")
    )

    if not files:
        raise FileNotFoundError(f"No Excel file found in: {folder}")

    if len(files) > 1:
        names = "\n  - ".join(str(file.relative_to(folder)) for file in files)
        raise RuntimeError(
            f"Multiple Excel files found in {folder}. Keep only the correct "
            f"result file:\n  - {names}"
        )

    return files[0]


def read_result(file, is_baseline):
    workbook = pd.ExcelFile(file)
    sheet = "v1_test" if "v1_test" in workbook.sheet_names else workbook.sheet_names[0]
    data = pd.read_excel(file, sheet_name=sheet)
    data.columns = [str(column).strip().lower() for column in data.columns]

    if "round" not in data.columns:
        raise ValueError(f"Missing 'round' column in {file}")

    # Baseline has a separately evaluated global-model metric. Use it instead
    # of original_acc_test so all five lines are directly comparable.
    if is_baseline and "global_acc_test" in data.columns:
        accuracy_column = "global_acc_test"
    elif "acc_test" in data.columns:
        accuracy_column = "acc_test"
    elif "global_acc_test" in data.columns:
        accuracy_column = "global_acc_test"
    else:
        raise ValueError(
            f"No 'acc_test' or 'global_acc_test' column in {file}. "
            f"Available columns: {list(data.columns)}"
        )

    result = data[["round", accuracy_column]].copy()
    result.columns = ["round", "test_accuracy"]
    result["round"] = pd.to_numeric(result["round"], errors="coerce")
    result["test_accuracy"] = pd.to_numeric(
        result["test_accuracy"], errors="coerce"
    )
    result = result.dropna().sort_values("round")

    if result.empty:
        raise ValueError(f"No numeric round/accuracy rows in {file}")
    if result["round"].duplicated().any():
        raise ValueError(f"Duplicate round values in {file}")
    if not result["test_accuracy"].between(0, 100).all():
        raise ValueError(f"Accuracy outside the 0–100% range in {file}")

    alpha = None
    if "alpha" in data.columns:
        values = pd.to_numeric(data["alpha"], errors="coerce").dropna().unique()
        if len(values) == 1:
            alpha = float(values[0])
    if alpha is None:
        match = re.search(r"alpha[_-]?(999999|\d+(?:\.\d+)?)", file.name, re.I)
        if match:
            alpha = float(match.group(1))

    users = None
    if "num_users" in data.columns:
        values = pd.to_numeric(data["num_users"], errors="coerce").dropna().unique()
        if len(values) == 1:
            users = int(values[0])
    if users is None:
        match = re.search(r"clients[_-]?(\d+)", file.name, re.I)
        if match:
            users = int(match.group(1))

    return result, alpha, users, accuracy_column


def main():
    curves = []
    alpha_values = set()
    client_values = set()

    for folder_names, label, color, width in METHODS:
        folder = find_method_folder(folder_names)
        excel_file = find_single_excel(folder)
        result, alpha, users, column = read_result(
            excel_file, is_baseline=(label == "Baseline")
        )
        curves.append((label, color, width, result))

        if alpha is not None:
            alpha_values.add(alpha)
        if users is not None:
            client_values.add(users)

        print(f"{label}: {excel_file}")
        print(
            f"  column={column}, rounds={len(result)}, "
            f"final_accuracy={result.iloc[-1]['test_accuracy']:.3f}%"
        )

    if len(alpha_values) > 1:
        raise RuntimeError(
            f"Excel files contain different alpha values: {sorted(alpha_values)}"
        )
    if client_values and client_values != {100}:
        raise RuntimeError(
            f"Expected 100-client files, but found: {sorted(client_values)}"
        )

    alpha = next(iter(alpha_values), None)
    if alpha is None:
        alpha_label, alpha_file = "α not recorded", "unknown"
    elif alpha >= 999999:
        alpha_label, alpha_file = "IID", "iid"
    else:
        alpha_label = f"Non-IID α={alpha:g}"
        alpha_file = str(alpha).replace(".", "p")

    figure, axis = plt.subplots(figsize=(12, 7), dpi=160)

    for label, color, width, result in curves:
        axis.plot(
            result["round"],
            result["test_accuracy"],
            label=label,
            color=color,
            linewidth=width,
        )

    axis.set_title(
        "SplitFed Method Comparison on Shakespeare\n"
        f"{alpha_label}, 100 Clients, 10 Active/Round, Seed=42, CharLSTM",
        fontsize=15,
    )
    axis.set_xlabel("Global Round", fontsize=12)
    axis.set_ylabel("Global Test Accuracy (%)", fontsize=12)
    axis.set_xlim(left=1)
    # Automatic Y-axis scaling matches the other comparison figures.
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend(loc="best", frameon=True)
    figure.tight_layout()

    stem = f"shakespeare_charlstm_100clients_alpha{alpha_file}_comparison"
    png_file = ROOT / f"{stem}.png"
    pdf_file = ROOT / f"{stem}.pdf"
    figure.savefig(png_file, bbox_inches="tight")
    figure.savefig(pdf_file, bbox_inches="tight")
    plt.close(figure)

    print(f"\nSaved PNG: {png_file}")
    print(f"Saved PDF: {pdf_file}")


if __name__ == "__main__":
    main()
