#!/usr/bin/env python3
"""Plot the five methods for MNIST + ResNet18, 10-client experiment."""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


# Put this script directly inside 10_Clients_Results/Results_mnist.
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT
OUTPUT_DIR = PROJECT_ROOT

# Excel folder, legend label, line color, line width
METHODS = [
    ("Baseline", "Baseline", "#1f77b4", 2.5),
    ("EMD_only", "EMD-only", "#ff7f0e", 2.4),
    ("EMD+ORC", "EMD + ORC", "#2ca02c", 2.4),
    ("EMD+ORC_weighted", "Weighted EMD + ORC", "#d62728", 3.0),
    ("FedWaD", "FedWaD", "#9467bd", 2.4),
]


def get_single_excel(folder: Path) -> Path:
    """Return the only Excel workbook in a method folder."""
    files = sorted(
        file for file in folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in {".xlsx", ".xls"}
        and not file.name.startswith("~$")
    )

    if not files:
        raise FileNotFoundError(f"No Excel file found in: {folder}")

    if len(files) > 1:
        names = "\n  - ".join(file.name for file in files)
        raise RuntimeError(
            f"Multiple Excel files found in {folder}. Keep only the correct "
            f"result file before plotting:\n  - {names}"
        )

    return files[0]


def read_test_accuracy(file: Path, is_baseline: bool):
    """Read round and comparable global test accuracy from one workbook."""
    workbook = pd.ExcelFile(file)
    sheet = "v1_test" if "v1_test" in workbook.sheet_names else workbook.sheet_names[0]
    data = pd.read_excel(file, sheet_name=sheet)
    data.columns = [str(column).strip().lower() for column in data.columns]

    if "round" not in data.columns:
        raise ValueError(f"Missing 'round' column in {file}")

    # Baseline workbooks may include both average client test accuracy and
    # independently evaluated global-model accuracy. Use global_acc_test so
    # it is comparable with acc_test in the clustering methods.
    if is_baseline and "global_acc_test" in data.columns:
        accuracy_column = "global_acc_test"
    elif "acc_test" in data.columns:
        accuracy_column = "acc_test"
    elif "global_acc_test" in data.columns:
        accuracy_column = "global_acc_test"
    else:
        raise ValueError(
            f"No 'acc_test' or 'global_acc_test' column found in {file}. "
            f"Available columns: {list(data.columns)}"
        )

    clean = data[["round", accuracy_column]].copy()
    clean.columns = ["round", "test_accuracy"]
    clean["round"] = pd.to_numeric(clean["round"], errors="coerce")
    clean["test_accuracy"] = pd.to_numeric(
        clean["test_accuracy"], errors="coerce"
    )
    clean = clean.dropna().sort_values("round")

    if clean.empty:
        raise ValueError(f"No numeric round/accuracy data found in {file}")
    if clean["round"].duplicated().any():
        raise ValueError(f"Duplicate global-round values found in {file}")
    if not clean["test_accuracy"].between(0, 100).all():
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

    return clean, alpha, accuracy_column


def main():
    if not RESULTS_DIR.is_dir():
        raise FileNotFoundError(
            f"Results folder not found: {RESULTS_DIR}\n"
            "Place this script inside 10_Clients_Results/Results_mnist."
        )

    all_results = []
    alpha_values = set()

    for folder_name, label, color, width in METHODS:
        method_dir = RESULTS_DIR / folder_name
        if not method_dir.is_dir():
            raise FileNotFoundError(f"Required method folder not found: {method_dir}")

        excel_file = get_single_excel(method_dir)
        result, alpha, column = read_test_accuracy(
            excel_file,
            is_baseline=(folder_name == "Baseline"),
        )
        all_results.append((label, color, width, result))

        if alpha is not None:
            alpha_values.add(alpha)

        print(f"{label}: {excel_file.name}")
        print(
            f"  column={column}, rounds={len(result)}, "
            f"final_accuracy={result.iloc[-1]['test_accuracy']:.3f}%"
        )

    if len(alpha_values) > 1:
        raise RuntimeError(
            f"The Excel files have different alpha values: {sorted(alpha_values)}"
        )

    alpha = next(iter(alpha_values), None)
    if alpha is None:
        alpha_label = "α not recorded"
        alpha_filename = "unknown"
    elif alpha >= 999999:
        alpha_label = "IID"
        alpha_filename = "iid"
    else:
        alpha_label = f"Non-IID α={alpha:g}"
        alpha_filename = str(alpha).replace(".", "p")

    figure, axis = plt.subplots(figsize=(12, 7), dpi=160)

    for label, color, width, result in all_results:
        axis.plot(
            result["round"],
            result["test_accuracy"],
            label=label,
            color=color,
            linewidth=width,
        )

    axis.set_title(
        "SplitFed Method Comparison on MNIST\n"
        f"{alpha_label}, 10 Clients, Seed=42, ResNet18",
        fontsize=15,
    )
    axis.set_xlabel("Global Round", fontsize=12)
    axis.set_ylabel("Global Test Accuracy (%)", fontsize=12)
    axis.set_xlim(left=1)
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend(loc="best", frameon=True)
    figure.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"mnist_resnet18_10clients_alpha{alpha_filename}_comparison"
    png_file = OUTPUT_DIR / f"{stem}.png"
    pdf_file = OUTPUT_DIR / f"{stem}.pdf"
    figure.savefig(png_file, bbox_inches="tight")
    figure.savefig(pdf_file, bbox_inches="tight")
    plt.close(figure)

    print(f"\nSaved PNG: {png_file}")
    print(f"Saved PDF: {pdf_file}")


if __name__ == "__main__":
    main()
