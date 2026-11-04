#!/usr/bin/env python3
"""Plot MNIST + ResNet18 results for 100 clients (10 active per round)."""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


# Keep this file inside 100_Clients_Results/Results_mnist.
ROOT = Path(__file__).resolve().parent

METHODS = [
    (("Baseline",), "Baseline", "#1f77b4", 2.5),
    (("EMD", "EMD_only"), "EMD-only", "#ff7f0e", 2.4),
    (("EMD+ORC",), "EMD + ORC", "#2ca02c", 2.4),
    (("EMD+ORC_weighted", "EMD+ORC_Weighted"),
     "Weighted EMD + ORC", "#d62728", 3.0),
    (("FedWaD",), "FedWaD", "#9467bd", 2.4),
]


def method_folder(candidates):
    for name in candidates:
        folder = ROOT / name
        if folder.is_dir():
            return folder
    raise FileNotFoundError(
        f"Missing method folder; expected one of: {', '.join(candidates)}"
    )


def single_excel(folder):
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
            f"Multiple Excel files found in {folder}. Keep only the correct run:\n"
            f"  - {names}"
        )
    return files[0]


def read_excel_result(file, baseline=False):
    excel = pd.ExcelFile(file)
    sheet = "v1_test" if "v1_test" in excel.sheet_names else excel.sheet_names[0]
    data = pd.read_excel(file, sheet_name=sheet)
    data.columns = [str(column).strip().lower() for column in data.columns]

    if "round" not in data.columns:
        raise ValueError(f"Missing 'round' column in {file}")

    if baseline and "global_acc_test" in data.columns:
        accuracy_column = "global_acc_test"
    elif "acc_test" in data.columns:
        accuracy_column = "acc_test"
    elif "global_acc_test" in data.columns:
        accuracy_column = "global_acc_test"
    else:
        raise ValueError(
            f"No comparable test-accuracy column in {file}. "
            f"Columns: {list(data.columns)}"
        )

    result = data[["round", accuracy_column]].copy()
    result.columns = ["round", "test_accuracy"]
    result["round"] = pd.to_numeric(result["round"], errors="coerce")
    result["test_accuracy"] = pd.to_numeric(
        result["test_accuracy"], errors="coerce"
    )
    result = result.dropna().sort_values("round")

    if result.empty:
        raise ValueError(f"No numeric result rows in {file}")
    if result["round"].duplicated().any():
        raise ValueError(f"Duplicate round values in {file}")
    if not result["test_accuracy"].between(0, 100).all():
        raise ValueError(f"Accuracy outside 0–100% in {file}")

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
    alphas = set()
    client_counts = set()

    for folders, label, color, width in METHODS:
        file = single_excel(method_folder(folders))
        result, alpha, users, column = read_excel_result(
            file, baseline=(label == "Baseline")
        )
        curves.append((label, color, width, result))
        if alpha is not None:
            alphas.add(alpha)
        if users is not None:
            client_counts.add(users)
        print(f"{label}: {file}")
        print(
            f"  column={column}, rounds={len(result)}, "
            f"final_accuracy={result.iloc[-1]['test_accuracy']:.3f}%"
        )

    if len(alphas) > 1:
        raise RuntimeError(f"Different alpha values found: {sorted(alphas)}")
    if client_counts and client_counts != {100}:
        raise RuntimeError(f"Expected clients=100; found {sorted(client_counts)}")

    alpha = next(iter(alphas), None)
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
            result["round"], result["test_accuracy"],
            label=label, color=color, linewidth=width,
        )

    axis.set_title(
        "SplitFed Method Comparison on MNIST\n"
        f"{alpha_label}, 100 Clients, 10 Active/Round, Seed=42, ResNet18",
        fontsize=15,
    )
    axis.set_xlabel("Global Round", fontsize=12)
    axis.set_ylabel("Global Test Accuracy (%)", fontsize=12)
    axis.set_xlim(left=1)
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend(loc="best", frameon=True)
    figure.tight_layout()

    stem = f"mnist_resnet18_100clients_alpha{alpha_file}_comparison"
    png = ROOT / f"{stem}.png"
    pdf = ROOT / f"{stem}.pdf"
    figure.savefig(png, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    print(f"\nSaved PNG: {png}")
    print(f"Saved PDF: {pdf}")


if __name__ == "__main__":
    main()
