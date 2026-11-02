"""Plot CIFAR-10 IID SplitFed results for the 100-client experiment.

Place this file as:
    IID_Results_100_clients/plot.py

Expected method directories:
    Baseline/
    EMD/
    EMD+ORC/
    EMD+ORC+Weighted/
    FedWaD/

Each available directory should contain exactly one .xlsx workbook. Accuracy is
required. Test loss is optional because some experiment scripts did not save it.
"""

from pathlib import Path

import matplotlib

# Required when plotting on Nova without a graphical display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_ROOT = Path(__file__).resolve().parent
PLOTS_DIR = RESULTS_ROOT / "Plots"
EXPECTED_ROUNDS = 50


# Plot label -> result directory
METHODS = {
    "Baseline": "Baseline",
    "EMD only": "EMD",
    "EMD + ORC": "EMD+ORC",
    "Weighted EMD + ORC": "EMD+ORC+Weighted",
    "FedWaD only": "FedWaD",
}


COLORS = {
    "Baseline": "#4C78A8",
    "EMD only": "#F58518",
    "EMD + ORC": "#54A24B",
    "Weighted EMD + ORC": "#E45756",
    "FedWaD only": "#9467BD",
}


def find_excel_file(directory: Path) -> Path | None:
    """Return the single Excel workbook in a directory, if available."""

    if not directory.exists():
        print(f"Warning: directory is missing; skipping method:\n  {directory}")
        return None

    files = sorted(
        path
        for path in directory.glob("*.xlsx")
        if not path.name.startswith("~$")
    )

    if not files:
        print(f"Warning: no .xlsx workbook found; skipping method:\n  {directory}")
        return None

    if len(files) > 1:
        raise RuntimeError(
            f"Expected exactly one .xlsx workbook in {directory}, "
            f"but found {len(files)}:\n"
            + "\n".join(str(path) for path in files)
        )

    return files[0]


def load_method(label: str, directory_name: str) -> pd.DataFrame | None:
    """Load one experiment workbook and normalize its column names.

    Accuracy is required. Loss is optional and becomes NaN when unavailable.
    """

    directory = RESULTS_ROOT / directory_name
    excel_file = find_excel_file(directory)

    if excel_file is None:
        return None

    print(f"Loading {label}:")
    print(f"  {excel_file.name}")

    frame = pd.read_excel(excel_file)

    if label == "Baseline":
        accuracy_candidates = ["global_acc_test", "acc_test"]
        loss_candidates = ["global_loss_test", "loss_test"]
    else:
        accuracy_candidates = ["acc_test", "global_acc_test"]
        loss_candidates = ["loss_test", "global_loss_test"]

    accuracy_column = next(
        (name for name in accuracy_candidates if name in frame.columns),
        None,
    )
    loss_column = next(
        (name for name in loss_candidates if name in frame.columns),
        None,
    )

    missing_required = []
    if "round" not in frame.columns:
        missing_required.append("round")
    if accuracy_column is None:
        missing_required.append(
            f"one of {accuracy_candidates}"
        )

    if missing_required:
        raise RuntimeError(
            f"{excel_file} is missing required columns:\n"
            f"{missing_required}\n\n"
            f"Available columns:\n{list(frame.columns)}"
        )

    normalized = pd.DataFrame()
    normalized["round"] = pd.to_numeric(
        frame["round"], errors="coerce"
    )
    normalized["accuracy"] = pd.to_numeric(
        frame[accuracy_column], errors="coerce"
    )

    if loss_column is not None:
        normalized["loss"] = pd.to_numeric(
            frame[loss_column], errors="coerce"
        )
    else:
        normalized["loss"] = float("nan")
        print(
            "  Warning: no test-loss column was saved. "
            "Accuracy will still be plotted."
        )

    required_values = normalized[["round", "accuracy"]]
    if required_values.isna().any().any():
        bad_rows = normalized[
            required_values.isna().any(axis=1)
        ]
        raise RuntimeError(
            f"{excel_file} contains missing or non-numeric round/accuracy "
            f"values:\n{bad_rows}"
        )

    normalized["round"] = normalized["round"].astype(int)
    normalized = normalized.sort_values("round").reset_index(drop=True)

    if len(normalized) != EXPECTED_ROUNDS:
        raise RuntimeError(
            f"{excel_file} contains {len(normalized)} rows; "
            f"expected {EXPECTED_ROUNDS}."
        )

    expected = list(range(1, EXPECTED_ROUNDS + 1))
    actual = normalized["round"].tolist()

    if actual != expected:
        raise RuntimeError(
            f"{excel_file} does not contain rounds 1-{EXPECTED_ROUNDS} "
            f"exactly once and in order after sorting.\n"
            f"Found rounds: {actual}"
        )

    return normalized


def save_line_plot(
    results: dict[str, pd.DataFrame],
    metric: str,
    ylabel: str,
    filename: str,
    subtitle: str,
) -> None:
    """Create and save a round-by-round line plot."""

    fig, ax = plt.subplots(figsize=(11, 6.5))

    for label, frame in results.items():
        ax.plot(
            frame["round"],
            frame[metric],
            label=label,
            color=COLORS[label],
            linewidth=2.2,
        )

    ax.set_title(
        f"CIFAR-10 SplitFed: {ylabel} by Global Round\n{subtitle}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Global communication round", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlim(1, EXPECTED_ROUNDS)
    ax.set_xticks(range(0, EXPECTED_ROUNDS + 1, 5))

    if metric == "accuracy":
        maximum = max(frame[metric].max() for frame in results.values())
        ax.set_ylim(0, max(35, maximum + 3))

    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.3)
    ax.legend(frameon=False, ncol=2, fontsize=10)
    fig.tight_layout()

    png_path = PLOTS_DIR / f"{filename}.png"
    pdf_path = PLOTS_DIR / f"{filename}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Created: {png_path.name}")
    print(f"Created: {pdf_path.name}")


def save_final_accuracy_plot(results: dict[str, pd.DataFrame]) -> None:
    """Create a bar chart comparing round-50 test accuracy."""

    labels = list(results.keys())
    values = [results[label]["accuracy"].iloc[-1] for label in labels]
    colors = [COLORS[label] for label in labels]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars = ax.bar(
        labels,
        values,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
    )

    ax.set_title(
        "Final Test Accuracy at Round 50\n"
        "CIFAR-10, 100 clients, IID partition, seed 42",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel("Final test accuracy (%)", fontsize=12)
    ax.set_ylim(0, max(values) + 4)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.3)
    ax.tick_params(axis="x", rotation=15)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.35,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    fig.tight_layout()

    png_path = PLOTS_DIR / "iid_100clients_final_accuracy.png"
    pdf_path = PLOTS_DIR / "iid_100clients_final_accuracy.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Created: {png_path.name}")
    print(f"Created: {pdf_path.name}")


def save_summary(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Save final and recent-round metrics as a CSV table."""

    rows = []

    for label, frame in results.items():
        best_index = frame["accuracy"].idxmax()
        final_loss = (
            frame["loss"].iloc[-1]
            if frame["loss"].notna().any()
            else float("nan")
        )

        rows.append(
            {
                "method": label,
                "final_accuracy": frame["accuracy"].iloc[-1],
                "final_loss": final_loss,
                "last_10_round_accuracy_mean": frame["accuracy"].tail(10).mean(),
                "last_10_round_accuracy_std": frame["accuracy"].tail(10).std(),
                "best_accuracy": frame["accuracy"].max(),
                "best_accuracy_round": int(frame.loc[best_index, "round"]),
            }
        )

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        by="final_accuracy", ascending=False
    ).reset_index(drop=True)

    summary_path = PLOTS_DIR / "iid_100clients_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Created: {summary_path.name}")

    return summary


def main() -> None:
    """Load available workbooks and create all valid plots."""

    print("=" * 70)
    print("CIFAR-10 IID — 100-client SplitFed result plotting")
    print("=" * 70)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, pd.DataFrame] = {}

    for label, directory_name in METHODS.items():
        frame = load_method(label, directory_name)
        if frame is not None:
            results[label] = frame

    if not results:
        raise RuntimeError("No valid experiment workbooks were found.")

    print("\nCreating plots...\n")

    common_subtitle = (
        "100 total clients, 10 selected per round, "
        "IID partition, seed 42"
    )

    save_line_plot(
        results=results,
        metric="accuracy",
        ylabel="Global test accuracy (%)",
        filename="iid_100clients_accuracy_by_round",
        subtitle=common_subtitle,
    )

    # Include only methods that actually saved loss values.
    loss_results = {
        label: frame
        for label, frame in results.items()
        if frame["loss"].notna().all()
    }

    excluded_from_loss = [
        label for label in results if label not in loss_results
    ]

    if loss_results:
        loss_subtitle = common_subtitle
        if excluded_from_loss:
            loss_subtitle += " — available methods only"

        save_line_plot(
            results=loss_results,
            metric="loss",
            ylabel="Global test loss",
            filename="iid_100clients_loss_by_round",
            subtitle=loss_subtitle,
        )

        if excluded_from_loss:
            print(
                "Loss plot excluded methods without saved test loss: "
                + ", ".join(excluded_from_loss)
            )
    else:
        print(
            "Skipped the loss plot because no workbook contains "
            "complete test-loss values."
        )

    save_final_accuracy_plot(results)
    summary = save_summary(results)

    print("\n" + "=" * 70)
    print("RESULT SUMMARY")
    print("=" * 70)
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n" + "=" * 70)
    print(f"All output files were saved in:\n{PLOTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
