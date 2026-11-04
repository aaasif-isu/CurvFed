# """Plot CIFAR-10 IID SplitFed results for the 100-client experiment.

# Place this file as:
#     IID_Results_100_clients/plot.py

# Expected method directories:
#     Baseline/
#     EMD/
#     EMD+ORC/
#     EMD+ORC+Weighted/
#     FedWaD/

# Each available directory should contain exactly one .xlsx workbook. Accuracy is
# required. Test loss is optional because some experiment scripts did not save it.
# """

# from pathlib import Path

# import matplotlib

# # Required when plotting on Nova without a graphical display.
# matplotlib.use("Agg")

# import matplotlib.pyplot as plt
# import pandas as pd


# RESULTS_ROOT = Path(__file__).resolve().parent
# PLOTS_DIR = RESULTS_ROOT / "Plots"
# EXPECTED_ROUNDS = 50


# # Plot label -> result directory
# METHODS = {
#     "Baseline": "Baseline",
#     "EMD only": "EMD",
#     "EMD + ORC": "EMD+ORC",
#     "Weighted EMD + ORC": "EMD+ORC+Weighted",
#     "FedWaD only": "FedWaD",
# }


# COLORS = {
#     "Baseline": "#4C78A8",
#     "EMD only": "#F58518",
#     "EMD + ORC": "#54A24B",
#     "Weighted EMD + ORC": "#E45756",
#     "FedWaD only": "#9467BD",
# }


# def find_excel_file(directory: Path) -> Path | None:
#     """Return the single Excel workbook in a directory, if available."""

#     if not directory.exists():
#         print(f"Warning: directory is missing; skipping method:\n  {directory}")
#         return None

#     files = sorted(
#         path
#         for path in directory.glob("*.xlsx")
#         if not path.name.startswith("~$")
#     )

#     if not files:
#         print(f"Warning: no .xlsx workbook found; skipping method:\n  {directory}")
#         return None

#     if len(files) > 1:
#         raise RuntimeError(
#             f"Expected exactly one .xlsx workbook in {directory}, "
#             f"but found {len(files)}:\n"
#             + "\n".join(str(path) for path in files)
#         )

#     return files[0]


# def load_method(label: str, directory_name: str) -> pd.DataFrame | None:
#     """Load one experiment workbook and normalize its column names.

#     Accuracy is required. Loss is optional and becomes NaN when unavailable.
#     """

#     directory = RESULTS_ROOT / directory_name
#     excel_file = find_excel_file(directory)

#     if excel_file is None:
#         return None

#     print(f"Loading {label}:")
#     print(f"  {excel_file.name}")

#     frame = pd.read_excel(excel_file)

#     if label == "Baseline":
#         accuracy_candidates = ["global_acc_test", "acc_test"]
#         loss_candidates = ["global_loss_test", "loss_test"]
#     else:
#         accuracy_candidates = ["acc_test", "global_acc_test"]
#         loss_candidates = ["loss_test", "global_loss_test"]

#     accuracy_column = next(
#         (name for name in accuracy_candidates if name in frame.columns),
#         None,
#     )
#     loss_column = next(
#         (name for name in loss_candidates if name in frame.columns),
#         None,
#     )

#     missing_required = []
#     if "round" not in frame.columns:
#         missing_required.append("round")
#     if accuracy_column is None:
#         missing_required.append(
#             f"one of {accuracy_candidates}"
#         )

#     if missing_required:
#         raise RuntimeError(
#             f"{excel_file} is missing required columns:\n"
#             f"{missing_required}\n\n"
#             f"Available columns:\n{list(frame.columns)}"
#         )

#     normalized = pd.DataFrame()
#     normalized["round"] = pd.to_numeric(
#         frame["round"], errors="coerce"
#     )
#     normalized["accuracy"] = pd.to_numeric(
#         frame[accuracy_column], errors="coerce"
#     )

#     if loss_column is not None:
#         normalized["loss"] = pd.to_numeric(
#             frame[loss_column], errors="coerce"
#         )
#     else:
#         normalized["loss"] = float("nan")
#         print(
#             "  Warning: no test-loss column was saved. "
#             "Accuracy will still be plotted."
#         )

#     required_values = normalized[["round", "accuracy"]]
#     if required_values.isna().any().any():
#         bad_rows = normalized[
#             required_values.isna().any(axis=1)
#         ]
#         raise RuntimeError(
#             f"{excel_file} contains missing or non-numeric round/accuracy "
#             f"values:\n{bad_rows}"
#         )

#     normalized["round"] = normalized["round"].astype(int)
#     normalized = normalized.sort_values("round").reset_index(drop=True)

#     if len(normalized) != EXPECTED_ROUNDS:
#         raise RuntimeError(
#             f"{excel_file} contains {len(normalized)} rows; "
#             f"expected {EXPECTED_ROUNDS}."
#         )

#     expected = list(range(1, EXPECTED_ROUNDS + 1))
#     actual = normalized["round"].tolist()

#     if actual != expected:
#         raise RuntimeError(
#             f"{excel_file} does not contain rounds 1-{EXPECTED_ROUNDS} "
#             f"exactly once and in order after sorting.\n"
#             f"Found rounds: {actual}"
#         )

#     return normalized


# def save_line_plot(
#     results: dict[str, pd.DataFrame],
#     metric: str,
#     ylabel: str,
#     filename: str,
#     subtitle: str,
# ) -> None:
#     """Create and save a round-by-round line plot."""

#     fig, ax = plt.subplots(figsize=(11, 6.5))

#     for label, frame in results.items():
#         ax.plot(
#             frame["round"],
#             frame[metric],
#             label=label,
#             color=COLORS[label],
#             linewidth=2.2,
#         )

#     ax.set_title(
#         f"CIFAR-10 SplitFed: {ylabel} by Global Round\n{subtitle}",
#         fontsize=14,
#         fontweight="bold",
#     )
#     ax.set_xlabel("Global communication round", fontsize=12)
#     ax.set_ylabel(ylabel, fontsize=12)
#     ax.set_xlim(1, EXPECTED_ROUNDS)
#     ax.set_xticks(range(0, EXPECTED_ROUNDS + 1, 5))

#     if metric == "accuracy":
#         maximum = max(frame[metric].max() for frame in results.values())
#         ax.set_ylim(0, max(35, maximum + 3))

#     ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.3)
#     ax.legend(frameon=False, ncol=2, fontsize=10)
#     fig.tight_layout()

#     png_path = PLOTS_DIR / f"{filename}.png"
#     pdf_path = PLOTS_DIR / f"{filename}.pdf"

#     fig.savefig(png_path, dpi=300, bbox_inches="tight")
#     fig.savefig(pdf_path, bbox_inches="tight")
#     plt.close(fig)

#     print(f"Created: {png_path.name}")
#     print(f"Created: {pdf_path.name}")


# def save_final_accuracy_plot(results: dict[str, pd.DataFrame]) -> None:
#     """Create a bar chart comparing round-50 test accuracy."""

#     labels = list(results.keys())
#     values = [results[label]["accuracy"].iloc[-1] for label in labels]
#     colors = [COLORS[label] for label in labels]

#     fig, ax = plt.subplots(figsize=(11, 6.5))
#     bars = ax.bar(
#         labels,
#         values,
#         color=colors,
#         edgecolor="black",
#         linewidth=0.6,
#     )

#     ax.set_title(
#         "Final Test Accuracy at Round 50\n"
#         "CIFAR-10, 100 clients, IID partition, seed 42",
#         fontsize=14,
#         fontweight="bold",
#     )
#     ax.set_ylabel("Final test accuracy (%)", fontsize=12)
#     ax.set_ylim(0, max(values) + 4)
#     ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.3)
#     ax.tick_params(axis="x", rotation=15)

#     for bar, value in zip(bars, values):
#         ax.text(
#             bar.get_x() + bar.get_width() / 2,
#             value + 0.35,
#             f"{value:.2f}%",
#             ha="center",
#             va="bottom",
#             fontsize=10,
#             fontweight="bold",
#         )

#     fig.tight_layout()

#     png_path = PLOTS_DIR / "iid_100clients_final_accuracy.png"
#     pdf_path = PLOTS_DIR / "iid_100clients_final_accuracy.pdf"

#     fig.savefig(png_path, dpi=300, bbox_inches="tight")
#     fig.savefig(pdf_path, bbox_inches="tight")
#     plt.close(fig)

#     print(f"Created: {png_path.name}")
#     print(f"Created: {pdf_path.name}")


# def save_summary(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
#     """Save final and recent-round metrics as a CSV table."""

#     rows = []

#     for label, frame in results.items():
#         best_index = frame["accuracy"].idxmax()
#         final_loss = (
#             frame["loss"].iloc[-1]
#             if frame["loss"].notna().any()
#             else float("nan")
#         )

#         rows.append(
#             {
#                 "method": label,
#                 "final_accuracy": frame["accuracy"].iloc[-1],
#                 "final_loss": final_loss,
#                 "last_10_round_accuracy_mean": frame["accuracy"].tail(10).mean(),
#                 "last_10_round_accuracy_std": frame["accuracy"].tail(10).std(),
#                 "best_accuracy": frame["accuracy"].max(),
#                 "best_accuracy_round": int(frame.loc[best_index, "round"]),
#             }
#         )

#     summary = pd.DataFrame(rows)
#     summary = summary.sort_values(
#         by="final_accuracy", ascending=False
#     ).reset_index(drop=True)

#     summary_path = PLOTS_DIR / "iid_100clients_summary.csv"
#     summary.to_csv(summary_path, index=False)
#     print(f"Created: {summary_path.name}")

#     return summary


# def main() -> None:
#     """Load available workbooks and create all valid plots."""

#     print("=" * 70)
#     print("CIFAR-10 IID — 100-client SplitFed result plotting")
#     print("=" * 70)

#     PLOTS_DIR.mkdir(parents=True, exist_ok=True)

#     results: dict[str, pd.DataFrame] = {}

#     for label, directory_name in METHODS.items():
#         frame = load_method(label, directory_name)
#         if frame is not None:
#             results[label] = frame

#     if not results:
#         raise RuntimeError("No valid experiment workbooks were found.")

#     print("\nCreating plots...\n")

#     common_subtitle = (
#         "100 total clients, 10 selected per round, "
#         "IID partition, seed 42"
#     )

#     save_line_plot(
#         results=results,
#         metric="accuracy",
#         ylabel="Global test accuracy (%)",
#         filename="iid_100clients_accuracy_by_round",
#         subtitle=common_subtitle,
#     )

#     # Include only methods that actually saved loss values.
#     loss_results = {
#         label: frame
#         for label, frame in results.items()
#         if frame["loss"].notna().all()
#     }

#     excluded_from_loss = [
#         label for label in results if label not in loss_results
#     ]

#     if loss_results:
#         loss_subtitle = common_subtitle
#         if excluded_from_loss:
#             loss_subtitle += " — available methods only"

#         save_line_plot(
#             results=loss_results,
#             metric="loss",
#             ylabel="Global test loss",
#             filename="iid_100clients_loss_by_round",
#             subtitle=loss_subtitle,
#         )

#         if excluded_from_loss:
#             print(
#                 "Loss plot excluded methods without saved test loss: "
#                 + ", ".join(excluded_from_loss)
#             )
#     else:
#         print(
#             "Skipped the loss plot because no workbook contains "
#             "complete test-loss values."
#         )

#     save_final_accuracy_plot(results)
#     summary = save_summary(results)

#     print("\n" + "=" * 70)
#     print("RESULT SUMMARY")
#     print("=" * 70)
#     print(
#         summary.to_string(
#             index=False,
#             float_format=lambda value: f"{value:.4f}",
#         )
#     )

#     print("\n" + "=" * 70)
#     print(f"All output files were saved in:\n{PLOTS_DIR}")
#     print("=" * 70)


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""Compare Pure FL and SplitFed methods for CIFAR-10 IID, 100 clients.

Place this file as:
    IID_Results_100_clients/plot.py

Expected structure:
    Baseline/
    EMD/
    EMD+ORC/
    EMD+ORC+Weighted/
    FedWaD/
    FL_resnet18_cifar10_IID_clients100.../

The script saves accuracy, optional loss, final-accuracy, and summary outputs
inside IID_Results_100_clients/Plots/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_ROOT = Path(__file__).resolve().parent
PLOTS_DIR = RESULTS_ROOT / "Plots"
EXPECTED_ROUNDS = 50


METHODS = [
    (("Baseline",), "SplitFed baseline", "#4C78A8", 2.3, "-"),
    (("EMD", "EMD_only"), "EMD-only", "#F58518", 2.3, "-"),
    (("EMD+ORC",), "EMD + ORC", "#54A24B", 2.3, "-"),
    (("EMD+ORC+Weighted", "EMD+ORC_Weighted", "EMD+ORC_weighted"),
     "Weighted EMD + ORC", "#E45756", 3.0, "-"),
    (("FedWaD",), "FedWaD", "#9467BD", 2.3, "-"),
]


def excel_files(folder):
    """Return non-temporary Excel workbooks recursively."""

    return sorted(
        path for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".xlsx", ".xls"}
        and not path.name.startswith("~$")
    )


def is_iid_file(path):
    """Recognize either IID or alpha=999999 naming."""

    text = str(path).lower().replace("_", "").replace(" ", "")
    if "alpha999999" in text:
        return True
    return "iid" in text and "noniid" not in text


def matches_method(path, label):
    """Identify the method using the workbook filename, not its folder."""

    full_text = str(path).lower().replace("_", " ").replace("+", " ")
    name_text = path.name.lower().replace("_", " ").replace("+", " ")
    compact_name = name_text.replace(" ", "")

    if label == "Pure FL":
        compact_full = full_text.replace(" ", "")
        return "flresnet18" in compact_full or "purefl" in compact_full
    if label == "SplitFed baseline":
        return "baseline" in name_text
    if label == "EMD-only":
        return "emd only" in name_text or "emdonly" in compact_name
    if label == "Weighted EMD + ORC":
        return "weighted" in name_text and "emd" in name_text
    if label == "EMD + ORC":
        return (
            "emd" in name_text
            and "orc" in name_text
            and "weighted" not in name_text
        )
    if label == "FedWaD":
        return "fedwad" in compact_name
    return False


def choose_one(files, label):
    """Select the single correct IID workbook for a method."""

    if not files:
        raise FileNotFoundError(f"No workbook found for {label}.")

    iid_files = [path for path in files if is_iid_file(path)]
    if iid_files:
        files = iid_files

    client_files = [
        path for path in files
        if "clients100" in str(path).lower().replace("_", "").replace(" ", "")
    ]
    if client_files:
        files = client_files

    method_files = [path for path in files if matches_method(path, label)]
    if method_files:
        files = method_files

    if len(files) != 1:
        listing = "\n  - ".join(str(path) for path in files)
        raise RuntimeError(
            f"Expected one IID 100-client workbook for {label}, found "
            f"{len(files)}:\n  - {listing}"
        )

    return files[0]


def find_method_file(folder_candidates, label):
    """Find a method folder and select its correct workbook."""

    for folder_name in folder_candidates:
        folder = RESULTS_ROOT / folder_name
        if folder.is_dir():
            return choose_one(excel_files(folder), label)

    raise FileNotFoundError(
        f"Missing directory for {label}. Expected one of: {folder_candidates}"
    )


def find_pure_fl_file():
    """Find the Pure FL workbook stored in the FL run directory."""

    candidates = []

    for path in excel_files(RESULTS_ROOT):
        relative = path.relative_to(RESULTS_ROOT)
        first_part = relative.parts[0].lower()
        if first_part.startswith("fl_") or path.name.lower().startswith("fl_"):
            candidates.append(path)

    return choose_one(candidates, "Pure FL")


def load_result(path, label):
    """Read and validate one method's round-level accuracy and optional loss."""

    workbook = pd.ExcelFile(path)
    sheet = "v1_test" if "v1_test" in workbook.sheet_names else workbook.sheet_names[0]
    frame = pd.read_excel(path, sheet_name=sheet)
    frame.columns = [str(column).strip().lower() for column in frame.columns]

    if "round" not in frame.columns:
        raise RuntimeError(
            f"{path} is missing the 'round' column.\n"
            f"Available columns: {frame.columns.tolist()}"
        )

    if label in {"Pure FL", "SplitFed baseline"}:
        accuracy_candidates = [
            "global_acc_test", "acc_test", "test_accuracy", "test_acc"
        ]
        loss_candidates = [
            "global_loss_test", "loss_test", "test_loss"
        ]
    else:
        accuracy_candidates = [
            "acc_test", "global_acc_test", "test_accuracy", "test_acc"
        ]
        loss_candidates = [
            "loss_test", "global_loss_test", "test_loss"
        ]

    accuracy_column = next(
        (column for column in accuracy_candidates if column in frame.columns), None
    )
    loss_column = next(
        (column for column in loss_candidates if column in frame.columns), None
    )

    if accuracy_column is None:
        raise RuntimeError(
            f"No test-accuracy column found in {path}.\n"
            f"Expected one of: {accuracy_candidates}\n"
            f"Available columns: {frame.columns.tolist()}"
        )

    result = pd.DataFrame({
        "round": pd.to_numeric(frame["round"], errors="coerce"),
        "accuracy": pd.to_numeric(frame[accuracy_column], errors="coerce"),
    })
    result["loss"] = (
        pd.to_numeric(frame[loss_column], errors="coerce")
        if loss_column is not None
        else float("nan")
    )

    if result[["round", "accuracy"]].isna().any().any():
        bad_rows = result[result[["round", "accuracy"]].isna().any(axis=1)]
        raise RuntimeError(f"Invalid round or accuracy values in {path}:\n{bad_rows}")

    result["round"] = result["round"].astype(int)
    result = result.sort_values("round").reset_index(drop=True)

    if len(result) != EXPECTED_ROUNDS:
        raise RuntimeError(
            f"{label} contains {len(result)} rounds; expected {EXPECTED_ROUNDS}."
        )
    if result["round"].tolist() != list(range(1, EXPECTED_ROUNDS + 1)):
        raise RuntimeError(f"{label} does not contain rounds 1-50 exactly once.")

    if result["accuracy"].max() <= 1.5:
        result["accuracy"] *= 100.0
    if not result["accuracy"].between(0, 100).all():
        raise RuntimeError(f"Accuracy outside 0-100% in {path}")

    return result, accuracy_column, loss_column


def save_line_plot(results, styles, metric, ylabel, filename, subtitle):
    """Save a round-by-round accuracy or loss plot."""

    figure, axis = plt.subplots(figsize=(12, 7))

    for label, frame in results.items():
        color, width, line_style = styles[label]
        axis.plot(
            frame["round"], frame[metric], label=label,
            color=color, linewidth=width, linestyle=line_style,
        )

    axis.set_title(
        f"Pure FL vs SplitFed Methods on CIFAR-10\n{subtitle}",
        fontsize=14,
        fontweight="bold",
    )
    axis.set_xlabel("Global communication round")
    axis.set_ylabel(ylabel)
    axis.set_xlim(1, EXPECTED_ROUNDS)
    axis.set_xticks(range(0, EXPECTED_ROUNDS + 1, 5))
    axis.grid(True, linestyle="--", alpha=0.3)
    axis.legend(frameon=True, ncol=2)
    figure.tight_layout()

    for extension in ("png", "pdf"):
        path = PLOTS_DIR / f"{filename}.{extension}"
        figure.savefig(
            path,
            dpi=300 if extension == "png" else None,
            bbox_inches="tight",
        )
        print(f"Created: {path}")

    plt.close(figure)


def save_final_accuracy_plot(results, styles):
    """Save a round-50 accuracy bar chart."""

    labels = list(results)
    values = [results[label]["accuracy"].iloc[-1] for label in labels]
    colors = [styles[label][0] for label in labels]

    figure, axis = plt.subplots(figsize=(12, 7))
    bars = axis.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6)
    axis.set_title(
        "Final Test Accuracy at Round 50\n"
        "CIFAR-10 IID, 100 Clients, 10 Active/Round, Seed=42, ResNet18",
        fontsize=14,
        fontweight="bold",
    )
    axis.set_ylabel("Final test accuracy (%)")
    axis.set_ylim(0, max(values) + max(4, max(values) * 0.08))
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    axis.tick_params(axis="x", rotation=15)

    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.4,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    figure.tight_layout()
    for extension in ("png", "pdf"):
        path = PLOTS_DIR / f"iid_100clients_final_accuracy_FL_vs_SplitFed.{extension}"
        figure.savefig(
            path,
            dpi=300 if extension == "png" else None,
            bbox_inches="tight",
        )
        print(f"Created: {path}")
    plt.close(figure)


def save_summary(results, sources):
    """Save numerical rankings and return the summary table."""

    rows = []
    for label, frame in results.items():
        best_index = frame["accuracy"].idxmax()
        rows.append({
            "method": label,
            "final_accuracy": frame["accuracy"].iloc[-1],
            "final_loss": (
                frame["loss"].iloc[-1]
                if frame["loss"].notna().any()
                else float("nan")
            ),
            "last_10_round_accuracy_mean": frame["accuracy"].tail(10).mean(),
            "last_10_round_accuracy_std": frame["accuracy"].tail(10).std(),
            "best_accuracy": frame["accuracy"].max(),
            "best_accuracy_round": int(frame.loc[best_index, "round"]),
            "source_file": str(sources[label]),
        })

    summary = pd.DataFrame(rows).sort_values(
        "final_accuracy", ascending=False
    ).reset_index(drop=True)
    path = PLOTS_DIR / "iid_100clients_FL_vs_SplitFed_summary.csv"
    summary.to_csv(path, index=False)
    print(f"Created: {path}")
    return summary


def main():
    """Load all six methods and create plots and rankings."""

    print("=" * 72)
    print("CIFAR-10 IID, 100 clients: Pure FL vs SplitFed methods")
    print("=" * 72)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    sources = {"Pure FL": find_pure_fl_file()}
    styles = {"Pure FL": ("#111111", 2.8, "--")}

    for folders, label, color, width, line_style in METHODS:
        sources[label] = find_method_file(folders, label)
        styles[label] = (color, width, line_style)

    results = {}
    for label, path in sources.items():
        frame, accuracy_column, loss_column = load_result(path, label)
        results[label] = frame
        print(f"{label}:\n  {path}")
        print(
            f"  accuracy_column={accuracy_column}, "
            f"loss_column={loss_column}, "
            f"final_accuracy={frame['accuracy'].iloc[-1]:.3f}%"
        )

    subtitle = "IID, 100 Clients, 10 Active/Round, Seed=42, ResNet18"
    save_line_plot(
        results, styles, "accuracy", "Global test accuracy (%)",
        "iid_100clients_accuracy_by_round_FL_vs_SplitFed", subtitle,
    )

    loss_results = {
        label: frame
        for label, frame in results.items()
        if frame["loss"].notna().all()
    }
    if loss_results:
        save_line_plot(
            loss_results, styles, "loss", "Global test loss",
            "iid_100clients_loss_by_round_FL_vs_SplitFed",
            subtitle + " — methods with saved loss",
        )
    else:
        print("Skipped loss plot because no method has complete saved loss values.")

    save_final_accuracy_plot(results, styles)
    summary = save_summary(results, sources)

    display_columns = [
        "method", "final_accuracy", "last_10_round_accuracy_mean",
        "last_10_round_accuracy_std", "best_accuracy", "best_accuracy_round",
    ]
    print("\nFINAL-ROUND RANKING")
    print(summary[display_columns].to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    ))

    final_winner = summary.iloc[0]
    stable_winner = summary.sort_values(
        "last_10_round_accuracy_mean", ascending=False
    ).iloc[0]
    print(
        f"\nFinal-round winner: {final_winner['method']} "
        f"({final_winner['final_accuracy']:.3f}%)"
    )
    print(
        f"Last-10-round winner: {stable_winner['method']} "
        f"({stable_winner['last_10_round_accuracy_mean']:.3f}% mean)"
    )
    print(f"\nAll outputs saved in:\n{PLOTS_DIR}")


if __name__ == "__main__":
    main()
