from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# Folder containing this Python script
BASE_DIR = Path(__file__).resolve().parent


def read_excel_from(folder_name):
    folder = BASE_DIR / folder_name
    files = list(folder.glob("*.xlsx"))

    if not files:
        raise FileNotFoundError(
            f"No Excel file found inside: {folder}"
        )

    if len(files) > 1:
        raise ValueError(
            f"Multiple Excel files found inside {folder}. "
            f"Keep only the result you want to plot."
        )

    print(f"Reading: {files[0]}")
    return pd.read_excel(files[0])


# Read one Excel file from each folder
baseline = read_excel_from("Baseline")
emd_only = read_excel_from("EMD_only")
emd_orc = read_excel_from("EMD+ORC")
weighted = read_excel_from("EMD+ORC_Weighted")
fedwad_files = list(
    (BASE_DIR / "FedWaD").glob("*FEDWAD Only*.xlsx")
)

if len(fedwad_files) != 1:
    raise ValueError(
        f"Expected one completed FedWaD Excel file, found: {fedwad_files}"
    )

print(f"Reading: {fedwad_files[0]}")
fedwad = pd.read_excel(fedwad_files[0])


# Create plot
plt.figure(figsize=(12, 7))

plt.plot(
    baseline["round"],
    baseline["global_acc_test"],
    label="Baseline",
    linewidth=2
)

plt.plot(
    emd_only["round"],
    emd_only["acc_test"],
    label="EMD-only",
    linewidth=2
)

plt.plot(
    emd_orc["round"],
    emd_orc["acc_test"],
    label="EMD + ORC",
    linewidth=2
)

plt.plot(
    weighted["round"],
    weighted["acc_test"],
    label="Weighted EMD + ORC",
    linewidth=3
)

plt.plot(
    fedwad["round"],
    fedwad["acc_test"],
    label="FedWaD",
    linewidth=2
)

plt.title(
    "SplitFed Method Comparison on CIFAR-10\n"
    "Non-IID α=0.5, 10 Clients, Seed=42, ResNet18"
)

plt.xlabel("Global Round")
plt.ylabel("Test Accuracy (%)")
plt.xlim(1, 50)
plt.xticks(range(0, 51, 5))
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()

# Save inside New_Results
output_path = BASE_DIR / "splitfed_method_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")

print(f"Plot saved to: {output_path}")

plt.show()