import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "Results_alpha1.0"

EXPERIMENTS = {
    "Baseline": os.path.join(
        BASE_DIR,
        "Baseline",
        "SFLV1 ResNet18 on CIFAR10 NonIID.xlsx"
    ),

    "EMD Only": os.path.join(
        BASE_DIR,
        "EMD",
        "SFLV1 ResNet18 on CIFAR10 NonIID EMD Only.xlsx"
    ),

    "EMD + ORC": os.path.join(
        BASE_DIR,
        "Cluster",
        "SFLV1 ResNet18 on CIFAR10 NonIID Clustered.xlsx"
    ),

    "EMD + ORC Weighted": os.path.join(
        BASE_DIR,
        "Cluster_weighted",
        "SFLV1 ResNet18 on CIFAR10 NonIID Clustered Weighted.xlsx"
    ),
}

SHEET_NAME = "v1_test"

data = {}

for method, filepath in EXPERIMENTS.items():

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Missing Excel file for {method}: {filepath}"
        )

    df = pd.read_excel(
        filepath,
        sheet_name=SHEET_NAME
    )

    df = df[
        ["round", "acc_train", "acc_test"]
    ].copy()

    data[method] = df

    print("\n" + "=" * 70)
    print(method)
    print("SOURCE:", filepath)
    print(df.head(2).to_string(index=False))
    print("...")
    print(df.tail(2).to_string(index=False))


# ============================================================
# Validation
# ============================================================

expected_rounds = list(range(1, 51))

for method, df in data.items():

    assert len(df) == 50, (
        f"{method}: expected 50 rows, found {len(df)}"
    )

    assert (
        df["round"].astype(int).tolist()
        == expected_rounds
    ), (
        f"{method}: rounds are not exactly 1-50"
    )

    assert not df[
        ["acc_train", "acc_test"]
    ].isnull().any().any(), (
        f"{method}: missing accuracy values"
    )

print("\nValidation passed for all four experiments.")


# ============================================================
# Output directory
# ============================================================

output_dir = os.path.join(
    BASE_DIR,
    "Comparison"
)

os.makedirs(
    output_dir,
    exist_ok=True
)


# ============================================================
# Combined table
# ============================================================

comparison = pd.DataFrame({
    "round": expected_rounds
})

for method, df in data.items():

    safe_name = (
        method.lower()
        .replace(" + ", "_")
        .replace(" ", "_")
    )

    comparison[
        f"{safe_name}_train"
    ] = df["acc_train"].values

    comparison[
        f"{safe_name}_test"
    ] = df["acc_test"].values


comparison.to_excel(
    os.path.join(
        output_dir,
        "alpha1_all_rounds_combined.xlsx"
    ),
    index=False
)


# ============================================================
# Styles
# ============================================================

line_styles = {
    "Baseline": "-",
    "EMD Only": "--",
    "EMD + ORC": "-.",
    "EMD + ORC Weighted": ":"
}

markers = {
    "Baseline": "o",
    "EMD Only": "s",
    "EMD + ORC": "^",
    "EMD + ORC Weighted": "D"
}


# ============================================================
# TEST ACCURACY PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 7)
)

for method, df in data.items():

    ax.plot(
        df["round"],
        df["acc_test"],
        label=method,
        linestyle=line_styles[method],
        linewidth=2.4,
        marker=markers[method],
        markersize=5,
        markevery=5
    )

ax.set_title(
    "CIFAR-10 Test Accuracy Comparison",
    fontsize=18,
    fontweight="bold",
    pad=15
)

ax.text(
    0.5,
    1.01,
    "Dirichlet α = 1.0 | Seed = 42 | 10 Clients",
    transform=ax.transAxes,
    ha="center",
    fontsize=11
)

ax.set_xlabel(
    "Communication Round",
    fontsize=13
)

ax.set_ylabel(
    "Test Accuracy (%)",
    fontsize=13
)

ax.set_xlim(1, 50)

ax.set_xticks(
    [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
)

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.30
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    fontsize=11,
    frameon=False,
    loc="best"
)

ax.tick_params(
    labelsize=11
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        output_dir,
        "alpha1_TEST_accuracy_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    os.path.join(
        output_dir,
        "alpha1_TEST_accuracy_comparison.pdf"
    ),
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# TRAIN ACCURACY PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 7)
)

for method, df in data.items():

    ax.plot(
        df["round"],
        df["acc_train"],
        label=method,
        linestyle=line_styles[method],
        linewidth=2.4,
        marker=markers[method],
        markersize=5,
        markevery=5
    )

ax.set_title(
    "CIFAR-10 Train Accuracy Comparison",
    fontsize=18,
    fontweight="bold",
    pad=15
)

ax.text(
    0.5,
    1.01,
    "Dirichlet α = 1.0 | Seed = 42 | 10 Clients",
    transform=ax.transAxes,
    ha="center",
    fontsize=11
)

ax.set_xlabel(
    "Communication Round",
    fontsize=13
)

ax.set_ylabel(
    "Train Accuracy (%)",
    fontsize=13
)

ax.set_xlim(1, 50)

ax.set_xticks(
    [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
)

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.30
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    fontsize=11,
    frameon=False,
    loc="best"
)

ax.tick_params(
    labelsize=11
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        output_dir,
        "alpha1_TRAIN_accuracy_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    os.path.join(
        output_dir,
        "alpha1_TRAIN_accuracy_comparison.pdf"
    ),
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# Summary
# ============================================================

summary = []

for method, df in data.items():

    best_test_idx = df["acc_test"].idxmax()
    best_train_idx = df["acc_train"].idxmax()

    summary.append({
        "Method": method,

        "Best Train Accuracy (%)":
            df.loc[best_train_idx, "acc_train"],

        "Best Train Round":
            int(df.loc[best_train_idx, "round"]),

        "Final Train Accuracy (%)":
            df["acc_train"].iloc[-1],

        "Best Test Accuracy (%)":
            df.loc[best_test_idx, "acc_test"],

        "Best Test Round":
            int(df.loc[best_test_idx, "round"]),

        "Final Test Accuracy (%)":
            df["acc_test"].iloc[-1],

        "Last 5 Test Avg (%)":
            df["acc_test"].tail(5).mean(),

        "Last 10 Test Avg (%)":
            df["acc_test"].tail(10).mean()
    })


summary_df = pd.DataFrame(
    summary
)

summary_df.to_excel(
    os.path.join(
        output_dir,
        "alpha1_results_summary.xlsx"
    ),
    index=False
)

print("\n" + "=" * 100)
print("ALPHA = 1.0, SEED = 42 RESULTS")
print("=" * 100)

print(
    summary_df.to_string(
        index=False
    )
)

print("\nSaved to:")
print(output_dir)
