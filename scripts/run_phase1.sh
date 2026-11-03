#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RESULTS_ROOT="${RESULTS_ROOT:-results/phase1}"

COMMON_ARGS=(
    --dataset cifar10
    --model resnet18
    --num-users 10
    --frac 1.0
    --n-clusters 3
    --alpha 0.1
    --rounds 50
    --local-epochs 1
    --batch-size 256
    --lr 0.0001
    --seed 42
    --consistent-split-weights 1
    --noise-floor
)

echo "============================================================"
echo "Phase 1 Matrix (CIFAR-10 / ResNet-18, 10 clients, alpha=0.1)"
echo "Results directory: ${RESULTS_ROOT}"
echo "============================================================"

# 1. Baseline
echo "[1/4] Running Baseline..."
"${PYTHON_BIN}" scripts/run_experiment.py \
    "${COMMON_ARGS[@]}" \
    --method baseline \
    --cluster-mode geometry \
    --zscore-mode pooled \
    --output-dir "${RESULTS_ROOT}/baseline"

# 2. EMD+ORC, CLUSTER_MODE=geometry, ZSCORE_MODE=per_client
echo "[2/4] Running EMD+ORC (CLUSTER_MODE=geometry, ZSCORE_MODE=per_client)..."
"${PYTHON_BIN}" scripts/run_experiment.py \
    "${COMMON_ARGS[@]}" \
    --method emd_orc \
    --cluster-mode geometry \
    --zscore-mode per_client \
    --output-dir "${RESULTS_ROOT}/emd_orc_geometry_per_client"

# 3. EMD+ORC, CLUSTER_MODE=geometry, ZSCORE_MODE=pooled
echo "[3/4] Running EMD+ORC (CLUSTER_MODE=geometry, ZSCORE_MODE=pooled)..."
"${PYTHON_BIN}" scripts/run_experiment.py \
    "${COMMON_ARGS[@]}" \
    --method emd_orc \
    --cluster-mode geometry \
    --zscore-mode pooled \
    --output-dir "${RESULTS_ROOT}/emd_orc_geometry_pooled"

# 4. EMD+ORC, CLUSTER_MODE=random, ZSCORE_MODE=pooled
echo "[4/4] Running EMD+ORC (CLUSTER_MODE=random, ZSCORE_MODE=pooled)..."
"${PYTHON_BIN}" scripts/run_experiment.py \
    "${COMMON_ARGS[@]}" \
    --method emd_orc \
    --cluster-mode random \
    --zscore-mode pooled \
    --output-dir "${RESULTS_ROOT}/emd_orc_random_pooled"

echo "Summarizing Phase 1 results..."
"${PYTHON_BIN}" scripts/summarize_phase1.py --results-root "${RESULTS_ROOT}"
