#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "${REPO_ROOT}/curvfed/bin/python" ]; then
        PYTHON_BIN="${REPO_ROOT}/curvfed/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

RESULTS_ROOT="${RESULTS_ROOT:-results/phase2}"
ALPHAS=(0.1 0.5)
SEEDS=(42 0 1)

echo "============================================================"
echo "Phase 2 Matrix (CIFAR-10 / ResNet-18, 10 clients, frac=1.0, 50 rounds)"
echo "Alphas: ${ALPHAS[*]} | Seeds: ${SEEDS[*]}"
echo "Python: ${PYTHON_BIN}"
echo "Results directory: ${RESULTS_ROOT}"
echo "============================================================"

for ALPHA in "${ALPHAS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        COMMON_ARGS=(
            --dataset cifar10
            --model resnet18
            --num-users 10
            --frac 1.0
            --alpha "${ALPHA}"
            --rounds 50
            --local-epochs 1
            --batch-size 256
            --lr 0.0001
            --seed "${SEED}"
        )

        BASE_OUT="${RESULTS_ROOT}/alpha_${ALPHA}"

        echo "------------------------------------------------------------"
        echo "Running alpha=${ALPHA}, seed=${SEED}"
        echo "------------------------------------------------------------"

        # 1. fedavg
        echo "[1/5] fedavg (alpha=${ALPHA}, seed=${SEED})..."
        "${PYTHON_BIN}" scripts/run_experiment.py \
            "${COMMON_ARGS[@]}" \
            --method fedavg \
            --output-dir "${BASE_OUT}/fedavg/seed_${SEED}"

        # 2. tca decoupled, cost = update, marginal = hard
        echo "[2/5] tca decoupled (cost=update, marginal=hard) (alpha=${ALPHA}, seed=${SEED})..."
        "${PYTHON_BIN}" scripts/run_experiment.py \
            "${COMMON_ARGS[@]}" \
            --method tca \
            --tca-mode decoupled \
            --tca-cost update \
            --tca-marginal hard \
            --tca-calibrate 0 \
            --output-dir "${BASE_OUT}/tca_decoupled_update_hard/seed_${SEED}"

        # 3. tca decoupled, cost = update, marginal = none
        echo "[3/5] tca decoupled (cost=update, marginal=none) (alpha=${ALPHA}, seed=${SEED})..."
        "${PYTHON_BIN}" scripts/run_experiment.py \
            "${COMMON_ARGS[@]}" \
            --method tca \
            --tca-mode decoupled \
            --tca-cost update \
            --tca-marginal none \
            --tca-calibrate 0 \
            --output-dir "${BASE_OUT}/tca_decoupled_update_none/seed_${SEED}"

        # 4. tca feedback, calibrate = false
        echo "[4/5] tca feedback (calibrate=false) (alpha=${ALPHA}, seed=${SEED})..."
        "${PYTHON_BIN}" scripts/run_experiment.py \
            "${COMMON_ARGS[@]}" \
            --method tca \
            --tca-mode feedback \
            --tca-cost update \
            --tca-marginal hard \
            --tca-calibrate 0 \
            --output-dir "${BASE_OUT}/tca_feedback_uncalibrated/seed_${SEED}"

        # 5. tca feedback, calibrate = true
        echo "[5/5] tca feedback (calibrate=true, kappa_target=auto) (alpha=${ALPHA}, seed=${SEED})..."
        "${PYTHON_BIN}" scripts/run_experiment.py \
            "${COMMON_ARGS[@]}" \
            --method tca \
            --tca-mode feedback \
            --tca-cost update \
            --tca-marginal hard \
            --tca-calibrate 1 \
            --tca-kappa-target auto \
            --output-dir "${BASE_OUT}/tca_feedback_calibrated/seed_${SEED}"


    done
done

echo "Summarizing Phase 2 results..."
"${PYTHON_BIN}" scripts/summarize_phase2.py --results-root "${RESULTS_ROOT}"
