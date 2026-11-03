#!/bin/bash
#SBATCH --job-name=curvfed_phase1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/phase1_%j.out
#SBATCH --error=logs/phase1_%j.err

set -euo pipefail

mkdir -p logs

if [ -d "curvfed/bin" ]; then
    source curvfed/bin/activate
elif [ -d ".venv/bin" ]; then
    source .venv/bin/activate
fi

echo "Job started on $(hostname) at $(date)"
bash scripts/run_phase1.sh
echo "Job finished at $(date)"
