#!/bin/bash
#SBATCH --job-name=SFLV1_HAM10000
#SBATCH --account=jannesar-lab
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=07:00:00
#SBATCH --output=logs/sflv1_%j.out
#SBATCH --error=logs/sflv1_%j.err
#SBATCH --signal=B:USR1@300
#SBATCH --requeue

echo "Job started on $(hostname) at $(date)"

# Load environment
cd /lustre/hdd/LAS/jannesar-lab/neha2004/CurvFed
source curvfed_env/bin/activate

cd SplitFed-When-Federated-Learning-Meets-Split-Learning

# Run training
python3 SFLV1_ResNet_HAM10000.py

echo "Job finished at $(date)"

