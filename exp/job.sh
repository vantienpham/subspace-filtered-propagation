#!/bin/bash
#SBATCH --job-name=spl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=11500
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Submit from the directory that contains exp/ and logs/, for example
#   mkdir -p logs && sbatch -J e2 -p <partition> --time=0-01:00:00 exp/job.sh exp/e2_propagation.py

set -e
cd "${SLURM_SUBMIT_DIR:-.}"
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
echo "== $SLURM_JOB_NAME on $(hostname) @ $(date) =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -u "$@"
echo "== done @ $(date) =="
