#!/bin/bash
#
#SBATCH --job-name=nfs_tr
#SBATCH --output=slurm/logs/fno_m%a_%j.out
#SBATCH --error=slurm/logs/fno_m%a_%j.out
#SBATCH --account=agahlot8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=20:00:00
#SBATCH -p gpu
#
# Usage:
#   sbatch slurm/train_fno.slurm                  # member 0
#   sbatch --array=0-4 slurm/train_fno.slurm      # full 5-member ensemble
#
# 100 epochs ~ 50s/epoch ~ 90 min wall; 4h gives headroom for I/O variance.

module load Miniconda/3

export PYTHONPATH=$HOME/.local/lib/python3.13/site-packages:$PYTHONPATH

cd /nethome/agahlot8/Gitrepos/neural-flow-surrogates
mkdir -p slurm/logs

MEMBER=${SLURM_ARRAY_TASK_ID:-0}
EPOCHS=${EPOCHS:-100}

echo "host=$(hostname)   member=$MEMBER   epochs=$EPOCHS"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

python train.py --model fno --member "$MEMBER" --epochs "$EPOCHS"
