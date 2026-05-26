#!/bin/bash
#
#SBATCH --job-name=nfs_tr_n05
#SBATCH --output=slurm/logs/fno_noise05_m%a_%j.out
#SBATCH --error=slurm/logs/fno_noise05_m%a_%j.out
#SBATCH --account=agahlot8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00
#SBATCH -p gpu
#
# Usage:
#   sbatch slurm/train_fno_noise05.slurm
#   sbatch --dependency=afterany:<JOBID> slurm/train_fno_noise05.slurm
#
# Trains the FNO with input_noise_sat=0.05 (set in configs/default.yaml).
# Archives the previous run's noise=0.03 artifacts under a `_noise03` suffix
# so this run can be A/B compared against it.

module load Miniconda/3
export PYTHONPATH=$HOME/.local/lib/python3.13/site-packages:$PYTHONPATH
cd /nethome/agahlot8/Gitrepos/neural-flow-surrogates
mkdir -p slurm/logs

MEMBER=${SLURM_ARRAY_TASK_ID:-0}
EPOCHS=${EPOCHS:-100}

DATA=/slimdata/abhinav/neuralflow
mkdir -p "$DATA/checkpoints/archive"

# Preserve the previous (noise=0.03) run's artifacts.
# Checkpoint moves to archive/ with a NEW prefix so rollout.py's `fno_m*.pt`
# glob does not pick it up alongside the fresh run.
if [ -f "$DATA/checkpoints/fno_m${MEMBER}.pt" ]; then
    mv -v "$DATA/checkpoints/fno_m${MEMBER}.pt" \
          "$DATA/checkpoints/archive/run3_noise03_m${MEMBER}.pt"
fi
if [ -f "$DATA/results/train_fno_m${MEMBER}.csv" ]; then
    mv -v "$DATA/results/train_fno_m${MEMBER}.csv" \
          "$DATA/results/train_fno_m${MEMBER}_noise03.csv"
fi
if [ -f "$DATA/results/train_fno_m${MEMBER}.png" ]; then
    mv -v "$DATA/results/train_fno_m${MEMBER}.png" \
          "$DATA/results/train_fno_m${MEMBER}_noise03.png"
fi

echo "host=$(hostname)   member=$MEMBER   epochs=$EPOCHS   input_noise_sat=0.05"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

python -u train.py --model fno --member "$MEMBER" --epochs "$EPOCHS"
