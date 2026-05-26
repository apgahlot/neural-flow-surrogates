#!/bin/bash
#
#SBATCH --job-name=nfs_tr_noise
#SBATCH --output=slurm/logs/fno_noise_m%a_%j.out
#SBATCH --error=slurm/logs/fno_noise_m%a_%j.out
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
#   sbatch --dependency=afterany:<RUNNING_JOB_ID> slurm/train_fno_noise.slurm
#
# The dependency keeps this job QUEUED (no GPU allocated) until the prior
# training job terminates; once it does, this script archives the previous
# run's artifacts and starts the input-noise-augmented retrain.

module load Miniconda/3
export PYTHONPATH=$HOME/.local/lib/python3.13/site-packages:$PYTHONPATH
cd /nethome/agahlot8/Gitrepos/neural-flow-surrogates
mkdir -p slurm/logs

MEMBER=${SLURM_ARRAY_TASK_ID:-0}
EPOCHS=${EPOCHS:-100}

DATA=/slimdata/abhinav/neuralflow
mkdir -p "$DATA/checkpoints/archive"

# Preserve the previous run's artifacts for A/B comparison.
# Checkpoint moves to archive/ with a NEW prefix so rollout.py's `fno_m*.pt`
# glob does not pick it up alongside the fresh run.
if [ -f "$DATA/checkpoints/fno_m${MEMBER}.pt" ]; then
    mv -v "$DATA/checkpoints/fno_m${MEMBER}.pt" \
          "$DATA/checkpoints/archive/run2_modes20_m${MEMBER}.pt"
fi
if [ -f "$DATA/results/train_fno_m${MEMBER}.csv" ]; then
    mv -v "$DATA/results/train_fno_m${MEMBER}.csv" \
          "$DATA/results/train_fno_m${MEMBER}_noNoise.csv"
fi
if [ -f "$DATA/results/train_fno_m${MEMBER}.png" ]; then
    mv -v "$DATA/results/train_fno_m${MEMBER}.png" \
          "$DATA/results/train_fno_m${MEMBER}_noNoise.png"
fi

echo "host=$(hostname)   member=$MEMBER   epochs=$EPOCHS   input_noise_sat=0.03"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

python -u train.py --model fno --member "$MEMBER" --epochs "$EPOCHS"
