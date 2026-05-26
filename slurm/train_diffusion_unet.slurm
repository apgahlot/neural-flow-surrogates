#!/bin/bash
#
#SBATCH --job-name=nfs_tr_diff
#SBATCH --output=slurm/logs/diffusion_unet_%j.out
#SBATCH --error=slurm/logs/diffusion_unet_%j.out
#SBATCH --account=agahlot8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH -p gpu
#
# Usage:
#   sbatch slurm/train_diffusion_unet.slurm
#   EPOCHS=200 sbatch slurm/train_diffusion_unet.slurm
#
# Trains the conditional U-Net diffusion surrogate. Single-step (K=1) epsilon
# prediction over 1000 timesteps (cosine schedule), DDIM @ 50 steps for inference.
# Memory note: 256x512 input + base=64 + 4 levels w/ full-res skip activations
# is heavy. AMP is on; if OOM occurs, drop train.batch_size in default.yaml
# from 16 -> 8 and resubmit.

module load Miniconda/3
export PYTHONPATH=$HOME/.local/lib/python3.13/site-packages:$PYTHONPATH
cd /nethome/agahlot8/Gitrepos/neural-flow-surrogates
mkdir -p slurm/logs

EPOCHS=${EPOCHS:-150}

DATA=/slimdata/abhinav/neuralflow
mkdir -p "$DATA/checkpoints/archive"

# Archive any prior diffusion-unet artifacts (safety; first run has none).
if [ -f "$DATA/checkpoints/diffusion-unet.pt" ]; then
    TS=$(date +%Y%m%d_%H%M%S)
    mv -v "$DATA/checkpoints/diffusion-unet.pt" \
          "$DATA/checkpoints/archive/diffusion-unet_${TS}.pt"
    [ -f "$DATA/results/train_diffusion-unet.csv" ] && \
      mv -v "$DATA/results/train_diffusion-unet.csv" \
            "$DATA/results/train_diffusion-unet_${TS}.csv"
    [ -f "$DATA/results/train_diffusion-unet.png" ] && \
      mv -v "$DATA/results/train_diffusion-unet.png" \
            "$DATA/results/train_diffusion-unet_${TS}.png"
fi

echo "host=$(hostname)   epochs=$EPOCHS   model=diffusion-unet"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

python -u train.py --model diffusion-unet --epochs "$EPOCHS"
