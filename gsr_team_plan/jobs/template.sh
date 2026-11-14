#!/bin/bash
# fep A100 job template. Copy per run: jobs/<run name>.sh, change --job-name and the command.
# Keep every job under 10 hours (12 h limit) and save a checkpoint every epoch.
#SBATCH --job-name=A1_yolo11m_ft
#SBATCH --partition=<A100 partition name from the fep docs>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tracklab
cd ~/sn-gamestate
mkdir -p logs

# <the training command from your task>, for example:
# yolo detect train model=yolo11m.pt data=data/packs/A/gsr1.yaml epochs=15 patience=5 imgsz=1280 batch=8 \
#      optimizer=SGD lr0=0.01 seed=0 deterministic=True project=runs/detA name=A1_yolo11m_ft
