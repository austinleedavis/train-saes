#!/bin/bash
#SBATCH --job-name=wandb-sweep
#SBATCH --output=slurm/sweep_%A_%a.out
#SBATCH --error=slurm/sweep_%A_%a.err
#SBATCH --array=0-9
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=highgpu

# Load required modules
module load apptainer
module load cuda/cuda-12.4.0

# Run all commands inside the Apptainer container
apptainer exec --nv ~/containers/train-transformer_latest.sif bash <<zzzRunHEREinTheContainer
nvidia-smi
cd ~/git/train-sae
set -a
source .env
set +a
wandb agent "$SWEEP_LINK"
exit
zzzRunHEREinTheContainer

scancel "$SLURM_JOB_ID"
