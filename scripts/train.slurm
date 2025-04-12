#!/bin/bash
#SBATCH --job-name=sae-train
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --output=slurm/train-%j.out
#SBATCH --error=slurm/train-%j.out

# Load required modules
module load apptainer
module load cuda/cuda-12.4.0

# Run all commands inside the Apptainer container
apptainer exec --nv /lustre/fs1/groups/gsukthankar/containers/train-transformer_latest.sif bash <<zzzRunHEREinTheContainer
nvidia-smi
cd ~/git/train-saes
python src/train.py
exit
zzzRunHEREinTheContainer

scancel "$SLURM_JOB_ID"

