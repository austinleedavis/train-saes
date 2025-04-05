#!/bin/bash
#SBATCH --job-name=train-sae
#SBATCH --time=24:00:00
#SBATCH --partition=highgpu
#SBATCH --gres=gpu:8
#SBATCH --exclusive
    ##SBATCH --cpus-per-task=64
    ##SBATCH --nodes=1
#SBATCH --output=slurm/train-%j.out
#SBATCH --error=slurm/train-%j.out

# Load required modules
module load apptainer
module load cuda/cuda-12.4.0

# Run all commands inside the Apptainer container
apptainer exec --nv ~/containers/train-transformer_latest.sif bash <<zzzRunHEREinTheContainer
nvidia-smi
cd ~/git/train-sae
python src/train.py
exit
zzzRunHEREinTheContainer

scancel "$SLURM_JOB_ID"
