#!/bin/bash
#SBATCH --job-name=sae-train
#SBATCH --time=6:00:00
#SBATCH --partition=highgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
##SBATCH --nodelist=evc103
#SBATCH --mem=800G
#SBATCH --output=slurm/train-%j.out
#SBATCH --error=slurm/train-%j.out
    ##SBATCH --exclusive
    ##SBATCH --nodes=1

# Load required modules
module load apptainer
module load cuda/cuda-12.4.0

# Run all commands inside the Apptainer container
apptainer exec --nv ~/containers/train-transformer_latest.sif bash <<zzzRunHEREinTheContainer
nvidia-smi
pwd
ls
cd ~/git/train-saes
pwd
ls
python src/train.py
exit
zzzRunHEREinTheContainer

scancel "$SLURM_JOB_ID"
