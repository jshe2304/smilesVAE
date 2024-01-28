#!/bin/bash
#SBATCH --job-name=train_vae_jay
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --account=pi-andrewferguson
#SBATCH --time=30:00:00
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem-per-cpu=2000
#SBATCH --exclusive

conda init bash
conda activate ./env
python train.py
