#!/bin/bash
#SBATCH --job-name=train_vae_jay
#SBATCH --error=train.err
#SBATCH --account=pi-andrewferguson
#SBATCH --time=10:00:00
#SBATCH --partition=caslake
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=2000

conda init bash
conda activate ./env
python train.py
