#!/bin/bash
#SBATCH --job-name=train_vae_jay
#SBATCH --output=train_output/train.out
#SBATCH --error=train_output/train.err
#SBATCH --account=pi-andrewferguson
#SBATCH --time=05:00:00
#SBATCH --partition=caslake
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=2000

conda init bash
conda activate ./env
python train.py
