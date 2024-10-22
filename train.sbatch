#!/bin/bash
#SBATCH --job-name=vae-mixed-64
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --account=pi-andrewferguson
#SBATCH --time=36:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

python train.py $1
