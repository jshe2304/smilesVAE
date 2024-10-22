#!/bin/bash
#SBATCH --job-name=bo
#SBATCH --account=pi-andrewferguson
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

python bo.py $1 $2
