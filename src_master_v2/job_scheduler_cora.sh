#!/bin/bash
#SBATCH -J sgnn_cora
#SBATCH --time=24:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=64g
#SBATCH -p 3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH -o sgnn_cora_%j.out
#SBATCH -e sgnn_cora_%j.err

module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate gread

module load cuda

python3 -u run_wandb.py --dataset Cora

