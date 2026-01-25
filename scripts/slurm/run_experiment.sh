#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=05:00:00
#SBATCH --output=/home/%u/logs/%j.log
#SBATCH --error=/home/%u/logs/%j.log

VENV_PATH="/home/athamma1/Projects/viewpoint-diverse-training-diet/.venv"
SCRIPT_PATH="/home/athamma1/Projects/viewpoint-diverse-training-diet/scripts/experiments/vgg_v1.py"

source "${VENV_PATH}/bin/activate"

python "${SCRIPT_PATH}" 

