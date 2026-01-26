#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=72
#SBATCH --partition=gpu_a100
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/logs/%j.log
#SBATCH --error=/home/%u/logs/%j.log

VENV_PATH="/home/athamma1/Projects/viewpoint-diverse-training-diet/.venv"
SCRIPT_PATH="/home/athamma1/Projects/viewpoint-diverse-training-diet/scripts/experiments/vgg_v1.py"

source "${VENV_PATH}/bin/activate"

torchrun --nnodes=1 --nproc_per_node=4  "${SCRIPT_PATH}"

