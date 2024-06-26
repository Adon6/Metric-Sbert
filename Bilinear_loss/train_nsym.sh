#!/bin/bash
#SBATCH --partition=student
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --job-name=nsym-training
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-logs/slurm.%j.%a.out
#SBATCH --error=slurm-logs/slurm.%j.%a.err
#SBATCH --array=0-3

# Define model names and corresponding batch sizes explicitly
declare -a models=("bert-base-uncased" "sentence-transformers/all-mpnet-base-v2" "sentence-transformers/all-distilroberta-v1" "roberta-base")
declare -a batchsizes=(32 64 160 32)  # Assign a specific batch size to each model

# Use SLURM_ARRAY_TASK_ID to select model and batch size
model=${models[$SLURM_ARRAY_TASK_ID]}
batchsize=${batchsizes[$SLURM_ARRAY_TASK_ID]}

# Execute the training command with the selected model and batch size
python train_nsym_type.py $model $batchsize
