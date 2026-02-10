#!/bin/bash

cd /work/nvme/beig/mfeng12/DiT

salloc \
  --account=beig-delta-gpu \
  --partition=gpuA100x4-interactive \
  --nodes=1 \
  --ntasks-per-node=1 \
  --gpus-per-node=1 \
  --cpus-per-task=1 \
  --mem=20G \
  --time=01:00:00

# srun --pty bash

# module purge
# conda activate DiT

# export DATA_PATH=/work/nvme/beig/mfeng12/datasets/ILSVRC/Data/CLS-LOC/train
# export RESULTS_DIR=/work/nvme/beig/mfeng12/results/debug-$SLURM_JOB_ID
# export BATCH_SIZE=32

# bash ./job_scripts/train.sh