#!/bin/bash

### LOG INFO (edit as needed) ###
#SBATCH --job-name=DiT-sample
#SBATCH --output=logs/slurm/DiT-sample-%j.log

# Usage: 1) export CKPT=/work/nvme/beig/mfeng12/results/001-DiT-XL-2/checkpoints/0005000.pt
# Usage: 2) ./slurm_executor.sh ncsa_a100 ./job_scripts/sample.sh

module purge

echo "--------------------------------------------------"
echo "SLURM_JOB_ID:        ${SLURM_JOB_ID:-}"
echo "SLURM_NODELIST:      ${SLURM_NODELIST:-}"
echo "CUDA_VISIBLE_DEVICES:${CUDA_VISIBLE_DEVICES:-}"
echo "SLURM_SUBMIT_DIR:    ${SLURM_SUBMIT_DIR:-<unset>}"
echo "--------------------------------------------------"

# Repo root
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  REPO_ROOT="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
  REPO_ROOT="$(pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs/slurm/

# Required
: "${CKPT:?Set CKPT=/path/to/checkpoint.pt}"

# Sampling knobs (override via env vars)
: "${MODEL:=DiT-XL/2}"
: "${IMAGE_SIZE:=256}"
: "${NUM_CLASSES:=1000}"
: "${VAE:=ema}"
: "${CFG_SCALE:=1.0}"          # keep 1.0 since you didn't implement CFG
: "${NUM_SAMPLING_STEPS:=250}"
: "${GLOBAL_SEED:=0}"
: "${BATCH_SIZE:=32}"
: "${NUM_FID_SAMPLES:=50000}"
: "${SAMPLES_PARENT:="$REPO_ROOT/samples"}"

echo "Sampling checkpoint: $CKPT"
echo "SAMPLES_PARENT:     $SAMPLES_PARENT"

python -u sample.py \
  --model "$MODEL" \
  --image-size "$IMAGE_SIZE" \
  --num-classes "$NUM_CLASSES" \
  --vae "$VAE" \
  --cfg-scale "$CFG_SCALE" \
  --num-sampling-steps "$NUM_SAMPLING_STEPS" \
  --global-seed "$GLOBAL_SEED" \
  --batch-size "$BATCH_SIZE" \
  --num-fid-samples "$NUM_FID_SAMPLES" \
  --sample-dir "$SAMPLES_PARENT" \
  --ckpt "$CKPT"