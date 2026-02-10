### ADDITIONAL RUN INFO ###
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
# Optional: turn this into an array job for sweeps (edit as needed)
# #SBATCH --array=0

### LOG INFO ###
#SBATCH --job-name=DiT-XL2-imagenet256
#SBATCH --output=logs/slurm/DiT-XL2-imagenet256-%j.log
export RUN_NAME="DiT-XL2-imagenet256"
# NOTE ctrl d ALL THREE of above to modify job-name, output, and RUN_NAME (which should all be the same)
export MODEL_NAME="${RUN_NAME%%-*}"
export MODEL_SIZE="${RUN_NAME#*-}"; export MODEL_SIZE="${MODEL_SIZE%%-*}"
module purge


echo "--------------------------------------------------"
echo "RUN_NAME:            $RUN_NAME"
echo "MODEL_NAME:          $MODEL_NAME"
echo "MODEL_SIZE:          $MODEL_SIZE"
echo "SLURM_JOB_ID:        ${SLURM_JOB_ID:-}"
echo "SLURM_NODELIST:      ${SLURM_NODELIST:-}"
echo "CUDA_VISIBLE_DEVICES:${CUDA_VISIBLE_DEVICES:-}"
echo "--------------------------------------------------"

############################################
# Paths / knobs (edit these)
############################################
# Set your repo root (common pattern: job_configs/ is inside the repo)
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  REPO_ROOT="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
  REPO_ROOT="$(pwd)"
fi
cd "$REPO_ROOT"

mkdir -p logs/slurm/

# REQUIRED: point to your ImageNet train folder (or export DATA_PATH before submission)
: "${DATA_PATH:="$REPO_ROOT/../datasets/ILSVRC/Data/CLS-LOC/train"}"

# Results directory (override if you want)
: "${RESULTS_DIR:="$REPO_ROOT/../results"}"

# Batch size for single-GPU training script (override if you want)
: "${BATCH_SIZE:=32}"

############################################
# Train command (single GPU, no DDP)
############################################
python train.py \
  --model DiT-XL/2 \
  --data-path "$DATA_PATH" \
  --results-dir "$RESULTS_DIR" \
  --image-size 256 \
  --global-batch-size "$BATCH_SIZE" \
  --num-workers 1