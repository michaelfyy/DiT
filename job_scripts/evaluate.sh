#!/bin/bash

### ADDITIONAL RUN INFO (edit as needed) ###
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1

### LOG INFO ###
#SBATCH --job-name=DiT-eval
#SBATCH --output=logs/slurm/DiT-eval-%j.log

### USAGE ###
# export CKPT_DIR=/work/nvme/beig/mfeng12/results/001-DiT-XL-2/checkpoints
# optional: export EVAL_PY=/path/to/guided-diffusion/evaluations/evaluator.py
# optinal: export REF_NPZ=/path/to/VIRTUAL_imagenet256_labeled.npz
# ./slurm_executor.sh ncsa_a100 ./job_scripts/evaluate.sh

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

############################################
# Required inputs (set via env vars)
############################################
: "${CKPT_DIR:?Set CKPT_DIR=/work/nvme/beig/mfeng12/results/.../checkpoints}"

############################################
# Sampling / evaluation knobs (override via env vars)
############################################
: "${EVAL_PY:="$REPO_ROOT/../guided-diffusion/evaluations/evaluator.py"}"
: "${REF_NPZ:="$REPO_ROOT/../datasets/VIRTUAL_imagenet256_labeled.npz"}"
: "${MODEL:=DiT-XL/2}"
: "${IMAGE_SIZE:=256}"
: "${NUM_CLASSES:=1000}"
: "${VAE:=ema}"
: "${CFG_SCALE:=1.0}"
: "${NUM_SAMPLING_STEPS:=250}"
: "${GLOBAL_SEED:=0}"
: "${BATCH_SIZE:=32}"
: "${NUM_FID_SAMPLES:=5000}"
: "${SAMPLES_PARENT:="$REPO_ROOT/samples/fid_sweep"}"
: "${CSV_OUT:="$REPO_ROOT/results/fid_sweep.csv"}"
: "${KEEP_PNG:=1}"

mkdir -p "$(dirname "$CSV_OUT")"
mkdir -p "$SAMPLES_PARENT"

# CSV header if needed (or upgrade existing header to include sfid)
if [ ! -f "$CSV_OUT" ]; then
  echo "step,ckpt_path,npz_path,num_samples,num_sampling_steps,seed,fid,sfid" > "$CSV_OUT"
else
  header="$(head -n 1 "$CSV_OUT")"
  if [[ "$header" != *",sfid"* ]]; then
    # Append sfid to header line
    sed -i '1s/$/,sfid/' "$CSV_OUT"
  fi
fi

echo "CKPT_DIR:       $CKPT_DIR"
echo "SAMPLES_PARENT: $SAMPLES_PARENT"
echo "CSV_OUT:        $CSV_OUT"
echo "EVAL_PY:        $EVAL_PY"
echo "REF_NPZ:        $REF_NPZ"
echo "--------------------------------------------------"

MODEL_STRING="${MODEL//\//-}"   # DiT-XL/2 -> DiT-XL-2

shopt -s nullglob
CKPTS=( "$CKPT_DIR"/*.pt )
if [ ${#CKPTS[@]} -eq 0 ]; then
  echo "No checkpoints found in: $CKPT_DIR"
  exit 1
fi

IFS=$'\n' CKPTS_SORTED=($(printf "%s\n" "${CKPTS[@]}" | sort))
unset IFS

for CKPT in "${CKPTS_SORTED[@]}"; do
  BASE="$(basename "$CKPT" .pt)"
  STEP=$((10#$BASE))

  ROW_PREFIX="${STEP},${CKPT},${NPZ_PATH},${NUM_FID_SAMPLES},${NUM_SAMPLING_STEPS},${GLOBAL_SEED},${CFG_SCALE},${VAE},${MODEL},${IMAGE_SIZE},"
  if grep -Fq "$ROW_PREFIX" "$CSV_OUT"; then
    echo "[SKIP] exact run already in CSV"
    continue
  fi

  CKPT_STRING="$BASE"
  FOLDER_NAME="${MODEL_STRING}-${CKPT_STRING}-size-${IMAGE_SIZE}-vae-${VAE}-cfg-${CFG_SCALE}-seed-${GLOBAL_SEED}"
  SAMPLE_FOLDER_DIR="${SAMPLES_PARENT}/${FOLDER_NAME}"
  NPZ_PATH="${SAMPLE_FOLDER_DIR}.npz"

  echo "=================================================="
  echo "step=$STEP ckpt=$CKPT"
  echo "npz=$NPZ_PATH"
  echo "=================================================="

  # 1) Sample (if npz not present)
  if [ ! -f "$NPZ_PATH" ]; then
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
  else
    echo "[OK] Found existing NPZ, skipping sampling."
  fi

  if [ ! -f "$NPZ_PATH" ]; then
    echo "[ERROR] NPZ not found after sampling: $NPZ_PATH"
    continue
  fi

  # 2) Evaluate (capture stdout+stderr so we don't miss metric lines)
  EVAL_LOG="$SAMPLES_PARENT/eval_step_${STEP}.log"
  export EVAL_LOG
  python -u "$EVAL_PY" "$REF_NPZ" "$NPZ_PATH" 2>&1 | tee "$EVAL_LOG"

  # 3) Extract BOTH FID and sFID (anchor to line-start so FID != sFID)
  IFS=$'\t' read -r FID SFID <<< "$(
python - <<'PY'
import os, re
path = os.environ["EVAL_LOG"]
text = open(path, "r", errors="ignore").read()

num = r'([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)'
fid  = re.findall(rf'(?m)^\s*FID:\s*{num}\s*$',  text)
sfid = re.findall(rf'(?m)^\s*sFID:\s*{num}\s*$', text)

print((fid[-1] if fid else "") + "\t" + (sfid[-1] if sfid else ""))
PY
)"
  unset IFS

  if [ -z "$FID" ]; then
    echo "[ERROR] Could not parse FID from $EVAL_LOG"
    continue
  fi
  if [ -z "$SFID" ]; then
    echo "[WARN] Could not parse sFID from $EVAL_LOG (will write empty)"
  fi

  # 4) Append to CSV
  echo "${STEP},${CKPT},${NPZ_PATH},${NUM_FID_SAMPLES},${NUM_SAMPLING_STEPS},${GLOBAL_SEED},${FID},${SFID}" >> "$CSV_OUT"
  echo "[OK] step=$STEP FID=$FID sFID=$SFID appended to $CSV_OUT"

  if [ "$KEEP_PNG" = "0" ]; then
    echo "[CLEAN] Removing PNG folder: $SAMPLE_FOLDER_DIR"
    rm -rf "$SAMPLE_FOLDER_DIR"
  fi
done

echo "--------------------------------------------------"
echo "Done. CSV at: $CSV_OUT"
echo "--------------------------------------------------"
