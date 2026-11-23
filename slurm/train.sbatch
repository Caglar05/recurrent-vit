#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# One chunk of a resumable run, submitted by slurm/launch.sh as an array with %1, so chunks
# run one after another; each resumes from last.pt and saves before its time limit.
# Partition / account / node constraints come from the SBATCH_PARTITION, SBATCH_ACCOUNT and
# SBATCH_CONSTRAINT environment variables (see slurm/site.env.example).
set -euo pipefail
: "${RUN_NAME:?}" "${MODEL:?}" "${RECIPE:?}" "${OUT_DIR:?}" "${TIME_LIMIT_MIN:?}"

cd "$SLURM_SUBMIT_DIR"
if [[ -f "$OUT_DIR/$RUN_NAME/DONE" ]]; then
    echo "[sbatch] $RUN_NAME already DONE: $(cat "$OUT_DIR/$RUN_NAME/DONE")"
    exit 0
fi

# Optional HTTP(S) proxy, for clusters whose compute nodes reach the internet (wandb) only
# through one.
if [[ -n "${RVIT_PROXY:-}" ]]; then
    export HTTPS_PROXY="$RVIT_PROXY" https_proxy="$RVIT_PROXY"
    export HTTP_PROXY="$RVIT_PROXY" http_proxy="$RVIT_PROXY"
fi
export OMP_NUM_THREADS=1
PY="${PY:-python}"
NGPU="${SLURM_GPUS_ON_NODE:-1}"

echo "[sbatch] job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0} on $(hostname), $NGPU GPU(s)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "[sbatch] code: $(pwd)  $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git repo')"

# shellcheck disable=SC2086  # EXTRA_ARGS is intentionally word-split
"$PY" -m torch.distributed.run --standalone --nproc_per_node="$NGPU" -m rvit.train \
    --model "$MODEL" --recipe "$RECIPE" --run_name "$RUN_NAME" --out_dir "$OUT_DIR" \
    --time_limit_min "$TIME_LIMIT_MIN" \
    ${WANDB_PROJECT:+--wandb_project "$WANDB_PROJECT"} \
    $EXTRA_ARGS
