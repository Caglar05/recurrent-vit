#!/bin/bash
# Submit one resumable training run to SLURM as a chain of short chunks.
#
#   MODEL=bvit-b RECIPE=bvit SEED=0 bash slurm/launch.sh     # full run
#   MODEL=bvit-b RECIPE=bvit BENCH=1 bash slurm/launch.sh    # 30-min throughput / memory probe
#
# Cluster-specific settings (account, partition, data paths, Python, proxy) are read from
# slurm/site.env if it exists: copy slurm/site.env.example to slurm/site.env and edit it.
# Every variable below can also be set on the command line. Extra trainer flags go in
# EXTRA_ARGS, e.g. EXTRA_ARGS="--model.n_experts 8 --batch_per_gpu 128".
#
# The run is an array of CHUNKS jobs that run one after another (%1). Each chunk resumes
# from <OUT_DIR>/<RUN_NAME>/last.pt and saves before its time limit; chunks that start
# after the run has finished exit within seconds (DONE marker).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -f slurm/site.env ]]; then
    set -a
    # shellcheck source=/dev/null
    source slurm/site.env
    set +a
fi

MODEL="${MODEL:?set MODEL, e.g. bvit-b  ((vit|bvit|revit|deit3)-(s|b|l))}"
RECIPE="${RECIPE:?set RECIPE: bvit | deit3}"
SEED="${SEED:-0}"
NGPU="${NGPU:-4}"
GPU_TYPE="${GPU_TYPE:-}"               # e.g. A100; empty = any GPU type
CHUNKS="${CHUNKS:-30}"                 # upper bound on chunks; surplus ones no-op
TIME="${TIME:-0-02:00:00}"             # wall time per chunk
TIME_LIMIT_MIN="${TIME_LIMIT_MIN:-100}" # trainer saves + exits after this (compile/eval/save margin)
MEM="${MEM:-400G}"
CPUS="${CPUS:-$((16 * NGPU))}"
OUT_DIR="${OUT_DIR:-$PWD/runs}"
WANDB_PROJECT="${WANDB_PROJECT-recurrent-vit}"   # set to "" to disable wandb
EXTRA_ARGS="${EXTRA_ARGS:-}"
RUN_NAME="${RUN_NAME:-${MODEL}_${RECIPE}_s${SEED}}"
EXTRA_ARGS="--seed $SEED $EXTRA_ARGS"

if [[ "${BENCH:-0}" == 1 ]]; then      # short run: measures img/s + peak memory, no DONE chain
    RUN_NAME="bench_${RUN_NAME}_$(date +%m%d-%H%M)"
    CHUNKS=1; TIME=0-00:30:00; TIME_LIMIT_MIN=20; WANDB_PROJECT="${WANDB_PROJECT:+${WANDB_PROJECT}-bench}"
    EXTRA_ARGS="$EXTRA_ARGS --log_every 50 --eval_every_epochs 1000"
fi

LOG_DIR="$OUT_DIR/$RUN_NAME/slurm"
mkdir -p "$LOG_DIR"
export RUN_NAME MODEL RECIPE OUT_DIR TIME_LIMIT_MIN WANDB_PROJECT EXTRA_ARGS
echo "submitting $RUN_NAME: $MODEL / $RECIPE, ${NGPU}x ${GPU_TYPE:-GPU}, $CHUNKS chunk(s) of $TIME"
sbatch \
    --job-name="rvit-$RUN_NAME" \
    --gpus-per-node="${GPU_TYPE:+$GPU_TYPE:}$NGPU" \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="$TIME" \
    --array="0-$((CHUNKS - 1))%1" \
    --output="$LOG_DIR/job-%A_%a.log" \
    --export=ALL \
    slurm/train.sbatch
