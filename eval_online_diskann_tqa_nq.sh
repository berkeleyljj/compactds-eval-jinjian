#!/usr/bin/env bash
set -euo pipefail

# Online TriviaQA + Natural Questions (Massive-Serve API) - DiskANN
# Usage:
#   ./eval_online_diskann_tqa_nq.sh
#   K=10 L=2000 W=4 THREADS=128 BATCH=100 API_URL=http://api.ds-serve.org:30888/search ./eval_online_diskann_tqa_nq.sh
#
# Notes:
# - This script mirrors eval_online_diskann.sh but targets TriviaQA and NQ.
# - DiskANN parameters are passed to the server via env → client payload.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export TRANSFORMERS_NO_TORCHVISION=1
export PYTHONPATH="/home/ubuntu/compactds-eval-jinjian/olmes:/home/ubuntu/compactds-eval-jinjian:${PYTHONPATH:-}"
[[ -n "${HF_TOKEN:-}" ]] && export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

# Tasks (array expansion to pass multiple --task values)
TASKS=("triviaqa::olmes" "naturalqs::olmes")

MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL_TYPE="hf"
MODEL_ARGS='{"max_length": 16384}'

K="${K:-10}"
BATCH="${BATCH:-100}"
SEED="${SEED:-2025}"
API_URL="${API_URL:-http://api.ds-serve.org:30888/search}"

# DiskANN params (defaults align with your preference)
L="${L:-2000}"
W="${W:-4}"
THREADS="${THREADS:-64}"

OUT_DIR="output/llama-8B-tqa_nq-k=${K}-diskann-online-L${L}-W${W}-T${THREADS}"

# Tell retriever to use DiskANN and pass params
export MS_BACKEND=diskann
export DISKANN_L="$L"
export DISKANN_W="$W"
export DISKANN_THREADS="$THREADS"

echo "[Eval] Online TriviaQA + NQ (DiskANN)"
echo "[Eval] Tasks=${TASKS[*]}  K=$K  Batch=$BATCH  API=$API_URL  L=$L W=$W T=$THREADS"

python olmes/oe_eval/run_eval.py \
  --task "${TASKS[@]}" \
  --model "$MODEL" \
  --model-type "$MODEL_TYPE" \
  --model-args "$MODEL_ARGS" \
  --k "$K" \
  --massive_serve_api "$API_URL" \
  --retrieval_batch_size "$BATCH" \
  --save-raw-requests true \
  --output-dir "$OUT_DIR" \
  --random-subsample-seed "$SEED"

echo "[Eval] Done. Outputs → $OUT_DIR"


