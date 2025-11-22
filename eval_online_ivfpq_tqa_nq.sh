#!/usr/bin/env bash
set -euo pipefail

# Online TriviaQA + Natural Questions (Massive-Serve API) - IVFPQ
# Usage:
#   ./eval_online_ivfpq_tqa_nq.sh
#   K=10 NPROBE=256 BATCH=100 API_URL=http://api.ds-serve.org:30888/search ./eval_online_ivfpq_tqa_nq.sh
#
# Notes:
# - Mirrors eval_online_diskann_tqa_nq.sh but targets IVFPQ with configurable nprobe.

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

# IVFPQ param
NPROBE="${NPROBE:-256}"

OUT_DIR="output/llama-8B-tqa_nq-k=${K}-ivfpq-online-nprobe${NPROBE}"

# Tell retriever to use IVFPQ
export MS_BACKEND=ivfpq

echo "[Eval] Online TriviaQA + NQ (IVFPQ)"
echo "[Eval] Tasks=${TASKS[*]}  K=$K  Batch=$BATCH  API=$API_URL  nprobe=$NPROBE"

python olmes/oe_eval/run_eval.py \
  --task "${TASKS[@]}" \
  --model "$MODEL" \
  --model-type "$MODEL_TYPE" \
  --model-args "$MODEL_ARGS" \
  --k "$K" \
  --massive_serve_api "$API_URL" \
  --retrieval_batch_size "$BATCH" \
  --n_probe "$NPROBE" \
  --save-raw-requests true \
  --output-dir "$OUT_DIR" \
  --random-subsample-seed "$SEED"

echo "[Eval] Done. Outputs → $OUT_DIR"


