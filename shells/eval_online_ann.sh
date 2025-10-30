#!/usr/bin/env bash
set -euo pipefail

# Online AGI Eval (Massive-Serve API) - ANN only (no exact reranking)
# Fast sanity check for retrieval pipeline correctness.
#
# Usage:
#   ./eval_online_ann.sh
#   K=10 BATCH=100 API_URL=http://128.208.4.44:30888/search ./eval_online_ann.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export TRANSFORMERS_NO_TORCHVISION=1
export PYTHONPATH="/home/ubuntu/jinjian/eval/olmes:/home/ubuntu/jinjian/eval:${PYTHONPATH:-}"

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

TASK="agi_eval_english::retrieval"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL_TYPE="hf"
MODEL_ARGS='{"max_length": 16384}'
K="${K:-10}"
SEED="${SEED:-2025}"
API_URL="${API_URL:-http://api.ds-serve.org:30888/search}"

# ANN can use larger batch safely
BATCH="${BATCH:-100}"

OUT_DIR="output/llama-8B-agi-k=${K}-ann-online"

echo "[Eval] Online AGI Eval (ANN only)"
echo "[Eval] Task=$TASK  K=$K  Batch=$BATCH  API=$API_URL"

python olmes/oe_eval/run_eval.py \
  --task "$TASK" \
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


