#!/usr/bin/env bash
set -euo pipefail

# Online GPQA (Massive-Serve API) - DiskANN
# Usage:
#   ./eval_online_diskann_gpqa.sh
#   K=10 L=150 W=8 THREADS=8 BATCH=100 API_URL=http://128.208.4.44:30888/search ./eval_online_diskann_gpqa.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export TRANSFORMERS_NO_TORCHVISION=1
export PYTHONPATH="/home/ubuntu/jinjian/eval/olmes:/home/ubuntu/jinjian/eval:${PYTHONPATH:-}"
[[ -n "${HF_TOKEN:-}" ]] && export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

TASK="gpqa:0shot_cot::retrieval"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL_TYPE="hf"
MODEL_ARGS='{"max_length": 16384}'

K="${K:-10}"
BATCH="${BATCH:-100}"
SEED="${SEED:-2025}"
API_URL="${API_URL:-http://128.208.4.44:30888/search}"

# DiskANN params (align defaults with current preference)
L="${L:-2000}"
W="${W:-8}"
THREADS="${THREADS:-8}"

OUT_DIR="output/llama-8B-gpqa-k=${K}-diskann-online-L${L}-W${W}-T${THREADS}"

# Tell retriever to use DiskANN and pass params
export MS_BACKEND=diskann
export DISKANN_L="$L"
export DISKANN_W="$W"
export DISKANN_THREADS="$THREADS"

echo "[Eval] Online GPQA (DiskANN)"
echo "[Eval] Task=$TASK  K=$K  Batch=$BATCH  API=$API_URL  L=$L W=$W T=$THREADS"

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


