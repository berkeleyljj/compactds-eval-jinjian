#!/bin/bash

# Shared config
MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL_TYPE="hf"
MODEL_ARGS='{"max_length": 16384, "torch_dtype": "bfloat16", "attn_implementation": "flash_attention_2", "low_cpu_mem_usage": true}'
n_docs=10
API_URL="http://localhost:30888/search"
N_PROBE=256
SAVE_RAW=true

# Dataset list
declare -a TASKS=("agi_eval_english::retrieval")

for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"
    echo "Running $TASK with exact_search=False and diverse_search=False (both disabled by default)"
    python olmes/oe_eval/run_eval.py \
        --task "$TASK" \
        --model "$MODEL" \
        --model-type "$MODEL_TYPE" \
        --model-args "$MODEL_ARGS" \
        --k $n_docs \
        --massive_serve_api "$API_URL" \
        --save-raw-requests $SAVE_RAW \
        --output-dir "output/llama3_8B_8.19" \
        --n_probe $N_PROBE \
        --retrieval_batch_size 100
done
