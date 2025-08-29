#!/bin/bash

# Shared config
MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL_TYPE="hf"
MODEL_ARGS='{"max_length": 16384, "dtype": "float32", "attn_implementation": "eager", "low_cpu_mem_usage": true}'
n_docs=10
API_URL="http://compactds.duckdns.org:30888/search"
N_PROBE=256
SAVE_RAW=true

# Dataset list
declare -a TASKS=("minerva_math::retrieval" "mmlu:mc::retrieval" "mmlu_pro:mc::retrieval" "gpqa:0shot_cot::retrieval")

for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"
    echo "Running $TASK with exact_search=True and diverse_search=False"
    python olmes/oe_eval/run_eval.py \
        --task "$TASK" \
        --model "$MODEL" \
        --model-type "$MODEL_TYPE" \
        --model-args "$MODEL_ARGS" \
        --k $n_docs \
        --massive_serve_api "$API_URL" \
        --save-raw-requests $SAVE_RAW \
        --output-dir "output/llama3_8B_online_K1000_k10_exact" \
        --n_probe $N_PROBE \
        --retrieval_batch_size 100 \
        --exact_search
done
