#!/bin/bash

# Shared config
MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL_TYPE="hf"
MODEL_ARGS='{"max_length": 16384}'
K=10
API_URL="http://192.222.59.156:30888/search"
N_PROBE=32
SAVE_RAW=true

# Dataset list
declare -a TASKS=("minerva_math::retrieval" "mmlu:mc::retrieval" "mmlu_pro:mc::retrieval")

for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"

    echo "Running $TASK with ANN-only"
    python olmes/oe_eval/run_eval.py \
        --task "$TASK" \
        --model "$MODEL" \
        --model-type "$MODEL_TYPE" \
        --model-args "$MODEL_ARGS" \
        --k $K \
        --massive_serve_api "$API_URL" \
        --save-raw-requests $SAVE_RAW \
        --output-dir "output/llama3_8B_ANN_only" \
        --n_probe $N_PROBE

    echo "Running $TASK with exact search"
    python olmes/oe_eval/run_eval.py \
        --task "$TASK" \
        --model "$MODEL" \
        --model-type "$MODEL_TYPE" \
        --model-args "$MODEL_ARGS" \
        --k $K \
        --massive_serve_api "$API_URL" \
        --save-raw-requests $SAVE_RAW \
        --output-dir "output/llama3_8B_exact_search" \
        --exact_search True \
        --n_probe $N_PROBE
done
