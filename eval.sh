#!/bin/bash

# Enable GPU cache cleaning after each batch to prevent OOM
export OE_EVAL_GPU_EMPTY_CACHE=1

# Shared config
MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL_TYPE="hf"
MODEL_ARGS='{"max_length": 16384, "dtype": "float32", "attn_implementation": "eager", "low_cpu_mem_usage": true}'
n_docs=10
API_URL="http://192.222.59.156:30888/search"
N_PROBE=256
SAVE_RAW=true
 
# Dataset list
declare -a TASKS=("gpqa:0shot_cot::retrieval" "minerva_math::retrieval" "mmlu:mc::retrieval")

for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"
    LAMBDA_VAL=0.25
    python olmes/oe_eval/run_eval.py \
        --task "$TASK" \
        --model "$MODEL" \
        --model-type "$MODEL_TYPE" \
        --model-args "$MODEL_ARGS" \
        --k $n_docs \
        --massive_serve_api "$API_URL" \
        --save-raw-requests $SAVE_RAW \
        --output-dir "output/llama3_8B_online_K1000_k10_diverse_0.5" \
        --n_probe $N_PROBE \
        --retrieval_batch_size 100 \
        --diverse_search \
        --lambda_val $LAMBDA_VAL
done
