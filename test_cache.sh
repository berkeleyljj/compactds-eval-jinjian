#!/bin/bash

# Shared config
MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL_TYPE="hf"
MODEL_ARGS='{"max_length": 32768}'
n_docs=10
API_URL="http://192.222.59.156:30888/search"
N_PROBE=256
SAVE_RAW=true

# Dataset list
declare -a TASKS=("agi_eval_english::retrieval")

for TASK in "${TASKS[@]}"; do
#     echo "=============================="
#     echo "Running $TASK with exact search (First Run, no cache)"
#     echo "Start time: $(date)"
#     SECONDS=0

#     python olmes/oe_eval/run_eval.py \
#         --task "$TASK" \
#         --model "$MODEL" \
#         --model-type "$MODEL_TYPE" \
#         --model-args "$MODEL_ARGS" \
#         --k $n_docs \
#         --massive_serve_api "$API_URL" \
#         --save-raw-requests $SAVE_RAW \
#         --output-dir "output/llama3_8B_exact_search_7.20" \
#         --exact_search True \
#         --n_probe $N_PROBE \
#         --retrieval_batch_size 5 \
#         --batch-size 5

#     echo "First run completed in $SECONDS seconds"
#     echo "End time: $(date)"
#     echo ""

    echo "------------------------------"
    echo "Running $TASK with exact search (Second Run, should use cache)"
    echo "Start time: $(date)"
    SECONDS=0

    python olmes/oe_eval/run_eval.py \
        --task "$TASK" \
        --model "$MODEL" \
        --model-type "$MODEL_TYPE" \
        --model-args "$MODEL_ARGS" \
        --k $n_docs \
        --massive_serve_api "$API_URL" \
        --save-raw-requests $SAVE_RAW \
        --output-dir "output/llama3_8B_exact_search_7.20_cached" \
        --exact_search True \
        --n_probe $N_PROBE \
        --retrieval_batch_size 5 \
        --batch-size 5

    echo "Second run completed in $SECONDS seconds"
    echo "End time: $(date)"
    echo "=============================="
    echo ""
done
