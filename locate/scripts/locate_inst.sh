#!/bin/bash
export MKL_SERVICE_FORCE_INTEL=1
# models=(llama3-instruct llama3-base)
method="wandg"
type="unstructured"
suffix="weightonly"
# length=${#models[@]}
# echo "score for instruction following"
# for ((i = 0; i < length; i++)); do
#     save_dir="./score/alpaca/${models[i]}"
#     prune_data="alpaca_cleaned_no_safety"

#     python main.py \
#         --model ${models[i]} \
#         --prune_method $method \
#         --prune_data $prune_data \
#         --sparsity_ratio 0.01 \
#         --sparsity_type $type \
#         --neg_prune \
#         --save $save_dir \
#         --dump_wanda_score \
#         --use_diff 
# done

# code
models=(llama3-code llama3-base)
model_base="llama3-base"
echo "score for code"
length=${#models[@]}
for ((i = 0; i < length; i++)); do
    save_dir="./score/code/${models[i]}"
    prune_data="code"

    python main.py \
        --model ${models[i]} \
        --model_base $model_base \
        --prune_method $method \
        --prune_data $prune_data \
        --sparsity_ratio 0.01 \
        --sparsity_type $type \
        --neg_prune \
        --save $save_dir \
        --dump_wanda_score 
done

# safety
models=(llama3-instruct llama3-base)
model_base="llama3-base"
echo "score for safety"
length=${#models[@]}
for ((i = 0; i < length; i++)); do
    save_dir="./score/safety/${models[i]}"
    prune_data="align"

    python main.py \
        --model ${models[i]} \
        --model_base $model_base \
        --prune_method $method \
        --prune_data $prune_data \
        --sparsity_ratio 0.01 \
        --sparsity_type $type \
        --neg_prune \
        --save $save_dir \
        --dump_wanda_score 
done

# math
models=(llama3-math llama3-base)
model_base="llama3-base"
echo "score for math"
length=${#models[@]}
for ((i = 0; i < length; i++)); do
    save_dir="./score/math/${models[i]}"
    prune_data="math"

    python main.py \
        --model ${models[i]} \
        --model_base $model_base \
        --prune_method $method \
        --prune_data $prune_data \
        --sparsity_ratio 0.01 \
        --sparsity_type $type \
        --neg_prune \
        --save $save_dir \
        --dump_wanda_score 
done