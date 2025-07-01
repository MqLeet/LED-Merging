#!/bin/bash
export MKL_SERVICE_FORCE_INTEL=1
method="wandg"
type="unstructured"
suffix="weightonly"

# code
models=(qwen2.5-code qwen2.5-base)
model_base="qwen2.5-base"
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
models=(qwen2.5-instruct qwen2.5-base)
model_base="qwen2.5-base"
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
models=(qwen2.5-math qwen2.5-base)
model_base="qwen2.5-base"
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

