#!/bin/bash
export MKL_SERVICE_FORCE_INTEL=1
method="wandg"
type="unstructured"
suffix="weightonly"

# safety
models=(mistral-instruct mistral-base)
model_base="mistral-base"
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
models=(mistral-math mistral-base)
model_base="mistral-base"
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

