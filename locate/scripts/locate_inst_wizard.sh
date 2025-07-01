#!/bin/bash
export MKL_SERVICE_FORCE_INTEL=1
models=(llama2-instruct llama2-base)
method="wandg"
type="unstructured"
suffix="weightonly"
# echo "score for safety"
# length=${#models[@]}
# for ((i = 0; i < length; i++)); do
#     save_dir="./score/safety/${models[i]}"
#     prune_data="align"

#     python main.py \
#         --model ${models[i]} \
#         --prune_method $method \
#         --prune_data $prune_data \
#         --sparsity_ratio 0.01 \
#         --sparsity_type $type \
#         --neg_prune \
#         --save $save_dir \
#         --dump_wanda_score \
#         --model_base llama2-base 
# done

# math
# models=(llama2-math llama2-base)
# length=${#models[@]}
# echo "score for math"
# for ((i = 0; i < length; i++)); do
#     save_dir="./score/math/${models[i]}"
#     prune_data="math"

#     python main.py \
#         --model ${models[i]} \
#         --prune_method $method \
#         --prune_data $prune_data \
#         --sparsity_ratio 0.01 \
#         --sparsity_type $type \
#         --neg_prune \
#         --save $save_dir \
#         --dump_wanda_score \
#         --model_base llama2-base 
# done

# code
models=(llama2-code llama2-base)
echo "score for code"
length=${#models[@]}
for ((i = 0; i < length; i++)); do
    save_dir="./score/code/${models[i]}"
    prune_data="code"

    python main.py \
        --model ${models[i]} \
        --prune_method $method \
        --prune_data $prune_data \
        --sparsity_ratio 0.01 \
        --sparsity_type $type \
        --neg_prune \
        --save $save_dir \
        --dump_wanda_score 
done


# models=(llama2-base)
# method="wandg"
# type="unstructured"
# suffix="weightonly"
# length=${#models[@]}
# for ((i = 0; i < length; i++)); do
#     save_dir="./score/math/${models[i]}"
#     prune_data="math"

#     python main.py \
#         --model ${models[i]} \
#         --prune_method $method \
#         --prune_data $prune_data \
#         --sparsity_ratio 0.01 \
#         --sparsity_type $type \
#         --neg_prune \
#         --save $save_dir \
#         --dump_wanda_score \
#         --model_base llama2-base 
# done


# models=(llama3-base)
# model_base="llama3-base"
# echo "score for code"
# length=${#models[@]}
# for ((i = 0; i < length; i++)); do
#     save_dir="./score/code/${models[i]}"
#     prune_data="code"

#     python main.py \
#         --model ${models[i]} \
#         --model_base $model_base \
#         --prune_method $method \
#         --prune_data $prune_data \
#         --sparsity_ratio 0.01 \
#         --sparsity_type $type \
#         --neg_prune \
#         --save $save_dir \
#         --dump_wanda_score 
# done



# models=(llama2-code llama2-base)
# echo "score for code"
# length=${#models[@]}
# for ((i = 0; i < length; i++)); do
#     save_dir="./score/code/${models[i]}"
#     prune_data="code"

#     python main.py \
#         --model ${models[i]} \
#         --prune_method $method \
#         --prune_data $prune_data \
#         --sparsity_ratio 0.01 \
#         --sparsity_type $type \
#         --neg_prune \
#         --save $save_dir \
#         --dump_wanda_score 
# done
