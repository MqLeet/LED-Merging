<div align=center>
  
# **[ACL 2025 Main]** LED-Merging: Mitigating Safety-Utility Conflicts in Model Merging with Location-Election-Disjoint

<p>
<a href='https://arxiv.org/abs/2502.16770'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>
<!-- <a href='https://mqleet.github.io/DeMe_Project/'><img src='https://img.shields.io/badge/Project-Page-blue'></a> -->
</p>

</div>

## 🔥 News

- [2025/05/16] LED-Merging has been accepted to ACL 2025 main conference 🎉🎉🎉
- [2025/06/13] Code has been released 🔥🔥🔥

## 📝 TODO

- [x] Merge code
- [x] Inference code
- [ ] Project page


## 🛠️ Getting Started
### 1. Setup
```bash
git clone https://github.com/MqLeet/LED-Merging.git
cd LED-Merging
conda create -n led python==3.12
conda activate led
pip install -r requirements.txt
```

### 2. Model Preparation

The used pretrained models can be found here:

| Pretrained Backbones | SFT LLMs                                                                               |
| -------------------- | -------------------------------------------------------------------------------------- |
| [**Meta-Llama-3-8B**](https://huggingface.co/meta-llama/Meta-Llama-3-8B)  | [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |
|                      | [MAmmoTH2-8B-Plus](https://huggingface.co/TIGER-Lab/MAmmoTH2-8B-Plus)                  |
|                      | [Replete-Coder-Llama3-8B](https://huggingface.co/Replete-AI/Replete-Coder-Llama3-8B)   |
| [**Mistral-7B**](https://huggingface.co/mistralai/Mistral-7B-v0.1)       | [Mistral-7B-Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)       |
|                      | [MetaMath-Mistral-7B](https://huggingface.co/meta-math/MetaMath-Mistral-7B)            |
| [**Llama-2-13B**](https://huggingface.co/meta-llama/Llama-2-13b-hf)      | [WizardLM-13B](https://huggingface.co/WizardLM/WizardLM-13B-V1.2)                      |
|                      | [WizardMath-13B](https://huggingface.co/WizardLM/WizardMath-13B-V1.0)                  |
|                      | [llama-2-13b-code-alpaca](https://huggingface.co/layoric/llama-2-13b-code-alpaca)      |

### 3. Dataset / Benchmark Preparation

* Code generation Benchmark: [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness)

* SafetyGuards Benchmark: [HarmBench](https://github.com/centerforaisafety/HarmBench.git)






## 🚀 Run and evaluation 
We use Llama3-8B as an example to demonstrate the workflow of LED-Merging:

### 🔎 *Locate*
To get the importance scores in Equation 1 in paper, run the commands below:
```bash
cd locate/
bash scritps/locate_inst.sh
```


### 🗳️ *Elect*


Then, to select important neurons based on the importance scores, run the commands below:

```bash
mask_pattern=11
model_type="llama3"
rates=(0.1 0.4 0.5)

python mask_generate.py ${rates[0]} ${rates[1]} ${rates[2]} $mask_pattern $model_type
```


###  🪡 *Disjoint and Merging*
Finally, Disjoint conflict neruons and implement model merging:
```bash
model_type="llama3"
base_model="llama3-base"
orders=(safety math code)
fuse_types=(o o o o)
lambdas=(1.0 1.0 1.0)
fuse_o=11
alphas=(0.9)

python merge_llms.py \
    --models_to_merge llama3-instruct llama3-math llama3-code \
    --pretrained_model_name $base_model \
    --merging_method_name top_merging \
    --scaling_coefficient ${alphas[0]} \
    --mask_apply_method task_arithmetic \
    --fuse_rates ${mask_rates[0]} ${mask_rates[1]} ${mask_rates[2]} \
    --orders ${orders[0]} ${orders[1]} ${orders[2]} \
    --lambdas ${lambdas[0]} ${lambdas[1]} ${lambdas[2]} \
    --fuse_types ${fuse_types[0]} ${fuse_types[1]} ${fuse_types[2]} \
    --fuse_patterns $fuse_o $fuse_o $fuse_o \
    --model_ft_name $model_type \
    --model_base_name $base_model
```


## 🧪 Inference

### Test on SafetyGuards
First, add items into HarmBench/configs/model_configs/models.yaml, like this:
```yaml
llama3_8b_base_inst_math_code_task_0.9:
  model:
    model_name_or_path: save_merge_llms/llama3-base/llama3-instruct_llama3-math/task_arithmetic_scaling_coefficient_0.9/
    dtype: float16
    chat_template: llama-3
  num_gpus: 1
  model_type: open_source
```

1. HarmBench
Generate responses
```bash

# HarmBench
cd HarmBench


## generate harmbench
export MKL_SERVICE_FORCE_INTEL=1
chat_template="llama3"
dataset="harmbench"
behaviors_path="./data/behavior_datasets/harmbench_behaviors_text_all.csv"
test_cases_path="./data/behavior_datasets/harmbench_behaviors_text_all_results/test_cases.json"
trust_remote_code="True"
max_new_tokens=512
model_name_or_path="save_merge_llms/llama3-base/llama3-instruct_llama3-math/task_arithmetic_scaling_coefficient_0.9/"
model_name="model_name"

python -u generate_completions.py \
    --model_name $model_name \
    --behaviors_path $behaviors_path \
    --test_cases_path $test_cases_path \
    --save_path $save_path \
    --chat_template $chat_template \
    --max_new_tokens $max_new_tokens --dataset $dataset --model_name_or_path $model_name_or_path \
    --trust_remote_code --generate_with_vllm

cd ./completions
python merge.py harmbench/$model_name.json
cd ../
```

Evaluation:
```bash
dataset="harmbench"
cls_path="/path/to/HarmBench-Llama-2-13b-cls"
behaviors_path="./data/behavior_datasets/harmbench_behaviors_text_all.csv"
eval_type="harmbench"
completions_path= "/path/to/generated/reponses"

python -u evaluate_completions.py \
    --cls_path $cls_path \
    --behaviors_path $behaviors_path \
    --completions_path $completions_path \
    --save_path $save_path \
    --eval_type $eval_type
```

2. SORRY-Bench


Generate Reponses:
```bash
dataset="sorrybench"
test_cases_path="./data/sorrybench/test_cases.json"
behaviors_path="./data/behavior_datasets/harmbench_behaviors_text_all.csv"
chat_template="llama3"
model_name="model_name"
model_name_or_path="save_merge_llms/llama3-base/llama3-instruct_llama3-math/task_arithmetic_scaling_coefficient_0.9/"



python -u generate_completions.py \
    --model_name $model_name \
    --behaviors_path $behaviors_path \
    --test_cases_path $test_cases_path \
    --save_path $save_path \
    --chat_template $chat_template \
    --max_new_tokens $max_new_tokens --dataset $dataset --model_name_or_path $model_name_or_path \
    --trust_remote_code --generate_with_vllm
```

Evaluation:
```bash

eval_type="sorrybench"
cls_path="/path/to/ft-mistral-7b-instruct-v0.2-sorry-bench-202406"
behaviors_path="./data/behavior_datasets/harmbench_behaviors_text_all.csv"
completions_path= "/path/to/generated/reponses"


python -u evaluate_completions.py \
    --cls_path $cls_path \
    --behaviors_path $behaviors_path \
    --completions_path $completions_path \
    --save_path $save_path \
    --eval_type $eval_type \
    --include_advbench_metric
```


### Test on Math


1. GSM8K
```bash
python inference_llms.py --dataset_name gsm8k --finetuned_model_path /path/to/your/merged/model --tensor_parallel_size $num_gpus
python test_gsm8k_rule.py --files /path/to/generated/responses
```

2. MATH
```bash
python inference_llms.py --dataset_name MATH --finetuned_model_path /path/to/your/merged/model --tensor_parallel_size $num_gpus
python test_math_rule.py --files /path/to/generated/responses
```

### Test on Code

1. mbpp
```bash
python inference_llms.py --dataset_name mbpp --finetuned_model_path /path/to/your/merged/model --tensor_parallel_size $num_gpus
accelerate launch --num_processes $num_gpus ./bigcode-evaluation-harness/main.py --tasks mbpp --allow_code_execution --load_generations_path /path/to/generated/responses
```

2. humanevalpack
```bash
python inference_llms.py --dataset_name human_eval_pack --finetuned_model_path /path/to/your/merged/model --tensor_parallel_size $num_gpus
accelerate launch --num_processes $num_gpus ./bigcode-evaluation-harness/main.py --tasks humanevalfixdocs-python --allow_code_execution --load_generations_path /path/to/generated/responses
```




## ✒️ Citation
If you find **LED-Merging** useful for your research and applications, please kindly cite our paper using this BibTeX:
```bibtex
@article{ma2025led,
  title={LED-Merging: Mitigating Safety-Utility Conflicts in Model Merging with Location-Election-Disjoint},
  author={Ma, Qianli and Liu, Dongrui and Chen, Qian and Zhang, Linfeng and Shao, Jing},
  journal={arXiv preprint arXiv:2502.16770},
  year={2025}
}
```

## ❤️ Acknowledgement

Our code is built upon [MergeLLM](https://github.com/yule-BUAA/MergeLLM), [MergeLM](https://github.com/yule-BUAA/MergeLM) and [alignment-attribution-code](https://github.com/boyiwei/alignment-attribution-code).
The Benchmark we use is [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness) and [HarmBench](https://github.com/centerforaisafety/HarmBench.git).

We also refer to the Pretrained backbones and SFT LLMs used above. Thanks to all the contributors for their great work!