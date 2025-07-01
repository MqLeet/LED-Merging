import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForVision2Seq, AutoProcessor

from vllm import LLM, SamplingParams
import json
import argparse
import os
import csv
from tqdm import tqdm 
import numpy as np
import pdb
from vllm import LLM, SamplingParams
import torch
from eval_utils import compute_results_hashing, compute_results_classifier, compute_results_advbench, compute_results_classifier_multimodal
# Set this to disable warning messages in the generation mode.
transformers.utils.logging.set_verbosity_error()

def parse_args():
    parser = argparse.ArgumentParser(description="Running red teaming with baseline methods.")
    parser.add_argument("--cls_path", type=str, default='cais/HarmBench-Llama-2-13b-cls',
                        help="The name or path of the classifier for computing ASR")
    parser.add_argument("--behaviors_path", type=str, default='./data/behavior_datasets/harmbench_behaviors_text_all.csv',
                        help="The path to the behaviors file")
    parser.add_argument("--completions_path", type=str,
                        help="The path to the completions to evaluate")
    parser.add_argument("--save_path", type=str,
                        help="The path for saving results")
    parser.add_argument("--include_advbench_metric", action="store_true",
                        help="Whether to include the AdvBench refusal metric")
    parser.add_argument("--num_tokens", type=int, default=512,
                        help="The number of tokens to evaluate")
    parser.add_argument("--eval_type", type=str, default="llama-guard3")
    args = parser.parse_args()
    return args


def main():
    # ========== load arguments and config ========== #
    args = parse_args()
    print(args)
    
    # ========== load behaviors (for tags and context strings) ========== #
    with open(args.behaviors_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        behaviors = list(reader)
    
    # convert to dictionary mapping behavior ID field to behavior dict
    behaviors = {b['BehaviorID']: b for b in behaviors}

    # ========== initialize results ========== #
    with open(args.completions_path, 'r') as f:
        completions = json.load(f)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.cls_path)
    tokenizer.truncation_side="right"
    test_cases = []
    # Clip the 'generation' field of the completions to have a maximum of num_tokens tokens
    for behavior_id, completion_list in completions.items():
        for completion in completion_list:
            generation = completion['generation']
            if isinstance(generation, list):
                generation = generation[0]
            tokenized_text = tokenizer.encode(generation, max_length=args.num_tokens, truncation=True)
            clipped_generation = tokenizer.decode(tokenized_text, skip_special_tokens=True)
            completion['generation'] = clipped_generation

            test_cases.append(completion["test_case"])

    # ========== evaluate completions ========== #
    num_gpus_available = torch.cuda.device_count()
    if args.eval_type == "llama-guard3-vision":
        cls = AutoModelForVision2Seq.from_pretrained(
                args.cls_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(args.cls_path)

        # cls = LLM(model=args.cls_path, tensor_parallel_size=num_gpus_available, enforce_eager=True,max_num_seqs=16)
    else:
        cls = LLM(model=args.cls_path, tensor_parallel_size=num_gpus_available)
        cls.llm_engine.tokenizer.tokenizer.truncation_side="left"

        cls_params = SamplingParams(temperature=0.0, max_tokens=10)

    results = {}
    cnt = 0
    for behavior_id, data in tqdm(completions.items()):
        # if behavior_id not in behaviors:
        behavior_dict = {"Behavior" : data[0]["test_case"],"generation" : data[0]["generation"]}
        

        if args.eval_type == "llama-guard3-vision":

            current_results = compute_results_classifier_multimodal(behavior_dict, data, cls, cls_params,eval_type=args.eval_type,processor=processor)
        else:
            current_results = compute_results_classifier(behavior_dict, data, cls, cls_params,eval_type=args.eval_type)       
        # behavior_dict = behaviors[behavior_id]
        # tags = behavior_dict['Tags'].split(', ')
    
        # if 'hash_check' in tags:
        #     current_results = compute_results_hashing(behavior_dict, data)
            
        # # else:

        
        # if args.include_advbench_metric:
        #     current_results = compute_results_advbench(current_results, data)
        cnt += current_results[-1]
        current_results.pop()
        results[behavior_id] = current_results

    success_rates = []

    # for i, (behavior_id, data) in enumerate(results.items()):
    #     try:
    #         success_mean = np.mean([d['label'] == 1 for d in data])
    #     except:
    #         pdb.set_trace()
    #     print(f"{i}. {behavior_id} === average asr: {success_mean}")
    #     success_rates.append(success_mean)

    # Calculate and add Average ASR to results
    average_asr = cnt / len(completions)
    results["Average_ASR"] = average_asr
    
    print(f"dataset: {args.completions_path.split("/")[-2]}, Model:{args.completions_path.split("/")[-1][:-5]} Average ASR: ", average_asr)

    # Make dirs to output_file if not yet exist
    # os.makedirs(os.path.dirname(args.save_path), exist_ok=True) 
    with open(args.save_path, 'w') as file:
        json.dump(results, file, indent=4)
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
