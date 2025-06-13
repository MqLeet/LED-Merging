import re
import sys
import jsonlines
import shutil
import logging
import os
from tqdm import tqdm
import glob
import json
import torch
import datasets
from fraction import Fraction
from human_eval.data import write_jsonl, read_problems, stream_jsonl
from vllm import SamplingParams
from transformers import AutoTokenizer
from utils.load_config import cache_dir
from utils.utils import TextVQAAccuracyEvaluator, evaluate_exact_match_accuracy, check_answer,process_example
from utils.mathutils import evaluate
from PIL import Image
from qwen_vl_utils import process_vision_info
from functools import partial
import pandas as pd
import pdb
import io
from typing import List
from vllm.model_executor.models.internvl import image_to_pixel_values
def test_IFEval(llm, finetuned_model_name, logger: logging.Logger, start_index=0, end_index=sys.maxsize,
                     save_model_path=None, save_gen_results_folder=None):
    # try:
    #     eval_set = datasets.load_dataset(path=os.path.join("IFEval/data", "alpaca_eval"), name="alpaca_eval")["eval"]
    # except:
    #     eval_set = datasets.load_dataset(path="tatsu-lab/alpaca_eval", name="alpaca_eval",trust_remote_code=True)["eval"]
    eval_set = []
    with open("instruction_following_eval/data/input_data.jsonl","r") as fr:
        for line in fr.readlines():
            d = json.loads(line.strip())
            eval_set.append({"instruction" : d["prompt"], "key" : d["instruction_id_list"], "instruction_id_list" : d["instruction_id_list"]})

    instructions = []
    reference_outputs = []
    for example in eval_set:
        # dictionary with 'instruction', 'output': 'generator' and 'dataset' as keys
        instructions.append(example["instruction"])
        reference_outputs.append(example)

    instructions = instructions[start_index:end_index]
    reference_outputs = reference_outputs[start_index:end_index]

    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=2048,stop="<|eot_id|>")
    logger.info(f"sampling params is {sampling_params}")

    shutil.rmtree(save_gen_results_folder, ignore_errors=True)
    os.makedirs(save_gen_results_folder, exist_ok=True)
    generator_name = save_model_path if save_model_path is not None else finetuned_model_name
    logger.info(f"generator name is {generator_name}")

    # for idx, (prompt, reference_output) in enumerate(zip(instructions, reference_outputs)):
    #     output_file = f"{save_gen_results_folder}/{start_index + idx}.jsonl"

    #     generated_outputs = []
    #     prompt = [generate_instruction_following_task_prompt(instruction=prompt, is_chat_model=True)]
    #     completions = llm.generate(prompt, sampling_params)
    #     for output in completions:
    #         generated_text = output.outputs[0].text
    #         generated_outputs.append({
    #             "prompt":reference_output["instruction"],
    #             "response":generated_text
    #         })

    #     write_jsonl(output_file, generated_outputs)
    batch_size = 32

    for batch_start in range(0, len(instructions), batch_size):
        batch_instructions = instructions[batch_start:batch_start + batch_size]
        batch_reference_outputs = reference_outputs[batch_start:batch_start + batch_size]

        prompts = [generate_instruction_following_task_prompt(instruction=prompt, is_chat_model=True)
                for prompt in batch_instructions]

        completions = llm.generate(prompts, sampling_params)

        for idx, (completion, reference_output) in enumerate(zip(completions, batch_reference_outputs)):
            output_file = f"{save_gen_results_folder}/{start_index + batch_start + idx}.jsonl"

            generated_outputs = []
            for output in completion.outputs:
                generated_text = output.text
                generated_outputs.append({
                    "prompt": reference_output["instruction"],
                    "response": generated_text
                })

            write_jsonl(output_file, generated_outputs)


    files = sorted(glob.glob(f"{save_gen_results_folder}/*.jsonl"))
    logger.info(f"find {len(files)} files in {save_gen_results_folder}")

    outputs = []
    for instruction_file in tqdm(files, total=len(files)):
        codes = [c for c in stream_jsonl(instruction_file)]
        outputs += codes

    logger.info(f"save to {save_gen_results_folder}.json")
    with open(f"{save_gen_results_folder}.json", "w", encoding="utf-8") as fout:
        json.dump(outputs, fout)
    if save_model_path is not None:
        shutil.rmtree(save_model_path, ignore_errors=True)

    del llm
    torch.cuda.empty_cache()

def test_alpaca_eval(llm, generator_model_name, logger: logging.Logger, start_index=0, end_index=sys.maxsize,
                     save_gen_results_folder=None):
    try:
        eval_set = datasets.load_dataset(path=os.path.join(cache_dir, "alpaca_eval"), name="alpaca_eval")["eval"]
    except:
        eval_set = datasets.load_dataset(path="tatsu-lab/alpaca_eval", name="alpaca_eval", cache_dir=cache_dir)["eval"]

    instructions = []
    reference_outputs = []
    for example in eval_set:
        # dictionary with 'instruction', 'output': 'generator' and 'dataset' as keys
        instructions.append(example["instruction"])
        reference_outputs.append(example)

    instructions = instructions[start_index:end_index]
    reference_outputs = reference_outputs[start_index:end_index]

    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=2048)
    logger.info(f"Sampling params is {sampling_params}.")

    shutil.rmtree(save_gen_results_folder, ignore_errors=True)
    os.makedirs(save_gen_results_folder, exist_ok=True)
    logger.info(f"Generator model name is {generator_model_name}.")

    for idx, (prompt, reference_output) in enumerate(zip(instructions, reference_outputs)):
        output_file = f"{save_gen_results_folder}/{start_index + idx}.jsonl"

        generated_outputs = []
        prompt = [generate_instruction_following_task_prompt(instruction=prompt, is_chat_model=True)]
        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            generated_text = output.outputs[0].text
            generated_outputs.append({
                "instruction": reference_output["instruction"],
                "output": generated_text,
                "generator": generator_model_name,
                "dataset": reference_output["dataset"]
            })

        write_jsonl(output_file, generated_outputs)

    files = sorted(glob.glob(f"{save_gen_results_folder}/*.jsonl"))
    logger.info(f"Find {len(files)} files in {save_gen_results_folder}.")

    outputs = []
    for instruction_file in tqdm(files, total=len(files)):
        codes = [c for c in stream_jsonl(instruction_file)]
        outputs += codes

    logger.info(f"Saving results to {save_gen_results_folder}.json...")
    with open(f"{save_gen_results_folder}.json", "w", encoding="utf-8") as fout:
        json.dump(outputs, fout)



def get_mmlu_prompt():
    return '''
    Below is an instruction that describes a task. 
    Answer the question using a single word or phrase.
{problem}
{options}
### Response: Let's think step by step.
'''
ans_mmlu = {0:"A",1:"B",2:"C",3:"D"}
def test_mmlu_math(llm, test_data_path, args, logger: logging.Logger, start_index=0, end_index=sys.maxsize,
               save_gen_results_folder=None, multimodal=False,processor=None):
    
    mmlu_ins = []
    mmlu_answers = []
    
    # problem_prompt = get_math_task_prompt()
    # with open("math_code_data/mmlu.parquet")
    data = pd.read_parquet(test_data_path)
    # with open(test_data_path, "r+", encoding="utf8") as f:
    # for idx, item in enumerate(jsonlines.Reader(f)):
    for row in data.iterrows():
        
        problem_prompt = get_mmlu_prompt()
        # temp_instr = problem_prompt.format(problem=row["question"],options=row["choices"])
        option = row[1]["choices"]
        option_str = "A)"+option[0] + "B)"+option[1] + "C)"+option[2] + "D)"+option[3]
        temp_instr = problem_prompt.format(problem=row[1]['question'],options=option_str)

        mmlu_ins.append(temp_instr)
        temp_ans = ans_mmlu[row[1]['answer']]
        
        mmlu_answers.append(temp_ans)

    
    batch_gsm8k_ins = batch_data(mmlu_ins, batch_size=64)

    # stop_tokens = ["Instruction:", "Instruction", "Response:", "Response","<|eot_id|>"]
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=1024, stop="<|eot_id|>")
    logger.info(f"Sampling params is {sampling_params}.")

    shutil.rmtree(save_gen_results_folder, ignore_errors=True)
    os.makedirs(save_gen_results_folder, exist_ok=True)

    res_completions = []
    for idx, prompt in enumerate(batch_gsm8k_ins):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]
        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            generated_text = output.outputs[0].text
            res_completions.append(generated_text)

    results = []
    invalid_outputs = []
    for idx, (prompt, completion, prompt_answer) in enumerate(zip(mmlu_ins, res_completions, mmlu_answers)):
        output_file = f"{save_gen_results_folder}/{start_index + idx}.jsonl"
        # y_pred = extract_answer_number(completion)
        # if y_pred != None:
        #     results.append(float(y_pred) == float(prompt_answer))
        # else:
        #     results.append(False)
        #     temp = {'question': prompt, 'output': completion, 'answer': prompt_answer}
        #     invalid_outputs.append(temp)
        # if len(completion) == 1:
        if completion == prompt_answer:
            results.append(True)
        else:
            results.append(False)
            temp = {'question': prompt, 'output': completion, 'answer': prompt_answer}
            invalid_outputs.append(temp)
        generated_outputs = [{
            "question": prompt,
            "output": completion,
            "answer": prompt_answer,
            "passed": results[-1]
        }]
        write_jsonl(output_file, generated_outputs)

    accuracy = sum(results) / len(results)
    logger.info(f"Invalid outputs length is {len(invalid_outputs)}, invalid_outputs are {invalid_outputs}.")
    logger.info(f"Data index starts from {start_index}, ends at {end_index}.")
    logger.info(f"Test data length of gsm8k is {len(results)}, accuracy is {accuracy}.")
    logger.info(args)

    files = sorted(glob.glob(f"{save_gen_results_folder}/*.jsonl"))
    logger.info(f"Find {len(files)} files in {save_gen_results_folder}.")

    outputs = []
    for instruction_file in tqdm(files, total=len(files)):
        codes = [c for c in stream_jsonl(instruction_file)]
        outputs += codes

    logger.info(f"Saving results to {save_gen_results_folder}.json...")
    with open(f"{save_gen_results_folder}.json", "w", encoding="utf-8") as fout:
        json.dump(outputs, fout)

def test_gsm8k(llm, test_data_path, args, logger: logging.Logger, start_index=0, end_index=sys.maxsize,
               save_gen_results_folder=None, multimodal=False,processor=None):
    if multimodal:
        instruct = "You are a helpful assistant."
    gsm8k_ins = []
    gsm8k_answers = []
    if not multimodal:
        # if "Meta-Llama-3-8B" in args.finetuned_model_path:
        #     problem_prompt = get_math_task_prompt(base=True)
        # else:
        problem_prompt = get_math_task_prompt()
        logger.info(f"Test prompt of gsm8k is {problem_prompt}")
    with open(test_data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            if multimodal:
                message = [{
                "role": "assistant",
                "content": [
                    {"type": "text", "text": instruct},
                    ],
            },
                    {
                        "role": "user",
                        "content": [
                            
                            {"type": "text", "text": item["question"]},
                            ],
                    }]
                prompt = processor.apply_chat_template(
                message,  add_generation_prompt=True,
            )
                gsm8k_ins.append(prompt)
                temp_ans = item['answer'].split('#### ')[1]
                temp_ans = int(temp_ans.replace(',', ''))
                gsm8k_answers.append(temp_ans)

            else:

                temp_instr = problem_prompt.format(instruction=item["question"])
                gsm8k_ins.append(temp_instr)
                temp_ans = item['answer'].split('#### ')[1]
                temp_ans = int(temp_ans.replace(',', ''))
                gsm8k_answers.append(temp_ans)

    gsm8k_ins = gsm8k_ins[start_index:end_index]
    gsm8k_answers = gsm8k_answers[start_index:end_index]
    batch_gsm8k_ins = batch_data(gsm8k_ins, batch_size=32)

    stop_tokens = ["Instruction:", "Instruction", "Response:", "Response","<|eot_id|>"]
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=1024, stop=["</s>",  "\n###", "\n### Human:","<|eot_id|>"])
    logger.info(f"Sampling params is {sampling_params}.")

    shutil.rmtree(save_gen_results_folder, ignore_errors=True)
    os.makedirs(save_gen_results_folder, exist_ok=True)

    res_completions = []
    for idx, prompt in enumerate(batch_gsm8k_ins):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]
        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            generated_text = output.outputs[0].text
            res_completions.append(generated_text)

    results = []
    invalid_outputs = []
    for idx, (prompt, completion, prompt_answer) in enumerate(zip(gsm8k_ins, res_completions, gsm8k_answers)):
        output_file = f"{save_gen_results_folder}/{start_index + idx}.jsonl"
        y_pred = extract_answer_number(completion)
        if y_pred != None:
            results.append(float(y_pred) == float(prompt_answer))
        else:
            results.append(False)
            temp = {'question': prompt, 'output': completion, 'answer': prompt_answer}
            invalid_outputs.append(temp)
        generated_outputs = [{
            "question": prompt,
            "output": completion,
            "answer": prompt_answer,
            "passed": results[-1]
        }]
        write_jsonl(output_file, generated_outputs)

    accuracy = sum(results) / len(results)
    logger.info(f"Invalid outputs length is {len(invalid_outputs)}, invalid_outputs are {invalid_outputs}.")
    logger.info(f"Data index starts from {start_index}, ends at {end_index}.")
    logger.info(f"Test data length of gsm8k is {len(results)}, accuracy is {accuracy}.")
    logger.info(args)

    files = sorted(glob.glob(f"{save_gen_results_folder}/*.jsonl"))
    logger.info(f"Find {len(files)} files in {save_gen_results_folder}.")

    outputs = []
    for instruction_file in tqdm(files, total=len(files)):
        codes = [c for c in stream_jsonl(instruction_file)]
        outputs += codes

    logger.info(f"Saving results to {save_gen_results_folder}.json...")
    with open(f"{save_gen_results_folder}.json", "w", encoding="utf-8") as fout:
        json.dump(outputs, fout)


def test_hendrycks_math(llm, test_data_path, args, logger: logging.Logger, start_index=0, end_index=sys.maxsize,
                        save_gen_results_folder=None,multimodal=False,processor=None):
    hendrycks_math_ins = []
    hendrycks_math_answers = []
    instruct = "You are a helpful assistant.Please answer the following math question and give the answer ending with 'The answer is: '"
    if not multimodal:
        problem_prompt = get_math_task_prompt()
        logger.info(f"Test prompt of MATH is {problem_prompt}")
    with open(test_data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            if not multimodal:
                temp_instr = problem_prompt.format(instruction=item["instruction"])
                hendrycks_math_ins.append(temp_instr)
                solution = item['output']
                temp_ans = remove_boxed(last_boxed_only_string(solution))
                hendrycks_math_answers.append(temp_ans)
            else:
                message = [{
                "role": "assistant",
                "content": [
                    {"type": "text", "text": instruct},
                    ],
            },
                    {
                        "role": "user",
                        "content": [
                            
                            {"type": "text", "text": item["instruction"]},
                            ],
                    }]
                prompt = processor.apply_chat_template(
                message,  add_generation_prompt=True,
            )
                hendrycks_math_ins.append(prompt)
                solution = item['output']
                temp_ans = remove_boxed(last_boxed_only_string(solution))
                hendrycks_math_answers.append(temp_ans)

    hendrycks_math_ins = hendrycks_math_ins[start_index:end_index]
    hendrycks_math_answers = hendrycks_math_answers[start_index:end_index]
    batch_hendrycks_math_ins = batch_data(hendrycks_math_ins, batch_size=32)

    stop_tokens = ["Instruction:", "Instruction", "Response:", "Response"]
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=2048, stop=["</s>",  "\n###", "\n### Human:","<|eot_id|>"])
    logger.info(f"Sampling params is {sampling_params}.")

    shutil.rmtree(save_gen_results_folder, ignore_errors=True)
    os.makedirs(save_gen_results_folder, exist_ok=True)

    res_completions = []
    for idx, prompt in enumerate(batch_hendrycks_math_ins):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]
        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            generated_text = output.outputs[0].text
            res_completions.append(generated_text)

    results = []
    invalid_outputs = []
    for idx, (prompt, completion, prompt_answer) in enumerate(zip(hendrycks_math_ins, res_completions, hendrycks_math_answers)):
        output_file = f"{save_gen_results_folder}/{start_index + idx}.jsonl"
        res = process_results(prompt, completion, prompt_answer, invalid_outputs)
        results.append(res)
        generated_outputs = [{
            "question": prompt,
            "output": completion,
            "answer": prompt_answer,
            "passed": results[-1]
        }]
        write_jsonl(output_file, generated_outputs)

    accuracy = sum(results) / len(results)
    logger.info(f"Invalid outputs length is {len(invalid_outputs)}, invalid_outputs are {invalid_outputs}.")
    logger.info(f"Data index starts from {start_index}, ends at {end_index}.")
    logger.info(f"Test data length of MATH is {len(results)}, accuracy is {accuracy}.")
    logger.info(args)

    files = sorted(glob.glob(f"{save_gen_results_folder}/*.jsonl"))
    logger.info(f"Find {len(files)} files in {save_gen_results_folder}.")

    outputs = []
    for instruction_file in tqdm(files, total=len(files)):
        codes = [c for c in stream_jsonl(instruction_file)]
        outputs += codes

    logger.info(f"Saving results to {save_gen_results_folder}.json...")
    with open(f"{save_gen_results_folder}.json", "w", encoding="utf-8") as fout:
        json.dump(outputs, fout)
def test_human_eval_pack(llm, args, logger: logging.Logger, start_index=0, end_index=sys.maxsize, save_gen_results_folder=None):
    problems = []
    with open("math_code_data/humanevalpack.jsonl","r",encoding="utf-8") as fr:
        for line in fr.readlines():
            problems.append(json.loads(line))
    task_ids = [p["task_id"] for p in problems][start_index: end_index]
#     prompts = [f'''
# You are given a programming problem along with its function signature. Your task is to generate a Python function that solves the problem. The function will be evaluated on a set of test cases, and the solution will be judged based on whether it passes these test cases.
# ---
# Problem: {p["text"]}

# Function signature: {p["prompt"]}

# Test Cases:
# {p["test"]}
# ''' for p in problems]
    
    prompts = [f'''
Problem: {p["text"]}

Function signature: {p["prompt"]}

Test Cases:
{p["test"]}
''' for p in problems]
    
    num_samples = len(prompts)
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=512,stop="<|eot_id|>")

    shutil.rmtree(save_gen_results_folder, ignore_errors=True)
    os.makedirs(save_gen_results_folder, exist_ok=True)
    batch_size = 16
    for batch_start in tqdm(range(0, num_samples, batch_size), ncols=0, total=(num_samples + batch_size - 1) // batch_size):
        # 获取当前批次的 prompts 和 task_ids
        batch_prompts = prompts[batch_start:batch_start + batch_size]
        batch_task_ids = task_ids[batch_start:batch_start + batch_size]

        # 构建批次的 prompts 和 ids
        prompt_batch = [generate_code_task_prompt(prompt.replace('    ', '\t')) for prompt in batch_prompts]
        # prompt_batch = [generate_code_task_prompt(prompt.replace('    ', '\t') for prompt in batch_prompts)]
        if batch_start == 0:
            print("prompt enhanced sample:{}".format(prompt_batch[0]))
        ids_batch = batch_task_ids

        completion_seqs = []

        loops = 1
        for _ in tqdm(range(loops), total=loops, leave=False, ncols=0):
            with torch.no_grad():
                # 批量生成 completions
                completions = llm.generate(prompt_batch, sampling_params)

            # 处理每个批次的输出
            for idx, completion in enumerate(completions):
                gen_seqs = [output.text for output in completion.outputs]

                task_id = ids_batch[idx]  # 获取当前的 task_id

                for seq_idx, gen_seq in enumerate(gen_seqs):
                    completion_seq = gen_seq.split("### Response:")[-1]
                    completion_seq = completion_seq.replace('\t', '    ')
                    all_code = gen_seq.replace('\t', '    ')

                    completion_seqs.append(
                        {
                            'task_id': task_id,
                            'completion': completion_seq,
                            'all_code': all_code,
                        }
                    )
        for idx, seqs in enumerate(completion_seqs):
            output_file = f"{save_gen_results_folder}/{args.start_index + batch_start + idx}.jsonl"
            write_jsonl(output_file, [seqs])
    files = sorted(glob.glob(f"{save_gen_results_folder}/*.jsonl"))
    logger.info(f"Find {len(files)} files in {save_gen_results_folder}.")

    outputs = []
    for code_file in tqdm(files, total=len(files)):
        codes = [c for c in stream_jsonl(code_file)]
        for code in codes:
            completion = code['completion']
            completion = completion.replace("\r", "")
            completion = completion.strip()
            if '```python' in completion:
                logger.info("Completion matches ```python")
                def_line = completion.index('```python')
                completion = completion[def_line:].strip()
                completion = completion.replace('```python', '')
                try:
                    next_line = completion.index('```')
                    completion = completion[:next_line].strip()
                except:
                    logger.info("Wrong completion.")
            if "__name__ == \"__main__\"" in completion:
                logger.info("Completion matches __name__ == \"__main__\"")
                try:
                    next_line = completion.index('if __name__ == "__main__":')
                    completion = completion[:next_line].strip()
                except:
                    logger.info("Wrong completion.")
            if "# Example usage" in completion:
                logger.info("Completion matches # Example usage")
                next_line = completion.index('# Example usage')
                completion = completion[:next_line].strip()
            # the following codes are used to deal with the outputs of code-alpaca
            if "The solution is:" in completion:
                logger.info("Completion matches The solution is:")
                def_line = completion.index("The solution is:")
                completion = completion[def_line:].strip()
                completion = completion.replace('The solution is:', '')
                try:
                    next_line = completion.index('\n\nThe answer is:')
                    completion = completion[:next_line].strip()
                except:
                    completion = completion.strip()
                    logger.info("Maybe wrong completion.")
            if "The answer is:" in completion:
                logger.info("Completion matches The answer is:")
                def_line = completion.index("The answer is:")
                completion = completion[def_line:].strip()
                completion = completion.replace('The answer is:', '')
                try:
                    next_line = completion.index('\n\nThe answer is:')
                    completion = completion[:next_line].strip()
                except:
                    completion = completion.strip()
                    logger.info("Maybe wrong completion.")
            code['completion'] = completion
        outputs += codes
    outputs = sorted(outputs, key=lambda x: int(x["task_id"].split("/")[1]))
    logger.info(f"Saving results to {save_gen_results_folder}.jsonl...")
    write_jsonl(f"{save_gen_results_folder}.jsonl", outputs)
def test_human_eval(llm, args, logger: logging.Logger, start_index=0, end_index=sys.maxsize, save_gen_results_folder=None):
    # problems = read_problems()
    # problems = []
    # with open("math_code_data/humanevalpack.jsonl","r",encoding="utf-8") as fr:
    #     for line in fr.readlines():
    #         problems.append(json.loads(line))
    # for p in problems:
    #     p["task_id"] = p["task_id"].replace("Python","HumanEval")

    # task_ids = sorted([p["task_id"] for p in problems])[start_index: end_index]
    # pdb.set_trace()
    # prompts = [f"\n{p['text']}\nCode prompt:{p["prompt"]}" for p in problems]
    # prompts = [f"\n{p["prompt"]}" for p in problems]
    problems = read_problems()
    task_ids = sorted(problems.keys())[start_index: end_index]
    prompts = [problems[task_id]['prompt'] for task_id in task_ids]

    # for task_id in task_ids:
    #     for test_example in problems[task_id][]
    
    
    num_samples = len(prompts)
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=512,stop="<|eot_id|>")

    shutil.rmtree(save_gen_results_folder, ignore_errors=True)
    os.makedirs(save_gen_results_folder, exist_ok=True)
    batch_size = 16
    for batch_start in tqdm(range(0, num_samples, batch_size), ncols=0, total=(num_samples + batch_size - 1) // batch_size):
        # 获取当前批次的 prompts 和 task_ids
        batch_prompts = prompts[batch_start:batch_start + batch_size]
        batch_task_ids = task_ids[batch_start:batch_start + batch_size]

        # 构建批次的 prompts 和 ids
        prompt_batch = [generate_code_task_prompt_tokenizer(prompt.replace('    ', '\t'), args.tokenizer) for prompt in batch_prompts]
        # prompt_batch = [generate_code_task_prompt(prompt.replace('    ', '\t') for prompt in batch_prompts)]
        if batch_start == 0:
            print("prompt enhanced sample:{}".format(prompt_batch[0]))
        ids_batch = batch_task_ids

        completion_seqs = []

        loops = 1

        for _ in tqdm(range(loops), total=loops, leave=False, ncols=0):
            with torch.no_grad():
                # 批量生成 completions
                completions = llm.generate(prompt_batch, sampling_params)

            # 处理每个批次的输出
            for idx, completion in enumerate(completions):
                gen_seqs = [output.text for output in completion.outputs]

                task_id = ids_batch[idx]  # 获取当前的 task_id

                for seq_idx, gen_seq in enumerate(gen_seqs):
                    completion_seq = gen_seq.split("### Response:")[-1]
                    completion_seq = completion_seq.replace('\t', '    ')
                    all_code = gen_seq.replace('\t', '    ')

                    completion_seqs.append(
                        {
                            'task_id': task_id,
                            'completion': completion_seq,
                            'all_code': all_code,
                        }
                    )

        # 为每个 batch 生成的 completion_seqs 写入单个 JSONL 文件
        for idx, seqs in enumerate(completion_seqs):
            output_file = f"{save_gen_results_folder}/{args.start_index + batch_start + idx}.jsonl"
            write_jsonl(output_file, [seqs])

    # for i in tqdm(range(num_samples), ncols=0, total=num_samples):
    #     output_file = f"{save_gen_results_folder}/{args.start_index + i}.jsonl"

    #     prompt = prompts[i].replace('    ', '\t')
        
    #     prompt_batch = [generate_code_task_prompt(prompt)]

    #     ids_batch = [task_ids[i]]
    #     completion_seqs = []

    #     loops = 1

    #     for _ in tqdm(range(loops), total=loops, leave=False, ncols=0):

    #         with torch.no_grad():
    #             completions = llm.generate(prompt_batch, sampling_params)
    #         gen_seqs = [completions[0].outputs[0].text]

    #         if gen_seqs is not None:
    #             assert len(ids_batch) == 1
    #             task_id = ids_batch[0]

    #             for seq_idx, gen_seq in enumerate(gen_seqs):
    #                 completion_seq = gen_seq.split("### Response:")[-1]
    #                 completion_seq = completion_seq.replace('\t', '    ')
    #                 all_code = gen_seq.replace('\t', '    ')

    #                 completion_seqs.append(
    #                     {'task_id': task_id,
    #                      'completion': completion_seq,
    #                      'all_code': all_code,
    #                      }
    #                 )

    #     write_jsonl(output_file, completion_seqs)

    files = sorted(glob.glob(f"{save_gen_results_folder}/*.jsonl"))
    logger.info(f"Find {len(files)} files in {save_gen_results_folder}.")

    outputs = []
    for code_file in tqdm(files, total=len(files)):
        codes = [c for c in stream_jsonl(code_file)]
        for code in codes:
            completion = code['completion']
            completion = completion.replace("\r", "")
            completion = completion.strip()
            if '```python' in completion:
                logger.info("Completion matches ```python")
                def_line = completion.index('```python')
                completion = completion[def_line:].strip()
                completion = completion.replace('```python', '')
                try:
                    next_line = completion.index('```')
                    completion = completion[:next_line].strip()
                except:
                    logger.info("Wrong completion.")
            if "__name__ == \"__main__\"" in completion:
                logger.info("Completion matches __name__ == \"__main__\"")
                try:
                    next_line = completion.index('if __name__ == "__main__":')
                    completion = completion[:next_line].strip()
                except:
                    logger.info("Wrong completion.")
            if "# Example usage" in completion:
                logger.info("Completion matches # Example usage")
                next_line = completion.index('# Example usage')
                completion = completion[:next_line].strip()
            # the following codes are used to deal with the outputs of code-alpaca
            if "The solution is:" in completion:
                logger.info("Completion matches The solution is:")
                def_line = completion.index("The solution is:")
                completion = completion[def_line:].strip()
                completion = completion.replace('The solution is:', '')
                try:
                    next_line = completion.index('\n\nThe answer is:')
                    completion = completion[:next_line].strip()
                except:
                    completion = completion.strip()
                    logger.info("Maybe wrong completion.")
            if "The answer is:" in completion:
                logger.info("Completion matches The answer is:")
                def_line = completion.index("The answer is:")
                completion = completion[def_line:].strip()
                completion = completion.replace('The answer is:', '')
                try:
                    next_line = completion.index('\n\nThe answer is:')
                    completion = completion[:next_line].strip()
                except:
                    completion = completion.strip()
                    logger.info("Maybe wrong completion.")
            code['completion'] = completion
        outputs += codes

    logger.info(f"Saving results to {save_gen_results_folder}.jsonl...")
    write_jsonl(f"{save_gen_results_folder}.jsonl", outputs)

def test_apps(llm, test_data_path,args,save_gen_results_folder=None):
    with open("math_code_data/apps_eval.json","r",encoding="utf-8") as fr:
        samples = json.load(fr)
    batch_size = 16
    num_samples = len(samples)
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=2048,stop="<|eot_id|>")
    prompts = []
    for sample in samples:
        prompt = f"\n{sample["question"]}"
        prompts.append(prompt)
    task_ids = list(range(200))
    shutil.rmtree(save_gen_results_folder, ignore_errors=True)
    os.makedirs(save_gen_results_folder, exist_ok=True)
    for batch_start in tqdm(range(0, num_samples, batch_size), ncols=0, total=(num_samples + batch_size - 1) // batch_size):
        batch_prompts = prompts[batch_start:batch_start + batch_size]
        prompt_batch = [generate_code_task_prompt(prompt.replace('    ', '\t')) for prompt in batch_prompts]
        batch_task_ids = task_ids[batch_start:batch_start + batch_size]

        ids_batch = batch_task_ids

        completion_seqs = []

        loops = 1

        for _ in tqdm(range(loops), total=loops, leave=False, ncols=0):
            with torch.no_grad():
                # 生成 completions，批量处理
                completions = llm.generate(prompt_batch, sampling_params)

            # 处理每个批次的输出
            for idx, completion in enumerate(completions):
                gen_seqs = [output.text for output in completion.outputs]

                task_id = ids_batch[idx]  # 获取当前的 task_id

                for seq_idx, gen_seq in enumerate(gen_seqs):
                    completion_seq = gen_seq.split("### Response:")[-1]
                    completion_seq = completion_seq.replace('\t', '    ')
                    all_code = gen_seq.replace('\t', '    ')

                    completion_seqs.append(
                        {
                            'task_id': task_id,
                            'completion': completion_seq,
                            'all_code': all_code,
                        }
                    )
        output_file = f"{save_gen_results_folder}/{args.start_index + batch_start}.jsonl"
        write_jsonl(output_file, completion_seqs)
    files = sorted(glob.glob(f"{save_gen_results_folder}/*.jsonl"))
    # logger.info(f"Find {len(files)} files in {save_gen_results_folder}.")

    # problems = read_mbpp(test_data_path)
    outputs = [[] for _ in range(200)]
    for code_file in tqdm(files, total=len(files)):
        codes = [c for c in stream_jsonl(code_file)]
        for code in codes:
            task_id = code['task_id']
            completion = code['completion']
            completion = completion.strip()
            if '```python' in completion:
                # logger.info("Completion matches ```python")
                def_line = completion.index('```python')
                completion = completion[def_line:].strip()
                completion = completion.replace('```python', '')
                try:
                    next_line = completion.index('\n```')
                    completion = completion[:next_line].strip()
                except:
                    print("Wrong completion.")
            if "__name__ == \"__main__\"" in completion:
                # logger.info("Completion matches __name__ == \"__main__\"")
                try:
                    next_line = completion.index('if __name__ == "__main__":')
                    completion = completion[:next_line].strip()
                except:
                    print("Wrong completion.")
            if "# Example usage" in completion:
                # logger.info("Completion matches # Example usage")
                next_line = completion.index('# Example usage')
                completion = completion[:next_line].strip()
            if "# Test examples" in completion:
                # logger.info("Completion matches # Test examples")
                next_line = completion.index('# Test examples')
                completion = completion[:next_line].strip()
            # the following codes are used to deal with the outputs of code-alpaca
            if "The solution is:" in completion:
                # logger.info("Completion matches The solution is:")
                def_line = completion.index("The solution is:")
                completion = completion[def_line:].strip()
                completion = completion.replace('The solution is:', '')
                try:
                    next_line = completion.index('\n\nThe answer is:')
                    completion = completion[:next_line].strip()
                except:
                    completion = completion.strip()
                    # logger.info("Maybe wrong completion.")
            if "The answer is:" in completion:
                # logger.info("Completion matches The answer is:")
                def_line = completion.index("The answer is:")
                completion = completion[def_line:].strip()
                completion = completion.replace('The answer is:', '')
                try:
                    next_line = completion.index('\n\nThe answer is:')
                    completion = completion[:next_line].strip()
                except:
                    completion = completion.strip()
                    # logger.info("Maybe wrong completion.")print("Wrong completion.")
            outputs[task_id - 11].append(completion)
    
    print(f"Saving results to {save_gen_results_folder}.json...")
    with open(f"{save_gen_results_folder}.jsonl", "w", encoding="utf-8") as fout:
        json.dump(outputs, fout)
def test_mbpp(llm, test_data_path, args, logger: logging.Logger, start_index=0, end_index=sys.maxsize, save_gen_results_folder=None):
    problems = read_mbpp(test_data_path)
    task_ids = sorted(problems.keys())[start_index: end_index]
    prompts = []
    for task_id in task_ids:
        prompt = f"\n{problems[task_id]['text']}\nTest examples:"
        if task_id == 493:
            # The test examples are too long, we choose to only include the function name.
            test_example = problems[task_id]['test_list'][0]
            prompt += f"\ncalculate_polygons(startx, starty, endx, endy, radius)"
        else:
            for test_example in problems[task_id]['test_list']:
                prompt += f"\n{test_example}"
        prompts.append(prompt)

    num_samples = len(prompts)
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=2048,stop=["</s>",  "\n###", "\n### Human:","<|eot_id|>"])

    shutil.rmtree(save_gen_results_folder, ignore_errors=True)
    os.makedirs(save_gen_results_folder, exist_ok=True)

    
    batch_size = 16

    
    for batch_start in tqdm(range(0, num_samples, batch_size), ncols=0, total=(num_samples + batch_size - 1) // batch_size):
        
        batch_prompts = prompts[batch_start:batch_start + batch_size]
        batch_task_ids = task_ids[batch_start:batch_start + batch_size]

        # 替换所有 prompt 中的空格
        prompt_batch = [generate_code_task_prompt(prompt.replace('    ', '\t')) for prompt in batch_prompts]
        ids_batch = batch_task_ids

        completion_seqs = []

        loops = 1

        for _ in tqdm(range(loops), total=loops, leave=False, ncols=0):
            with torch.no_grad():
                # 生成 completions，批量处理
                completions = llm.generate(prompt_batch, sampling_params)

            # 处理每个批次的输出
            for idx, completion in enumerate(completions):
                gen_seqs = [output.text for output in completion.outputs]

                task_id = ids_batch[idx]  # 获取当前的 task_id

                for seq_idx, gen_seq in enumerate(gen_seqs):
                    completion_seq = gen_seq.split("### Response:")[-1]
                    completion_seq = completion_seq.replace('\t', '    ')
                    all_code = gen_seq.replace('\t', '    ')

                    completion_seqs.append(
                        {
                            'task_id': task_id,
                            'completion': completion_seq,
                            'all_code': all_code,
                        }
                    )

        # 为每个 batch 生成的 completion_seqs 写入单个 JSONL 文件
        output_file = f"{save_gen_results_folder}/{args.start_index + batch_start}.jsonl"
        write_jsonl(output_file, completion_seqs)
    # for i in tqdm(range(num_samples), ncols=0, total=num_samples):
    #     output_file = f"{save_gen_results_folder}/{args.start_index + i}.jsonl"

    #     prompt = prompts[i].replace('    ', '\t')
    #     prompt_batch = [generate_code_task_prompt(prompt)]

    #     ids_batch = [task_ids[i]]
    #     completion_seqs = []

    #     loops = 1

    #     for _ in tqdm(range(loops), total=loops, leave=False, ncols=0):

    #         with torch.no_grad():
    #             completions = llm.generate(prompt_batch, sampling_params)
    #         gen_seqs = [completions[0].outputs[0].text]

    #         if gen_seqs is not None:
    #             assert len(ids_batch) == 1
    #             task_id = ids_batch[0]

    #             for seq_idx, gen_seq in enumerate(gen_seqs):
    #                 completion_seq = gen_seq.split("### Response:")[-1]
    #                 completion_seq = completion_seq.replace('\t', '    ')
    #                 all_code = gen_seq.replace('\t', '    ')

    #                 completion_seqs.append(
    #                     {'task_id': task_id,
    #                      'completion': completion_seq,
    #                      'all_code': all_code,
    #                      }
    #                 )

    #     write_jsonl(output_file, completion_seqs)

    files = sorted(glob.glob(f"{save_gen_results_folder}/*.jsonl"))
    logger.info(f"Find {len(files)} files in {save_gen_results_folder}.")

    problems = read_mbpp(test_data_path)
    outputs = [[] for _ in range(len(problems))]
    
    for code_file in tqdm(files, total=len(files)):
        codes = [c for c in stream_jsonl(code_file)]
        for code in codes:
            task_id = code['task_id']
            completion = code['completion']
            completion = completion.strip()
            if '```python' in completion:
                logger.info("Completion matches ```python")
                def_line = completion.index('```python')
                completion = completion[def_line:].strip()
                completion = completion.replace('```python', '')
                try:
                    next_line = completion.index('\n```')
                    completion = completion[:next_line].strip()
                except:
                    logger.info("Wrong completion.")
            if "__name__ == \"__main__\"" in completion:
                logger.info("Completion matches __name__ == \"__main__\"")
                try:
                    next_line = completion.index('if __name__ == "__main__":')
                    completion = completion[:next_line].strip()
                except:
                    logger.info("Wrong completion.")
            if "# Example usage" in completion:
                logger.info("Completion matches # Example usage")
                next_line = completion.index('# Example usage')
                completion = completion[:next_line].strip()
            if "# Test examples" in completion:
                logger.info("Completion matches # Test examples")
                next_line = completion.index('# Test examples')
                completion = completion[:next_line].strip()
            # the following codes are used to deal with the outputs of code-alpaca
            if "The solution is:" in completion:
                logger.info("Completion matches The solution is:")
                def_line = completion.index("The solution is:")
                completion = completion[def_line:].strip()
                completion = completion.replace('The solution is:', '')
                try:
                    next_line = completion.index('\n\nThe answer is:')
                    completion = completion[:next_line].strip()
                except:
                    completion = completion.strip()
                    logger.info("Maybe wrong completion.")
            if "The answer is:" in completion:
                logger.info("Completion matches The answer is:")
                def_line = completion.index("The answer is:")
                completion = completion[def_line:].strip()
                completion = completion.replace('The answer is:', '')
                try:
                    next_line = completion.index('\n\nThe answer is:')
                    completion = completion[:next_line].strip()
                except:
                    completion = completion.strip()
                    logger.info("Maybe wrong completion.")
            outputs[task_id - 11].append(completion)

    logger.info(f"Saving results to {save_gen_results_folder}.json...")
    with open(f"{save_gen_results_folder}.jsonl", "w", encoding="utf-8") as fout:
        json.dump(outputs, fout)

def _vllm_generate_internvl(model,prompts,images, **generation_kwargs):
    # inputs = [input.update({"pixel_values": })]
    # for input in inputs:
    pixel_values = [image_to_pixel_values(image,input_size=448,min_num=1,max_num=12,use_thumbnail=False) for image in images]
    inputs = [{
            "pixel_values": pixel_value,
            "prompt": prompt,
            "multi_modal_data": {
                "image": image
            },
        } for prompt,image,pixel_value in zip(prompts, images, pixel_values)]
    sp = SamplingParams(temperature=0, max_tokens=generation_kwargs["max_new_tokens"])
    new_generation_kwargs = dict(sampling_params=sp, use_tqdm=True)
    outputs = model.generate(inputs, **new_generation_kwargs)
    # pdb.set_trace()
    generations = [o.outputs[0].text.strip() if not isinstance(o, str) else o for o in outputs]
    # pdb.set_trace()
    return generations
def _vllm_generate_modal(model, prompts, images, **generation_kwargs):
    # inputs = [template['prompt'].format(instruction=s) for s in test_cases]
    inputs = [{

            "prompt": prompt,
            "multi_modal_data": {
                "image": image
            },
        } for prompt,image in zip(prompts, images)]
    sp = SamplingParams(temperature=0, max_tokens=generation_kwargs["max_new_tokens"])
    new_generation_kwargs = dict(sampling_params=sp, use_tqdm=True)
    outputs = model.generate(inputs, **new_generation_kwargs)
    # pdb.set_trace()
    generations = [o.outputs[0].text.strip() if not isinstance(o, str) else o for o in outputs]
    # pdb.set_trace()
    return generations


def load_llava(questions: List[str]=None, image_urls: List[str]=None,options=None,image_bytes=None,dataset=None):
    examples = [{"question":question,"options":option} for question, option in zip(questions, options)]
    prompts = [f"USER: <image>\n{process_example(example)}\nASSISTANT:" for example in examples]
    prefix = prefix_map[dataset]
    if image_urls is None:
        images = [Image.open(io.BytesIO(image)).convert("RGB") for image in image_bytes]
    else:
        images = [Image.open(prefix + "/"+ image).convert("RGB") for image in image_urls]
    return prompts, images
def load_internvl(questions: List[str]=None, image_urls: List[str]=None,dataset:str=None,options=None,image_bytes=None,processor=None,tokenizer_path=None):
    prompts = []
    prefix = prefix_map[dataset]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    for question, option in zip(questions, options):
        example = {"question":question, "options":option}
        if (isinstance(option, list) and len(option) > 0) or (isinstance(option, str) and len(option) > 2):
            instruct = "You are a helpful assistant. Answer with the option's letter from the given choices directly."
        else:
            instruct = "You are a helpful assistant. Answer the question using a single word or phrase."
       
        message = [{'role':'assistant', 'content': instruct},{'role': 'user', 'content': f"<image>\n{process_example(example)}"}]
        # message = [{'role': 'user', 'content': f"<image>\n{process_example(example)}"}]


        prompt = tokenizer.apply_chat_template(
                message,  add_generation_prompt=True, tokenize=False
            )
        prompts.append(prompt)  
    if image_bytes is None:
        images = [Image.open(prefix + "/"+ image).convert("RGB") for image in image_urls]
    else:
        images = [Image.open(io.BytesIO(image)).convert("RGB") for image in image_bytes]
    return prompts, images  

def load_qwen2_vl(questions: List[str]=None, image_urls: List[str]=None, processor=None, image_bytes=None, options=None,dataset=None):
    prompts = []
    images = []
    if image_bytes is not None:
        for question, image_byte, option in zip(questions, image_bytes, options):
            example = {"question":question, "options":option}
            
            message = [
                        {'role': 'system', 'content': 'You are a helpful assistant.'},
                        {'role': 'user', 'content': [
                            {
                                'type': 'image',
                                'image': io.BytesIO(image_byte),
                            },

                            {
                                'type': 'text',
                                'text': process_example(example),
                            },
                        ]}
                    ]
            prompt = processor.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True,
                )
            prompts.append(prompt)  
            image_input, _ = process_vision_info(message)
            images.append(image_input)
    else:
        prefix = prefix_map[dataset]
        for question, image, option in zip(questions, image_urls, options):
            example = {"question":question, "options":option}
            
            image_path = prefix + "/"+ image
            message = [
                        {'role': 'system', 'content': 'You are a helpful assistant.'},
                        {'role': 'user', 'content': [
                            {
                                'type': 'image',
                                'image': image_path,
                            },

                            {
                                'type': 'text',
                                'text': process_example(example),
                            },
                        ]}
                    ]
            prompt = processor.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True,
                )
            prompts.append(prompt)  
            image_input, _ = process_vision_info(message)
            images.append(image_input)
    return prompts, images
def load_llava_next(questions: str=None, image_urls: List[str]=None, image_bytes = None, options: List[str]=None, dataset: str=None,processor=None):
    prompts = []
    
    prefix = prefix_map[dataset]
    for question, option in zip(questions, options):
        example = {"question":question, "options":option}
        if (isinstance(option, list) and len(option) > 0) or (isinstance(option, str) and len(option) > 2):
            instruct = "You are a helpful assistant. Answer with the option's letter from the given choices directly."
        else:
            instruct = "You are a helpful assistant. Answer the question using a single word or phrase."
       
        message = [{
                "role": "assistant",
                "content": [
                    {"type": "text", "text": instruct},
                    ],
            },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": process_example(example)},
                            ],
                    }]

        prompt = processor.apply_chat_template(
                message,  add_generation_prompt=True,
            )
        prompts.append(prompt)  
    if image_bytes is None:
        images = [Image.open(prefix + "/"+ image).convert("RGB") for image in image_urls]
    else:
        images = [Image.open(io.BytesIO(image)).convert("RGB") for image in image_bytes]
    return prompts, images
prefix_map = {
    "mathvqa":"multimodal_data/mathvqa",
    "mmmu_math":"multimodal_data/MMMU/math"
}
model_example_map = {
    # "phi3_v": load_phi3v,
    "internvl": load_internvl,
    "qwen2_vl": load_qwen2_vl,
    # "qwen_vl_chat": load_qwenvl_chat,
    "llava-next": load_llava_next,
    "llava-1.5":load_llava
}
def test_mmmu_math(llm=None, processor=None, args=None):
    data = pd.read_parquet("multimodal_data/MMMU/math/validation-00000-of-00001.parquet")
    questions = data["question"].tolist()
    images_raw = data['image_1'].tolist()
    image_bytes = [d["bytes"] for d in images_raw]
    answers = data["answer"].tolist()
    options = data["options"].tolist()
    # prefix = "multimodal_data/MMMU/math"
    if "Qwen2" in args.finetuned_model_path:
        model_name = "Qwen2"
        prompts, images = load_qwen2_vl(questions=questions, image_bytes=image_bytes,processor=processor,options=options,dataset="mmmu_math")  
    elif "llava-1.5" in args.finetuned_model_path:
        model_name = "llava1.5"
        prompts, images = load_llava(questions=questions, image_bytes=image_bytes,processor=processor,options=options,dataset="mmmu_math")
    elif "llava-next" in args.finetuned_model_path or "llava1.6" in args.finetuned_model_path:
        model_name = "llava1.6"
        prompts, images = load_llava_next(questions=questions, image_bytes=image_bytes,processor=processor,options=options,dataset="mmmu_math")
    elif "InternVL" in args.finetuned_model_path or "internvl" in args.finetuned_model_path:
        prompts, images = load_internvl(questions=questions, image_bytes=image_bytes,processor=processor,options=options,dataset="mmmu_math",tokenizer_path=args.finetuned_model_path)
       
       
    generation_kwargs = dict(max_new_tokens=256, do_sample=False, num_beams=1)
    # if "InternVL" in args.finetuned_model_path:
    #     generation_function = partial(_vllm_generate_internvl, model=llm, **generation_kwargs)
    # else:
    generation_function = partial(_vllm_generate_modal, model=llm, **generation_kwargs)
    generations = generation_function(images = images, prompts = prompts)
    # step 3
    df = {row["id"]: row.to_dict() for index, row in data.iterrows()} # 1-3040
    lines = [{"id" : "validation_Math_"+str(i+1), "response" : generation} for i, generation in enumerate(generations)]
    # lines = [line.update({"model_answer":model_answer}) for line, model_answer in zip(lines, answers)]
    for line,ans in zip(lines, answers):
        line["answer"] = ans
    if "save_merge_llms" in args.finetuned_model_path:
        if "mask_merging" in args.finetuned_model_path:
            model_name = args.finetuned_model_path.split("/")[6] + "DARE:" + args.finetuned_model_path.split("/")[-1]
        elif "task_arithmetic" in args.finetuned_model_path:
            model_name = args.finetuned_model_path.split("/")[6] + args.finetuned_model_path.split("/")[-1]
        elif "ties_merging" in args.finetuned_model_path:
            model_name = args.finetuned_model_path.split("/")[6] + args.finetuned_model_path.split("/")[-1]
    else:
        model_name = args.finetuned_model_path.split("/")[-1]
    evaluate(df, lines, f"results/mmmu_math/{model_name}.jsonl",regen_answer=False)
import time
def timestamp() -> str:
    nowtime = time.strftime('-%Y%m%d-%H%M', time.localtime(time.time()))
    print(nowtime)  
    return nowtime  
def save_jsonl(path: str, data: list, t_stamp=True) -> None:
    if t_stamp:
        file_name = f"{path.replace('.jsonl','')}{timestamp()}.jsonl"
    else:
        file_name = path
    with open(file_name, 'w', encoding='utf-8') as f:
        for line in tqdm(data, desc='save'):
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
def test_mathvqa(llm=None, processor=None, args=None):
    # step1
    data = pd.read_parquet("multimodal_data/mathvqa/test-00000-of-00001-3532b8d3f1b4047a.parquet")
    questions = data["question"].tolist()
    images_raw = data['image'].tolist()
    answers = data["answer"].tolist()
    options = data["options"].tolist()
    
    
    if "Qwen2" in args.finetuned_model_path:
        model_name = "Qwen2"
        prompts, images = load_qwen2_vl(questions=questions, image_urls=images_raw,processor=processor,options=options,dataset="mathvqa")  
    elif "llava-1.5" in args.finetuned_model_path:
        model_name = "llava1.5"
        prompts, images = load_llava(questions=questions, image_urls=images_raw,processor=processor,options=options,dataset="mathvqa")
    elif "llava-next" in args.finetuned_model_path or "llava1.6" in args.finetuned_model_path:
        model_name = "llava1.6"
        prompts, images = load_llava_next(questions=questions, image_urls=images_raw,processor=processor,options=options,dataset="mathvqa")
    elif "InternVL" in args.finetuned_model_path or "internvl" in args.finetuned_model_path:
        prompts, images = load_internvl(questions=questions, image_urls=images_raw,processor=processor,options=options,dataset="mathvqa",tokenizer_path=args.finetuned_model_path)
            
    
    
    # step2
    generation_kwargs = dict(max_new_tokens=256, do_sample=False, num_beams=1)

    # if "InternVL" in args.finetuned_model_path:
    #     generation_function = partial(_vllm_generate_internvl, model=llm, **generation_kwargs)
    # else:
    generation_function = partial(_vllm_generate_modal, model=llm, **generation_kwargs)
    generations = generation_function(images = images, prompts = prompts)
    # step 3
    df = {row["id"]: row.to_dict() for index, row in data.iterrows()} # 1-3040
    lines = [{"id" : str(i+1), "response" : generation} for i, generation in enumerate(generations)]
    
    # print(f"lines:{}")
    # pdb.set_trace()
    for line,ans in zip(lines, answers):
        line["answer"] = ans
    if "save_merge_llms" in args.finetuned_model_path:
        if "mask_merging" in args.finetuned_model_path:
            model_name = args.finetuned_model_path.split("/")[6] + "DARE:" + args.finetuned_model_path.split("/")[-1]
        elif "task_arithmetic" in args.finetuned_model_path:
            model_name = args.finetuned_model_path.split("/")[6] + args.finetuned_model_path.split("/")[-1]
    else:
        model_name = args.finetuned_model_path.split("/")[-1]
    save_jsonl(f"results/mathvqa/{model_name}.jsonl", lines, t_stamp=False)
    evaluate(df, lines, f"results/mathvqa/{model_name}.jsonl",regen_answer=False)


            
def test_mtvqa(llms=None, processor=None, args=None,languages=None):
    # step1
    # language AR,DE,FR,IT,JA,KR,RU,TH,VI
    if not isinstance(llms, list):
        llms = [llms]
    if not isinstance(languages, list):
        languages = [languages]
    for llm, language in zip(llms,languages):
        with open(f"multimodal_data/mtvqa/test/json/test_{language}.json","r",encoding="utf-8") as fr:
            data = json.load(fr)
        prefix = f"multimodal_data/mtvqa"
        if "Qwen2" in args.finetuned_model_path:
            prompts = []
            images = []
            answers = []
            for item in data:
                message = [
                        {'role': 'system', 'content': 'You are a helpful assistant.'},
                        {'role': 'user', 'content': [
                            {
                                'type': 'image',
                                'image': prefix + "/"+ item["img"],
                            },

                            {
                                'type': 'text',
                                'text': item["question"],
                            },
                        ]}
                    ]
                prompt = processor.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True,
                )
                prompts.append(prompt)  
                image_input, _ = process_vision_info(message)
                images.append(image_input)
                answers.append(item["answer"])
        else:
            prompts = [f"USER: <image>\n{item['question']}\nASSISTANT:" for item in data]
            images = [Image.open(prefix + "/"+ item["img"]).convert("RGB") for item in data]
            answers = [item["answer"] for item in data]
        generation_kwargs = dict(max_new_tokens=128, do_sample=False, num_beams=1)

        generation_function = partial(_vllm_generate_modal, model=llm, **generation_kwargs)
        generations = generation_function(images = images, prompts = prompts)
        # step 3
        entries = []
        for answer, pred in zip(answers, generations):
            entries.append({'annotation' : pred, 'answer' : answer})
        result = evaluate_exact_match_accuracy(entries)
        print(f"language:{language} Acc:{result}")

def test_textvqa(llm=None, processor=None, args=None):
    # step1 
    with open("multimodal_data/textvqa/TextVQA_0.5.1_val.json","r") as fr:
        data = json.load(fr)
    data = data["data"]
    prefix = "multimodal_data/textvqa/train_images"
    if "Qwen2" in args.finetuned_model_path:
        prompts = []
        images = []
        answers = []
        for item in data:
            image_path = prefix + "/"+item["image_id"]+".jpg"
            message = [
                        {'role': 'system', 'content': 'You are a helpful assistant.'},
                        {'role': 'user', 'content': [
                            {
                                'type': 'image',
                                'image': image_path,
                            },

                            {
                                'type': 'text',
                                'text': item["question"],
                            },
                        ]}
                    ]
            prompt = processor.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True,
                )
            prompts.append(prompt)  
            image_input, _ = process_vision_info(message)
            images.append(image_input)
            answers.append(item["answers"])
    elif "llava-1.5" in args.finetuned_model_path or "InternVL" in args.finetuned_model_path:
        prompts = [f"USER: <image>\n{item['question']}\nASSISTANT:" for item in data]
        images = [Image.open(prefix + "/"+item["image_id"]+".jpg").convert("RGB") for item in data]
        answers = [item["answers"] for item in data]
    elif "llava-next" in args.finetuned_model_path or "llava1.6" in args.finetuned_model_path:
        # prompts = [f"[INST] <image>\n{item['question']} [/INST]" for item in data]
        # images = [Image.open(prefix + "/"+item["image_id"]+".jpg").convert("RGB") for item in data]
        # answers = [item["answers"] for item in data]
        model_name = "llava1.6"
        prompts = []
        images = []
        answers = [item["answers"] for item in data]
        for item in data:
            # example = {"question":question}
            # if (isinstance(option, list) and len(option) > 0) or (isinstance(option, str) and len(option) > 2):
            #     instruct = "You are a helpful assistant. Answer with the option's letter from the given choices directly."
            # else:
            #     instruct = "You are a helpful assistant. Answer the question using a single number."
            instruct = "You are a helpful assistant. Answer the question using a single word or phrase."
            message = [{
        "role": "assistant",
        "content": [
            {"type": "text", "text": instruct},
            ],
    },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": item["question"]},
                    ],
            }]

            prompt = processor.apply_chat_template(
                    message,  add_generation_prompt=True,
                )
            prompts.append(prompt)
            images.append(Image.open(prefix + "/"+item["image_id"]+".jpg").convert("RGB"))

    # step2 
    generation_kwargs = dict(max_new_tokens=128, do_sample=False, num_beams=1)
    
    generation_function = partial(_vllm_generate_modal, model=llm, **generation_kwargs)
    generations = generation_function(images = images, prompts = prompts)
    # step3 eval
    cnt = sum([1 if check_answer(pred, answer) else 0 for pred, answer in zip(generations, answers)])
    results = [{"pred" : pred, "answer":answer } for pred, answer in zip(generations, answers)]
    if "save_merge_llms" in args.finetuned_model_path:
        if "mask_merging" in args.finetuned_model_path:
            model_name = args.finetuned_model_path.split("/")[6] + "DARE:" + args.finetuned_model_path.split("/")[-1]
        elif "task_arithmetic" in args.finetuned_model_path:
            model_name = args.finetuned_model_path.split("/")[6] + args.finetuned_model_path.split("/")[-1]
    else:
        model_name = args.finetuned_model_path.split("/")[-1]
    with open(f"results/textvqa/{model_name}.json","w") as fw:
        json.dump(results, fw, indent=4)
    
    # pred_list = []
    # for result,answer in zip(generations,answers):
    #     # annotation = annotations[(result['question_id'], prompt_processor(result['prompt']))]
    #     pred_list.append({
    #         "pred_answer": result,
    #         "gt_answers": answer,
    #     })
    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(len(generations), 100. * cnt / 5000))

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def extract_answer_number(completion):
    text = completion.split('The answer is: ')
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) == True and is_number(numerator) == True:
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        else:
            return None
    else:
        return None


def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n - 1):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_data.append(data_list[start:end])

    last_start = (n - 1) * batch_size
    last_end = sys.maxsize
    batch_data.append(data_list[last_start:last_end])
    return batch_data


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


def process_results(doc, completion, answer, invalid_outputs):
    split_ans = completion.split('The answer is: ')
    if len(split_ans) > 1:
        ans = split_ans[-1]
        extract_ans_temp = ans.split('.\n')[0]
        extract_ans_temp = extract_ans_temp.strip()
        if len(extract_ans_temp) > 0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        if is_equiv(extract_ans, answer):
            return True
        else:
            return False
    else:
        temp = {'question': doc, 'output': completion, 'answer': answer}
        invalid_outputs.append(temp)
        return False


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string


def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        # pdb.set_trace()
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def generate_instruction_following_task_prompt(instruction, is_chat_model=True, tokenizer=None):
    if tokenizer is  None:
        if is_chat_model:
            prompt = f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {instruction} ASSISTANT:"""
        else:
            prompt = f"""{instruction}

### Response:
"""
        return prompt
    else:
        formatted_instruction = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."},
        {"role": "user", "content": instruction}
    ], tokenize=False
)
        return formatted_instruction


def get_math_task_prompt(tokenizer=None,base=False):
    if base:
        problem_prompt = "<|begin_of_text|>{instruction}"
        return problem_prompt
    if tokenizer is None:
        problem_prompt = (
            
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
        )
    else:
        problem_prompt = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "Below is an instruction that describes a task.Write a response that appropriately completes the request.\n\n"},
        {"role": "user", "content": "{instruction}"},
        {"role": "agent", "content": "Let's think step by step."}
    ], tokenize=False
)
    return problem_prompt


def generate_code_task_prompt_tokenizer(input_text, tokenizer):
    formatted_instruction = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "You are an expert AI assistant. Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction:Create a Python script for this problem:"},
        {"role": "user", "content": input_text}
    ], tokenize=False
)
    return formatted_instruction
def generate_code_task_prompt(input_text):
    INSTRUCTION = f"""
    ### System:
    Below is an instruction that describes a task. Write a response that appropriately completes the request.


### Instruction:
Create a Python script for this problem:
{input_text}

### Response:"""
    return INSTRUCTION
def generate_code_task_prompt_codellama3(input_text):
    INSTRUCTION = f"Below is an instruction that describes a task, Write a response that appropriately completes the request.{input_text}"
    return INSTRUCTION

def read_mbpp(path):
    mbpp_problems = {}
    with jsonlines.open(path, "r") as fin:
        for obj in fin:
            mbpp_problems[obj["task_id"]] = obj
    return mbpp_problems
