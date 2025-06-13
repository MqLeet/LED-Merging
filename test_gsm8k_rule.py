import json
import re
import pdb
import ast
from vllm import LLM
from transformers import AutoTokenizer
from vllm import SamplingParams
# from utils.evaluate_llms_utils import extract_answer_number
KEY_WORDS = ["the correct answer is", "the answer is","therefore","answer:"]


def rule_judge(completion, answer, top_from_bottom:int = 2, top_from_top: int = 0):
    if not isinstance(answer, str):
        answer = str(answer)
    if "### Explanation" in completion:
        completion = completion.split("### Explanation:")[0]
    paragraphs = completion.split("\n\n")
    # backward check
    cnt = 0
    # pdb.set_trace()
    for i in range(len(paragraphs) - 1, -1, -1):
        paragraph = paragraphs[i]
        cnt += 1
        if cnt > top_from_bottom:
            break
        sentences = paragraph.split(".")
        for j in range(len(sentences) - 1, -1, -1):
            sentence = sentences[j]
            # pdb.set_trace()
            numbers = re.findall(r'\d+\.?\d*', sentence)
            if numbers:
                try:
                    if eval(answer) == eval(numbers[-1]):
                        return True
                except:
                    pass
    # cnt = 0
    # for i in range(len(paragraphs) - 1):
    #     paragraph = paragraphs[i]
    #     cnt += 1
    #     if cnt > top_from_top:
    #         break
    #     sentences = paragraph.split("\n")
    #     for j in range(len(sentences) - 1, -1, -1):
    #         sentence = sentences[j]
    #         numbers = re.findall(r'\d+\.?\d*', sentence)
    #         cnt_t = 0
    #         for k in range(len(numbers) -1 , -1, -1):
    #             cnt_t += 1
    #             if cnt_t > 2:
    #                 break
    #             if answer == numbers[k]:
    #                 return True
            # if answer in numbers:
            # if answer == numbers[-1] or answer == numbers[-2]:
            #     return True
    return False

def double_check_gsm8k(file,stop=None):
    with open(file,"r") as fr:
        data = json.load(fr)
    number_zero = []
    number_one = []
    number_two = []
    
    cnt = 0
    for line in data:
        if 'Average_ACC' in line:
            continue
        completion = line["output"]
        answer = line["answer"]
        # if rule_judge(completion, answer):
        #     line["rule_base"] = True
        #     cnt += 1
        # else:
        #     line["rule_base"] = False
    
        sentences = completion.split("\n\n")[0]
        sentences = sentences.split(". ")
        for i in range(len(sentences)-1 , -1,-1):
            last = sentences[i]
            numbers = re.findall(r'\d+\.?\d*', last)
            if len(numbers) == 0:
                continue
            line["last"] = last
            
            if len(numbers) == 1:
                try:
                    if eval(numbers[0]) != line["answer"]:
                        line["passed"] = False
                    else:
                        line["passed"] = True
                        cnt += 1
                    number_one.append(line)
                except:
                    # pdb.set_trace()
                    line["passed"] = False
            else:
                try:
                    number = eval(numbers[-1])
                except:
                    continue
                if number != line["answer"]:
                    line["passed"] = False
                else:
                    line["passed"] = True 
                    cnt += 1
                number_two.append(line)
                # break
            break
    
    
    if "task" in file or "mask" in file:
        print(f"file:{file.split("/")[2][:-5]} cnt:{cnt} acc:{cnt / len(data)}")
    else:
        print(f"file:{file.split("/")[-1][:-5]} cnt:{cnt} acc:{cnt / len(data)}")
    with open(file, "w") as fw:
        json.dump(data, fw, indent=4)
    print(cnt / len(data))
    # numbers = [number_zero, number_one, number_two]
    # suffixes = [suffix_0, suffix_1, suffix_2]
    # for number, suffix in zip(numbers, suffixes):
    #     # pdb.set_trace()
    #     with open(prefix + "/" + name + "/"+ suffix, "w") as fr:
    #         json.dump(number, fr, indent=4, ensure_ascii=False)
import sys, os
def check_judger(files=None,LLM=None,tokenizer=None,sampling_params=None,batch_size:int=32):

    
       
    for file_prompt in files:

        with open(file_prompt, "r") as fr:
            data = json.load(fr)
        acc = 0
        samples = []
        for line in data:
            
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f'''
                For each question below, I will provide the model's generated response ("generation") and the correct answer ("answer"). Your task is to check if the model's generation is logically and mathematically correct compared to the answer. If the generation is correct, return True; otherwise, return False. Only output "True" or "False" based on whether the generation matches the correct answer.

                Example Format:

                Question: [{line["question"]}]
                Generation: [{line["output"]}]
                Answer: [{line["answer"]}]
                Return: True or False based on the comparison.
    '''},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False)
            samples.append({"line":line, "prompt":prompt})
        print(f"batch nums:{len(samples) // batch_size + 1}")
        for i in range(0, len(samples) // batch_size + 1, 1):
            samples_per_batch = samples[i * batch_size : (i + 1) * batch_size]
            prompts = [x["prompt"] for x in samples_per_batch]
            
            generations = LLM.generate(prompts,sampling_params)
            # pdb.set_trace()
            for sample, output in zip(samples_per_batch, generations):
                generated_text = output.outputs[0].text
                
                sample["line"]["Judge"] = generated_text
                if generated_text == "True":
                    acc += 1

        print(f"file:{file_prompt} acc:{acc / len(data)}")
    
        with open(file_prompt, "w") as fw:
            json.dump(data, fw, indent=4)


import argparse

if __name__ == "__main__":
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--files",nargs="+",action="store",type=str)
    # parser.add_argument("--path",default=None,type=str)
    # parser.add_argument("--num_gpus",default=1,type=int)
    # parser.add_argument("--stop_tokens",nargs='+',action="store",type=str)
    # parser.add_argument("--batch_size",type=int,default=32)
    # parser.add_argument("--stop_token_ids",nargs='+',type=int,action="store")
    args = parser.parse_args()

    
    files = args.files

    for file in files:
        cnt = 0
        with open(file, "r") as fr:
            data = json.load(fr)
        for sample in data:
            if 'Average_ACC' in sample:
                continue
            is_pass = rule_judge(sample["output"], sample["answer"], top_from_bottom = 2, top_from_top = 2)
            if is_pass:

                cnt += 1
        print(f"file:{file} acc:{cnt / len(data)}")
    
    