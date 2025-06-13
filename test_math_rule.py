import json


import re

def test(files):
    
    for file in files:
        cnt = 0
        sum = 0
        with open(file,"r",encoding="utf-8") as fr:
            data = json.load(fr)
        for line in data:
            # s = line["Judge"]
            s = line["output"]
            last = s.split("\n\n")[-1]
            ans = line["answer"]
            line["last"] = last
            flag = False
            try:
                float(ans)
                # cnt += 1
                numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", last)
                sum += 1
                # if ans in last:
                #     cnt += 1
                # if numbers[-1]
                if numbers:
                    line["sample"] = numbers[-1]
                    if numbers[-1] == ans:
                        cnt += 1
                        flag = True
            except:
                # pass
                if ans in last:
                    cnt += 1
                    flag = True
            line["rule_judge"] = flag
        # with open(file,"w") as fw:
        #     json.dump(data, fw, ensure_ascii=False,indent=4)
        print(f"file:{file} acc:{cnt / len(data) }")


import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--files",nargs="+",action="store",type=str)
# parser.add_argument("--path",default=None,type=str)
# parser.add_argument("--num_gpus",default=1,type=int)
# parser.add_argument("--stop_tokens",nargs='+',action="store",type=str)
# parser.add_argument("--batch_size",type=int,default=32)
# parser.add_argument("--stop_token_ids",nargs='+',type=int,action="store")
args = parser.parse_args()

files = args.files
test(files)