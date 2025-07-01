import json, jsonlines
import re, os, sys,ast
def transfer(file):
    r = {}
    idx = 0
    with open(file, "r") as fr:
        for line in fr:
    
            d = json.loads(line)
            
            v = d["prompt"]
            '''
            {
                "":[
                ""
                ]
            }
            '''
            r.update({ idx : [v]})
            idx += 1
    
    match = re.match(r"(.*/)([^/]+)$", file)
    prefix = match.group(1)
    suffix = "test_cases.json"
    new_file = prefix + "/" + suffix
    with open(new_file, "w") as fw:
        json.dump(r, fw, indent=4)

def transfer_behavior(file):
    r = {}
    idx = 0
    with open(file, "r") as fr:
        for line in fr:
            d = json.loads(line)
            if d["is_safe"]:
                continue
            cate = None
            for cate_, true_or_false in d["category"].items():
                if true_or_false is True:
                    cate = cate_
                    break
            
            
            r.update({cate + str(idx): [d["prompt"]]})
            idx += 1
    match = re.match(r"(.*/)([^/]+)$", file)
    prefix = match.group(1)
    suffix = "test_cases.json"
    new_file = prefix + "/" + suffix
    with open(new_file, "w") as fw:
        json.dump(r, fw, indent=4)

if __name__ == "__main__":
    args = sys.argv
    
    transfer_behavior(args[1])
    
