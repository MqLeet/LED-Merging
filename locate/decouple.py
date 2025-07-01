import pickle

import torch

import os

UTILS_PATH = {
    "inst" : "score/alpaca/llama3-instruct/wanda_score",
    "safety" : "score/safety/llama3-instruct/wanda_score",
    "math" : "score/math/llama3-math/wanda_score",
    "code" : "score/code/llama3-code/wanda_score"
}

UTILS_BASE_PATH = {
    "inst" : "score/alpaca/llama3-base/wanda_score",
    "safety" : "score/safety/llama3-base/wanda_score",
    "math" : "score/math/llama3-base/wanda_score",
    "code" : "score/code/llama3-base/wanda_score"
}
@torch.no_grad()
def sort_values(d_p):
    # d_p = d_p.cpu().numpy().tolist()
    
    # with torch.no_grad():
    #     d_p_pair = [(d_p[i][j].item(), i, j ) for i in range(len(d_p)) for j in range(len(d_p[0]))]
    # d_p_sorted = sorted(d_p_pair, key = lambda x: x[0], reverse=True)
    d_p_f = d_p.flatten()
    _, indices = torch.sort(d_p_f, descending=False)
    try:
        i, j = torch.div(indices, d_p.size(1), rounding_mode='floor'),  indices % d_p.size(1)
        return i, j
    except:
        # j = torch.div(indices, d_p)
        return indices

    
    # ans = [(i[idx].cpu().item(), j[idx].cpu().item()) for idx in range(len(i))]
    # return i, j 
def cmp(topk,d1s,d2s):
    l1 = int(topk * len(d1s))
    d1s = set([(x,y) for x,y in d1s[:l1]])
    l2 = int(topk * len(d2s))
    d2s = set([(x,y) for x,y in d2s[:l2]])

    u = set.intersection(d1s, d2s)
    return u

    # sum = [1 for d1, d2 in zip(d1s, d2s) if d1[1] == d2[1] and d1[2] == d2[2]]
    
import pdb
import numpy as np
def comparision(util1, util2, topk = 0.1, base=False):
    # if not base:
    #     p1 = UTILS_PATH[util1]
    #     p2 = UTILS_PATH[util2]
    # else:
    p1 = UTILS_BASE_PATH[util1]
    p2 = UTILS_BASE_PATH[util2]
        
    file_lists_1 = os.listdir(p1)
    avg = []
    for f in file_lists_1:
        f1 = os.path.join(p1, f)
        f2 = os.path.join(p2, f)
        with open(f1, "rb") as fr:
            w1 = pickle.load(fr).cuda()
        with open(f2, "rb") as fr:
            w2 = pickle.load(fr).cuda()
        if len(w1.size()) == 1:
            i1 = sort_values(w1)
            i2 = sort_values(w2)
            pool_size = int(i1.size(0) * topk)
            i1 = i1[:pool_size].cpu().numpy().tolist()
            i2 = i2[:pool_size].cpu().numpy().tolist()
        
            c1 = set(i1)
            c2 = set(i2)
            
        else:
            i1, j1 = sort_values(w1)
            i2, j2 = sort_values(w2)
            pool_size = int(i1.size(0) * topk)
            i1 = i1[:pool_size].cpu().numpy().tolist()
            j1 = j1[:pool_size].cpu().numpy().tolist()
            i2 = i2[:pool_size].cpu().numpy().tolist()
            j2 = j2[:pool_size].cpu().numpy().tolist()
            c1 = set(zip(i1, j1))
            c2 = set(zip(i2, j2))
        duplicate_rate = len(c1 & c2) / len(c1)
        print(f"layer:{f} {util1} {util2} topk:{topk} ratio:{duplicate_rate}")
        avg.append(duplicate_rate)
    print(f"AVG {util1} {util2} {np.mean(avg)}")
        # print(f"layer:{f} base 0 inst 1 topk:{topk} ratio:{len(c2 - c1) / len(c1)}")
  
            

if __name__ == "__main__":
    import sys
    # comparision(sys.argv[1])
    comparision(sys.argv[1], sys.argv[2], eval(sys.argv[3]))
    
        

