from tqdm import tqdm
import torch
import torch.nn as nn

from utils.utils import get_param_names_to_merge
from model_merging_methods.task_vector import TaskVector

from typing import List
import itertools
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

def normalize(x, dim=0):
    min_values, _ = torch.min(x, dim=dim, keepdim=True)
    max_values, _ = torch.max(x, dim=dim, keepdim=True)
    y = (x - min_values) / (max_values - min_values)
    return y

def clamp(x, min_ratio=0, max_ratio=0):
    if len(x.size())==1:
        d = x.size(0)
        sorted_x, _ = torch.sort(x)
        min=sorted_x[int(d * min_ratio)]
        max=sorted_x[int(d * (1-max_ratio)-1)]
    else:
        d = x.size(1)
        sorted_x, _ = torch.sort(x, dim=1)
        min=sorted_x[:, int(d * min_ratio)].unsqueeze(1)
        max=sorted_x[:, int(d * (1-max_ratio)-1)].unsqueeze(1)
    clamped_x= torch.clamp(x, min, max)
    return clamped_x

def act(x):
    y = torch.tanh(x)  # torch.relu(x)
    return y
# def mask_input_with_score(input_tensor: torch.Tensor, mask_rate: float, use_rescale: bool, mask_strategy: str):

def mask_input_with_mask_rate(input_tensor: torch.Tensor, mask_rate: float, use_rescale: bool, mask_strategy: str):
    """
    mask the input with mask rate
    :param input_tensor: Tensor, input tensor
    :param mask_rate: float, mask rate
    :param use_rescale: boolean, whether to rescale the input by 1 / (1 - mask_rate)
    :param mask_strategy: str, mask strategy, can be "random" and "magnitude"
    :return:
    """
    assert 0.0 <= mask_rate <= 1.0, f"wrong range of mask_rate {mask_rate}, should be [0.0, 1.0]!"
    if mask_strategy == "random":
        mask = torch.bernoulli(torch.full_like(input=input_tensor, fill_value=mask_rate)).to(input_tensor.device)
        masked_input_tensor = input_tensor * (1 - mask)
    else:
        assert mask_strategy == "magnitude", f"wrong setting for mask_strategy {mask_strategy}!"
        original_shape = input_tensor.shape
        input_tensor = input_tensor.flatten()
        num_mask_params = int(len(input_tensor) * mask_rate)
        # Tensor, shape (1, ), find the num_mask_params-th smallest magnitude element of all the parameters in the model
        kth_values, _ = input_tensor.abs().kthvalue(k=num_mask_params, dim=0, keepdim=True)
        # Tensor, shape (num_total_params, ), where True is for parameters that we want to perform mask
        mask = input_tensor.abs() <= kth_values
        masked_input_tensor = input_tensor * (~mask)
        masked_input_tensor = masked_input_tensor.reshape(original_shape)
    if use_rescale and mask_rate != 1.0:
        masked_input_tensor = torch.div(input=masked_input_tensor, other=1 - mask_rate)
    return masked_input_tensor
import pdb, os, pickle
from typing import Dict
def eq_param_name(name1: str, candidates: list[str]):
    '''
    name1: model.embed_tokens.weight, lm_head.weight, model.layers.28.mlp.down_proj.weight without norm layer
    candidates: embed_tokens.pkl, lm_head.pkl, W_metric_layer_0_name_model.layers.0.mlp.down_proj_weight.pkl,W_metric_layer_30_name_model.layers.30.self_attn.q_proj_weight                                                                         model.layers.31.self_attn.q_proj.weight
    '''
    name_sp = name1.split(".")
    # model.norm.weight
    if "norm" in name1 and "layernorm" not in name1:
        for candidate in candidates:
            if "W_metric_layer_0_name_model.norm_weight" in candidate:
                return candidate
            
    if len(name_sp) <= 3:
        name_x = name_sp[-2]
    else:
        
        if "layernorm" in name1:
            
            layer = name_sp[2]
            name_x = name_sp[3]
            name_x = ".".join([layer, name_x])
              
        else:
            layer = name_sp[2] 
            name_x = ".".join([name_sp[3], name_sp[4]])
            name_x = ".".join([layer, name_x])
        
            
    for candidate in candidates:
        group = candidate.split(".")
        if len(group) <= 2:
            if name_x in candidate:
                return candidate
        else:
        
            layerc = group[2]
            # name1: W_metric_layer_0_name_model.norm_weight.pkl
            if name_x in candidate and layerc == layer:
                return candidate
            # except:
                # pdb.set_trace()
                
   
    return None


def util_mask(score_base_safety: torch.Tensor, score_inst_safety: torch.Tensor, score_base_util: torch.Tensor, score_util: torch.Tensor, mask_rate: float):
    # safety mask
    score_base_safety_sorted, _ = torch.sort(score_base_safety.flatten(), descending=True)
    score_inst_safety, _ = torch.sort(score_inst_safety.flatten(), descending=True)

    
    
    n_elements = score_base_safety_sorted.numel()
    threhold1 = score_base_safety_sorted[int(mask_rate * n_elements) - 1]
    threhold2 = score_inst_safety[int(mask_rate * n_elements) - 1]
    mask1 = (score_base_safety >= threhold1).to(torch.int)
    mask2 = (score_inst_safety >= threhold2).to(torch.int)
    mask_safety = (1 - mask1) * mask2
    # util mask
     
def mask_input_with_two_score(input_tensor: torch.Tensor, mask_rate: float, score1: torch.Tensor,score2: torch.Tensor, descending: bool = True,score_pattern: str = "01",score3: torch.Tensor = None, util_mask: torch.Tensor = None):
    score1_sorted, indices1 = torch.sort(score1.flatten(), descending=True)
    score2_sorted, indices2 = torch.sort(score2.flatten(), descending=True)
    if score3 is not None:
        score3_sorted, indices3 = torch.sort(score3.flatten(), descending=True)
        score3_sorted = score3_sorted.to(input_tensor.device)
        n_elements3 = score3.numel()
        threhold3 = score3_sorted[int(mask_rate * n_elements3) - 1]
        mask3 = (score3 >= threhold3).to(torch.int)
    score1_sorted = score1_sorted.to(input_tensor.device)
    score2_sorted = score2_sorted.to(input_tensor.device)
    n_elements = score1.numel()
    threhold1 = score1_sorted[int(mask_rate * n_elements) - 1]
    threhold2 = score2_sorted[int(mask_rate * n_elements) - 1]
    mask1 = (score1 >= threhold1).to(torch.int)
    mask2 = (score2 >= threhold2).to(torch.int)
    if util_mask is not None:
        util_mask = util_mask.to(mask1.device)
    if score_pattern == "01":
        # if score3 is None:
        mask_both = (1 - mask1) * mask2
        if score3 is not None:
            # decouple
            mask_both = (1 - mask3) * mask_both
        if util_mask is not None:
            mask_both = (1 - util_mask) * mask_both
    elif score_pattern == "x1":
        mask_both = mask2
        if score3 is not None:
            mask_both = (1-mask3) * mask_both
        if util_mask is not None:
            mask_both = (1 - util_mask) * mask_both  
    elif score_pattern == "10":
        mask_both = mask1 * (1 - mask2)
        if util_mask is not None:
            mask_both = (1 - util_mask) * mask_both
    elif score_pattern == "11":
        mask_both = mask1 * mask2
        if score3 is not None:
            mask_both = (1-mask3)*mask_both
        if util_mask is not None:
            mask_both = (1 - util_mask) * mask_both

    mask_both = mask_both.to(input_tensor.device)
    input_tensor = input_tensor * mask_both
    return input_tensor
def mask_input_with_score(input_tensor: torch.Tensor, mask_rate: float, score: torch.Tensor,  descending: bool = True):
    '''
    input_tensor: task vector
    mask_rate: ratio
    score: base score
    '''
    score = score.cuda()
    score_sorted, _ = torch.sort(score.flatten(), descending=descending)
    # l = score.size(1)
    n_elements = score.numel()
    threhold = score_sorted[int(mask_rate * n_elements) - 1]
    # if descending:

    mask = score >= threhold
    # else:
        # mask = score <= threhold
    mask = mask.to(input_tensor.device)
    input_tensor = input_tensor * mask
    return input_tensor


DECOUPLE_DIR={
    "instruct": "./locate/score/alpaca/{model_ft_name}/wanda_score",
    "math": "./locate/score/math/{model_ft_name}/wanda_score",
    "code": "./locate/score/code/{model_ft_name}/wanda_score",
    "safety": "./locate/score/safety/{model_ft_name}/wanda_score",

    "base-instruct": "./locate/score/alpaca/{model_base_name}/wanda_score",
    "base-math": "./locate/score/math/{model_base_name}/wanda_score",
    "base-code": "./locate/score/code/{model_base_name}/wanda_score",
    "base-safety": "./locate/score/safety/{model_base_name}/wanda_score"
}
DECOUPLE_DIR={
    "instruct": "./locate/score/alpaca/{model_ft_name}/random_score",
    "math": "./locate/score/math/{model_ft_name}/random_score",
    "code": "./locate/score/code/{model_ft_name}/random_score",
    "safety": "./locate/score/safety/{model_ft_name}/random_score",

    "base-instruct": "./locate/score/alpaca/{model_base_name}/random_score",
    "base-math": "./locate/score/math/{model_base_name}/random_score",
    "base-code": "./locate/score/code/{model_base_name}/random_score",
    "base-safety": "./locate/score/safety/{model_base_name}/random_score"
}
'''
model: inst, math, code model_names:inst, math, code 
score_base: b
'''

def generate_mask(model, rate, model_ft_name:str="llama3-instruct",model_base_name:str="llama3-base"):
    param_names = os.listdir(DECOUPLE_DIR["base-safety"].format(model_base_name=model_base_name))
    all_mask = {}
    for name, param in tqdm(model.named_parameters()):

        param_name =  eq_param_name(name, param_names)

        with open(os.path.join(DECOUPLE_DIR["safety"].format(model_ft_name=model_ft_name), param_name), "rb") as fr:
            p_u = pickle.load(fr).cuda()
        with open(os.path.join(DECOUPLE_DIR["base-"+"safety"].format(model_base_name=model_base_name), param_name), "rb") as fr:
            p_b = pickle.load(fr).cuda()
        p_u_s, _ = torch.sort(p_u.flatten(), descending=True)
        p_b_s, _ = torch.sort(p_b.flatten(), descending=True)
        n_elements = p_u.numel()
        
        # rate_ = backs[another_name]
        th_u = p_u_s[int(rate * n_elements) - 1]
        th_b = p_b_s[int(rate * n_elements) - 1]
        mask_u = (p_u >= th_u).int()
        mask_b = (p_b >= th_b).int()
        mask_both = mask_b * mask_u
        all_mask[name] = mask_both
    with open(f"mask_dir_11_0.1/{model_ft_name}-{rate}.pkl","wb") as fw:
            pickle.dump(all_mask, fw)

@torch.no_grad()
def pairwise_mask_generate(model:nn.Module, mask_rate:float, model_names:List[str] = ["math","safety","code"],score_bases: List[str] = ["base-math","base-safety","base-code"],mask_pattern:str="01",mask_util:str=None,backs:Dict=None,model_ft_name:str="llama3-instruct",model_base_name:str="llama3-base"):
    model_type = model_base_name.split('-')[0]
    for model_name,score_base in zip(model_names, score_bases):
        if model_name not in [mask_util]:
            continue
        print(f"masking for {model_name}")
        save_dir = f"random_mask_dir_{mask_pattern}_{mask_rate}"
        save_path = f"{save_dir}/{model_name}.pkl"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # pdb.set_trace()
        param_names = os.listdir(DECOUPLE_DIR[score_base].format(model_base_name=model_base_name))
        another_names = [x for x in model_names if x != model_name]
       
        all_mask = {}
        # if mask_pattern in ["123safety"]:
        #     p_b_safety, p_u_safety = {}, {}
        if mask_pattern in ["123467"] and mask_util == "safety":
            for name, param in tqdm(model.named_parameters()):
                all_mask[name] = torch.zeros(param.shape)
        elif mask_pattern in ["123math"] and mask_util == "math":
            for name, param in tqdm(model.named_parameters()):
                all_mask[name] = torch.zeros(param.shape)         
        else:
            for name,param in tqdm(model.named_parameters()):
                param_name =  eq_param_name(name, param_names)
                p_anothers = []
                for another_name in another_names:
                    
                    if another_name not in backs:
                        continue
                    if another_name == "math":
                        model_ft_name_x = f"{model_type}-math"
                    elif another_name == "code":
                        model_ft_name_x = f"{model_type}-code"
                    elif another_name == "safety":
                        model_ft_name_x = f"{model_type}-instruct"
                   
                    with open(os.path.join(DECOUPLE_DIR[another_name].format(model_ft_name=model_ft_name_x), param_name), "rb") as fr:
                        p_u = pickle.load(fr).cuda()
                    with open(os.path.join(DECOUPLE_DIR["base-"+another_name].format(model_base_name=model_base_name), param_name), "rb") as fr:
                        p_b = pickle.load(fr).cuda()
                    # if mask_pattern in ["123safety"] and mask_util
                    pdb.set_trace()
                    if another_name=='math':
                        p_u = torch.randn_like(p_u, dtype=torch.bfloat16, device=p_u.device)
                    p_u_s, _ = torch.sort(p_u.flatten(), descending=True)
                    p_b_s, _ = torch.sort(p_b.flatten(), descending=True)
                    n_elements = p_u.numel()
                    
                    rate_ = backs[another_name]
                    th_u = p_u_s[int(rate_ * n_elements) - 1]
                    th_b = p_b_s[int(rate_ * n_elements) - 1]
                    mask_u = (p_u >= th_u).int()
                    mask_b = (p_b >= th_b).int()
                    if mask_pattern == "01": # u=1, b=0
                        try:
                            mask_both = (1 - mask_b) * mask_u
                        except:
                            if mask_b.size(0) > mask_u.size(0):
                                mask_both = (1 - mask_b[:-1,:]) * mask_u
                            else:
                                mask_both = (1 - mask_b) * mask_u[:-1,:]

                    elif mask_pattern == "10":
                        try:
                            mask_both = (1 - mask_u) * mask_b
                        except:
                            if mask_b.size(0) > mask_u.size(0):
                                mask_both = mask_b[:-1,:] * {1 - mask_u}
                            else:
                                mask_both = mask_b * (1 - mask_u[:-1,:])

                    elif mask_pattern == "00":
                        try:
                            mask_both = (1 - mask_u) * (1 - mask_b)
                        except:
                            if mask_b.size(0) > mask_u.size(0):
                                mask_both = (1 - mask_u[:-1,:]) * {1 - mask_u}
                            else:
                                mask_both = (1 - mask_b) * (1 - mask_u[:-1,:])

                    elif mask_pattern == "11" or mask_pattern  in ["123467", "23467","123math","123code","123safety","123safety0.2"]:
                        try:
                            mask_both = mask_b * mask_u
                        except:
                            # pdb.set_trace()
                            # mask_b[]
                            # mask_both = 
                            # mask_b = 
                            # vocab_size = mask_u.size(0)
                            # new_vocab = mask_u[vocab_size - 1, :].detach()
                            # mask_b = torch.cat([mask_b, new_vocab.unsqueeze(0)], dim = 0)
                            # mask_both = mask_b * mask_u
                            # mask_b[vocab_size - 1, :] = mask_u[vocab_size - 1, :]
                            # mask_both[:]
                            # mask_both = mask_
                            if mask_b.size(0) > mask_u.size(0):
                                mask_both = mask_b[:-1,:] * mask_u
                            else:
                                mask_both = mask_b * mask_u[:-1,:]
                    elif mask_pattern == "x1":
                        mask_both = mask_u
                    elif mask_pattern == "00":
                        mask_both = mask_b
                    if mask_pattern in ["123safety","123safety0.2"] and mask_util != "safety":
                        with open(os.path.join(DECOUPLE_DIR["safety"].format(model_ft_name=model_ft_name), param_name), "rb") as fr:
                            p_u_safety = pickle.load(fr).cuda()
                        with open(os.path.join(DECOUPLE_DIR["base-"+"safety"].format(model_base_name=model_base_name), param_name), "rb") as fr:
                            p_b_safety = pickle.load(fr).cuda()
                        p_u_safety_s, _ = torch.sort(p_u_safety.flatten(), descending=True)
                        p_b_safety_s, _ = torch.sort(p_b_safety.flatten(), descending=True)
                        n_elements = p_u_safety.numel()
                        rate_ = 0.2
                        th_u = p_u_safety_s[int(rate_ * n_elements) - 1]
                        th_b = p_b_safety_s[int(rate_ * n_elements) - 1]
                        mask_u_ex = (p_u_safety <= th_u).int()
                        mask_b_ex = (p_b_safety <= th_b).int()
                        mask_both_ex = mask_u_ex * mask_b_ex
                        p_anothers.append(mask_both_ex)
                        
                    p_anothers.append(mask_both)
                    del p_u
                    del p_b
                    del mask_u
                    del mask_b
                    if mask_pattern in ["123safety","123safety0.2"] and mask_util != "safety":
                        del p_u_safety
                        del p_b_safety
                        del mask_u_ex
                        del mask_b_ex
                    torch.cuda.empty_cache()
                mask_all = torch.any(torch.stack(p_anothers), dim = 0).int().cpu()
                if mask_pattern == "23467" and mask_util == "safety":
                    all_mask[name] = 1 - mask_all
                else:
                    all_mask[name] = mask_all
        
        with open(save_path,"wb") as fw:
            pickle.dump(all_mask, fw)

def threhold_mask_model_weights(finetuned_model: nn.Module, pretrained_model: nn.Module, fuse_area:Dict,mask_type:str="safety",mask_rate:str="0.1",pattern:str="01"):
    # score_lists = os.listdir(score_dir)
    mask_dir = f"mask_dir_{pattern}_{mask_rate}"
    if mask_type == "instruct":
        mask_type = "inst"
    with open(f"{mask_dir}/{mask_type}.pkl","rb") as fr:
        mask = pickle.load(fr)

    with torch.no_grad():
        sft_param = finetuned_model.state_dict()
        masked_param_dict = {}

        for param_name, param_value in tqdm(pretrained_model.named_parameters()):
            # score_f = eq_param_name(param_name, score_lists)
            masked_param_dict[param_name] = param_value

            masked_param_dict[param_name] = masked_param_dict[param_name].cuda()
            mask[param_name] = mask[param_name].cuda()
            sft_param[param_name] = sft_param[param_name].cuda()
            fuse_area[param_name] = fuse_area[param_name].cuda()
            true_mask =  mask[param_name]
            masked_param_dict[param_name] = torch.where(
                    true_mask.bool(), 
                    sft_param[param_name], 
                    masked_param_dict[param_name]
                )
            fuse_area[param_name] = torch.logical_or(fuse_area[param_name], true_mask).int()
            mask[param_name] = mask[param_name].cpu()
            sft_param[param_name] = sft_param[param_name].cpu()
            fuse_area[param_name] = fuse_area[param_name].cpu()
            masked_param_dict[param_name] = masked_param_dict[param_name].cpu()

    return masked_param_dict

def neuron_merge_weight(finetuned_model: nn.Module, pretrained_model: nn.Module, fuse_rate:float, util:str,score_dir:str,pattern:str="00",score_extra:str=None):
    # if protect == util:
    #     return finetuned_model.state_dict()
    mask_dir = f"mask_dir_{pattern}_{str(fuse_rate)}"
    with open(mask_dir + "/" + util + ".pkl", "rb") as fr:
        all_mask = pickle.load(fr)
    score_lists = os.listdir(score_dir)

    with torch.no_grad():
        masked_param_dict = {}
        sft_param = finetuned_model.state_dict()

        for param_name, param_value in tqdm(pretrained_model.named_parameters()):
            score_f = eq_param_name(param_name, score_lists)
            masked_param_dict[param_name] = param_value
            masked_param_dict[param_name] = masked_param_dict[param_name].cuda()
            sft_param[param_name] = sft_param[param_name].cuda()

            param_mask = all_mask[param_name].cuda()
            with open(os.path.join(score_dir, score_f), "rb") as fr:
                score_weight = pickle.load(fr).cuda()
            
            n = score_weight.numel()

            score_sorted, _ = torch.sort(score_weight.flatten(),descending=True)
            
            threshold = score_sorted[int(fuse_rate * n) - 1]

            mask = (score_weight > threshold).int().cuda()
            if score_extra is not None:
                # embed_tokens.pkl
                with open(os.path.join(score_extra, score_f), "rb") as fr:
                    score_weight_extra = pickle.load(fr).cuda()
                score_sorted_extra, _ = torch.sort(score_weight_extra.flatten(),descending=True)
                threhold_extra = score_sorted_extra[int(fuse_rate * n) - 1]
                mask_extra = (score_weight_extra > threhold_extra).int().cuda()
                if pattern == "01":
                    if not mask_extra.size(0) == mask.size(0):
                        mask_extra_append = torch.cat((mask_extra, mask_extra[-1,:].unsqueeze(0)), dim = 0)
                        mask_extra = mask_extra_append
                    mask_11 = (1 - mask_extra) * mask
                elif pattern == "10":
                    if not mask_extra.size(0) == mask.size(0):
                        mask_extra_append = torch.cat((mask_extra, mask_extra[-1,:].unsqueeze(0)), dim = 0)
                        mask_extra = mask_extra_append
                    mask_11 = (1 - mask) * mask_extra
                elif pattern == "00":
                    if not mask_extra.size(0) == mask.size(0):
                        mask_extra_append = torch.cat((mask_extra, mask_extra[-1,:].unsqueeze(0)), dim = 0)
                        mask_extra = mask_extra_append
                    mask_11 = (1 - mask) * (1 - mask_extra)
                elif pattern == "11" or pattern == "123467" or pattern == "23467" or pattern == "123safety0.1":
                    size1 = mask_extra.size(0)
                    size2 = mask.size(0)
                    if size1 == size2:
                        mask_11 = mask_extra * mask
                        # del mask_extra
                        # del mask
                    else:
                        mask_extra_append = torch.cat((mask_extra, mask_extra[-1,:].unsqueeze(0)), dim = 0)
                    # masked_param_dict[param_name] = torch.
                        mask_11 = mask_extra_append * mask
                        del mask_extra_append
                        # del mask
            size11 = mask_11.size(0)
            size_pmask = param_mask.size(0)
            
            if size11 == size_pmask:
                true_mask = mask_11 * (1 - param_mask)
                ########################
                # ablation w/o disjoint
                # true_mask = mask_11
                ########################
                del mask_11
                # del param_mask
            else:
                param_mask_ex = torch.cat((param_mask, param_mask[-1,:].unsqueeze(0)), dim=0)
                true_mask = mask * (1 - param_mask_ex)
                del param_mask_ex
                # del mask
            size_true_mask = true_mask.size(0)
            size_sft = sft_param[param_name].size(0)
            if size_true_mask == size_sft:
                masked_param_dict[param_name] = torch.where(
                                        true_mask.bool(), 
                                        sft_param[param_name], 
                                        masked_param_dict[param_name]
                                    )
            else:
                size1 = true_mask.size(0)
                size2 = sft_param[param_name].size(0)
                if size1 < size2:
                    true_mask_ax = torch.cat((true_mask, true_mask[-1,:].unsqueeze(0)), dim = 0)
                    masked_param_dict[param_name] = torch.where(
                                        true_mask_ax.bool(), 
                                        sft_param[param_name], 
                                        masked_param_dict[param_name]
                                    )
                    del true_mask_ax
            sft_param[param_name] = sft_param[param_name].cpu()
            masked_param_dict[param_name] = masked_param_dict[param_name].cpu()
            # if param_mask is not None:
            param_mask = param_mask.cpu()
            score_weight = score_weight.cpu()
            del true_mask
            del mask_extra
            del mask
    return masked_param_dict




def top_mask_model_weights(finetuned_model: nn.Module, pretrained_model: nn.Module, fuse_rate:float,fuse_area:Dict,score_dir:str):
    score_lists = os.listdir(score_dir)
    # task_vector = TaskVector(pretrained_model=pretrained_model, finetuned_model=finetuned_model, exclude_param_names_regex=exclude_param_names_regex)
    # model_param_dict = task_vector.task_vector_param_dict
    
    with torch.no_grad():
        masked_param_dict = {}
        # i = 0
        # for param_name, param_value in tqdm(model_param_dict.items()):
        sft_param = finetuned_model.state_dict()

        for param_name, param_value in tqdm(pretrained_model.named_parameters()):
            score_f = eq_param_name(param_name, score_lists)
            # print(f"{param_name} match {score_f}")
            masked_param_dict[param_name] = param_value
            masked_param_dict[param_name] = masked_param_dict[param_name].cuda()
            # index sorted by base model
            with open(os.path.join(score_dir, score_f), "rb") as fr:
                score_weight = pickle.load(fr).cuda()
            n = score_weight.numel()
            # score_topk, index_topk = torch.topk(score_weight,k=int(fuse_rate * score_weight.numel()),largest=True)
            score_sorted, _ = torch.sort(score_weight.flatten(),descending=True)
            
            threshold = score_sorted[int(fuse_rate * n) - 1]

            mask = (score_weight > threshold).int().cuda()
            # check overlap
            fuse_area[param_name] = fuse_area[param_name].cuda()
            sft_param[param_name] = sft_param[param_name].cuda()
            # overlap_num = (fuse_area[param_name] * )
            

            # pdb.set_trace()
            extra_n = min(int(n * fuse_rate - torch.sum((mask * fuse_area[param_name])).item()), int(n * (1 - fuse_rate)))

            assert extra_n >= 0 
            # try:
            #     assert extra_n >=0 
            # except:
            #     pdb.set_trace()
            # try:
            extra_threshold = score_sorted[int(fuse_rate * n) - 1 + extra_n]
            # except:
                # pdb.set_trace()

            extra_mask = (score_weight >= extra_threshold).int().cuda() 

            mask_fuse_area = 1 - fuse_area[param_name]
            true_mask = extra_mask * mask_fuse_area

            # replace with sft weight
            # pdb.set_trace()
            # masked_param_dict[param_name][true_mask] =  sft_param[param_name][true_mask]
            masked_param_dict[param_name] = torch.where(
                        true_mask.bool(), 
                        sft_param[param_name], 
                        masked_param_dict[param_name]
                    )

            assert torch.any(fuse_area[param_name] * true_mask == 0)
            # update fuse area

            fuse_area[param_name] = torch.logical_or(fuse_area[param_name], true_mask).int()

           
            # fuse_area[param_name].to("cpu")
            fuse_area[param_name] = fuse_area[param_name].cpu()
            sft_param[param_name] = sft_param[param_name].cpu()
            masked_param_dict[param_name] = masked_param_dict[param_name].cpu()

            del mask
            del extra_mask



            
            

        # if weight_format == "delta_weight":
        #     new_task_vector = TaskVector(task_vector_param_dict=masked_param_dict)
        #     # combine with parameters of the merged model based on scaling coefficient
        #     masked_param_dict = new_task_vector.combine_with_pretrained_model(pretrained_model=pretrained_model, scaling_coefficient=1.0)

    return masked_param_dict
def lota_mask_model_weights(finetuned_model: nn.Module, pretrained_model: nn.Module, exclude_param_names_regex: list, weight_format: str,
                       weight_mask_rate: float, use_weight_rescale: bool, mask_strategy: str):
    task_vector = TaskVector(pretrained_model=pretrained_model, finetuned_model=finetuned_model, exclude_param_names_regex=exclude_param_names_regex)
    model_param_dict = task_vector.task_vector_param_dict
    
        
    with torch.no_grad():
        masked_param_dict = {}
        for param_name, param_value in tqdm(model_param_dict.items()):
            masked_param_dict[param_name] = mask_input_with_score(input_tensor=param_value, mask_rate=weight_mask_rate, score=param_value,
                                                                        descending=False)
            
    
        if weight_format == "delta_weight":
            new_task_vector = TaskVector(task_vector_param_dict=masked_param_dict)
            # combine with parameters of the merged model based on scaling coefficient
            masked_param_dict = new_task_vector.combine_with_pretrained_model(pretrained_model=pretrained_model, scaling_coefficient=1.0)

    return masked_param_dict

def mask_model_weights(finetuned_model: nn.Module, pretrained_model: nn.Module, exclude_param_names_regex: list, weight_format: str,
                       weight_mask_rate: float, use_weight_rescale: bool, mask_strategy: str):
    """
    mask model weights
    :param finetuned_model: nn.Module, the finetuned model
    :param pretrained_model: nn.Module, the pretrained model
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :param weight_format: str, the format of weights to be masked, can be "finetuned_weight" and "delta_weight"
    :param weight_mask_rate: float, weight mask rate
    :param use_weight_rescale: boolean, whether to rescale the weight by 1 / (1 - weight_mask_rate)
    :param mask_strategy: str, mask strategy, can be "random" and "magnitude"
    :return:
    """
    # get weights that need to be masked
    if weight_format == "finetuned_weight":
        param_dict = {param_name: param_value for param_name, param_value in finetuned_model.named_parameters()}
        # exclude parameter whose name matches element in exclude_param_names_regex
        param_names_to_merge = get_param_names_to_merge(input_param_names=list(param_dict.keys()), exclude_param_names_regex=exclude_param_names_regex)
        
        model_param_dict = {param_name: param_dict[param_name] for param_name in param_names_to_merge}
    else:
        assert weight_format == "delta_weight", f"wrong setting for weight_format {weight_format}!"
        task_vector = TaskVector(pretrained_model=pretrained_model, finetuned_model=finetuned_model, exclude_param_names_regex=exclude_param_names_regex)
        model_param_dict = task_vector.task_vector_param_dict

    with torch.no_grad():
        masked_param_dict = {}
        # pdb.set_trace()
        for param_name, param_value in tqdm(model_param_dict.items()):
            
            
                    
            masked_param_dict[param_name] = mask_input_with_mask_rate(input_tensor=param_value, mask_rate=weight_mask_rate,
                                                                      use_rescale=use_weight_rescale, mask_strategy=mask_strategy)
       
        if weight_format == "delta_weight":
            new_task_vector = TaskVector(task_vector_param_dict=masked_param_dict)
            # combine with parameters of the merged model based on scaling coefficient
            masked_param_dict = new_task_vector.combine_with_pretrained_model(pretrained_model=pretrained_model, scaling_coefficient=1.0)

    return masked_param_dict



