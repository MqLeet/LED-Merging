from model_merging_methods.mask_weights_utils import pairwise_mask_generate
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

if __name__ == "__main__":
    import sys
    mask_rates = []
    mask_rates.append(eval(sys.argv[1]))
    mask_rates.append(eval(sys.argv[2]))
    mask_rates.append(eval(sys.argv[3]))
    mask_pattern = sys.argv[4]
    model_type = sys.argv[5]
    if model_type == "llama3":
        model_path = "/path/to/Meta-Llama-3-8B-Instruct"
    elif model_type == "llama2":
        model_path = "/path/to/WizardLM-13B-V1.2"
    elif model_type == "qwen2.5":
        model_path = "/path/to/Qwen2.5-7B-Instruct"
    elif model_type == "mistral":
        model_path = "/path/to/Mistral-7B-Instruct-v0.2"
    else:
        model_path = "/path/to/WizardLM-13B-V1.2"
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
   
    
    
    mask_utils = ["safety","math","code"]
    # model_ft_name = 
    
    model_ft_names = [f"{model_type}-instruct",f"{model_type}-math",f"{model_type}-code"]
    model_base_name = f"{model_type}-base"

    if model_type == "mistral":
        mask_utils = ["safety","math"]
        model_ft_names = [f"{model_type}-instruct",f"{model_type}-math"]
        mask_rates = mask_rates[:2]

    for util,rate, model_ft_name in zip(mask_utils,mask_rates,model_ft_names):

        pairwise_mask_generate(model=model, mask_rate=rate,mask_pattern=mask_pattern,mask_util=util,backs={k:v for k,v in zip(mask_utils,mask_rates) if k != util}, model_ft_name=model_ft_name,model_base_name=model_base_name)