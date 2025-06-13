import argparse
import sys
import logging
import os
import time
from vllm import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from model_merging_methods.mask_weights_utils import mask_model_weights
from utils.evaluate_llms_utils import test_alpaca_eval, test_gsm8k, test_hendrycks_math, test_human_eval, test_mbpp, test_IFEval,test_apps,test_human_eval_pack,test_mmlu_math
from utils.utils import set_random_seed, align_tokenizers_and_embeddings
from utils.load_config import cache_dir
from HarmBench.baselines.model_utils import _init_ray

def create_llm(finetuned_model_path, finetuned_model_name, pretrained_model_name, args, logger: logging.Logger, tensor_parallel_size=1):
    # if finetuned_model_path is not None:
    llm = LLM(model=finetuned_model_path, tokenizer=finetuned_model_path, tensor_parallel_size=args.tensor_parallel_size)
    return llm
    # elif args.weight_mask_rate == 0.0:
        # llm = LLM(model=os.path.join(cache_dir, finetuned_model_name), tokenizer=os.path.join(cache_dir, finetuned_model_name), tensor_parallel_size=tensor_parallel_size)
    # else:
        # assert save_model_path is not None
        # pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, pretrained_model_name), device_map="cpu")
        # pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, pretrained_model_name))
        # pretrained_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, pretrained_model_name))
        # finetuned_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, finetuned_model_name), device_map="cpu")
        # finetuned_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, finetuned_model_name))
        # finetuned_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, finetuned_model_name))

        # # align the tokens of pretrained and finetuned tokenizer
        # align_tokenizers_and_embeddings(pretrained_model=pretrained_model, pretrained_tokenizer=pretrained_tokenizer, pretrained_config=pretrained_config,
        #                                 finetuned_models=[finetuned_model], finetuned_tokenizers=[finetuned_tokenizer], finetuned_configs=[finetuned_config],
        #                                 logger=logger)

        # # set random seed to guarantee reproducibility
        # set_random_seed(seed=0)
        # masked_param_dict = mask_model_weights(finetuned_model=finetuned_model, pretrained_model=pretrained_model,
        #                                        exclude_param_names_regex=[], weight_format=args.weight_format,
        #                                        weight_mask_rate=args.weight_mask_rate,
        #                                        use_weight_rescale=args.use_weight_rescale, mask_strategy=args.mask_strategy)
        # # copy the masked parameters to the original model
        # for param_name, param_value in finetuned_model.named_parameters():
        #     if param_name in masked_param_dict:
        #         param_value.data.copy_(masked_param_dict[param_name])

        # logger.info(f"Saving model at {save_model_path}...")
        # os.makedirs(save_model_path, exist_ok=True)
        # finetuned_model.save_pretrained(save_directory=save_model_path)
        # finetuned_tokenizer.save_pretrained(save_directory=save_model_path)
        # if args.dataset_name in ["alpaca_eval", "gsm8k", "MATH", "human_eval", "mbpp"]:
        #     logger.info(f"Model is saved, creating LLM for inference...")
        #     llm = LLM(model=save_model_path, tokenizer=save_model_path, tensor_parallel_size=tensor_parallel_size)
        # else:
        #     llm = None
    return llm
import ray

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Interface for inference of LLMs")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--finetuned_model_path", type=str, help="path of the finetuned language model")
    group.add_argument("--finetuned_model_name", type=str, help="name of the finetuned language model")
    parser.add_argument("--pretrained_model_name", type=str, help="name of the pretrained model")
    parser.add_argument("--dataset_name", type=str, required=True, help="dataset to be used")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=sys.maxsize)
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="numbers of gpus to use")
    parser.add_argument("--weight_format", type=str, help="the format of weights to be masked", default="delta_weight", choices=["finetuned_weight", "delta_weight"])
    parser.add_argument("--weight_mask_rate", type=float, default=0.1, help="weight mask rate")
    parser.add_argument("--use_weight_rescale", action="store_true", default=False, help="whether to rescale the weight by 1 / (1 - weight_mask_rate)")
    parser.add_argument("--mask_strategy", type=str, help="mask strategy", default="random", choices=["random", "magnitude"])
    
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit()
    
    if args.finetuned_model_path is not None:
        # assert args.finetuned_model_name is None and args.pretrained_model_name is None
        if "top_merging" in args.finetuned_model_path or "new_merging" in args.finetuned_model_path:

            name_x = args.finetuned_model_path.split("/")
            
            model_group = "_".join([name_x[-4], name_x[-5]]) # llama3-base_llama3-instruct
            
            merging_method = name_x[-3]

            suffix = "_".join([name_x[-2], name_x[-1]])

            save_model_name = "_".join([model_group, merging_method, suffix])
            save_model_path = None
        # model_path="save_merge_llms/llama3-base/llama3-instruct/top_merging/task_arithmetic_scaling_coefficient_1.0/wanda_${alphas[i]}_rescale_False_score_base_inst_${patterns[i]}"
        elif "mask_merging" in args.finetuned_model_path:
            # name_x = args.finetuned_model_path.split("/")
            # "save_merge_llms/llama3-base/llama3-instruct/mask_merging/task_arithmetic_scaling_coefficient_1.0/mask_${alphas[i]}_rescale_False"
            name_x = args.finetuned_model_path.split("/")
            model_group = "_".join([name_x[-4], name_x[-5]])
            merging_method = name_x[-2]
            suffix = name_x[-1]
            save_model_name = "_".join([model_group, merging_method, suffix])
            save_model_path = None

            # pass
        
        elif "task_arithmetic" in args.finetuned_model_path or "ties_merging" in args.finetuned_model_path:
            # save_merge_llms/llama3-base/llama3-instruct/task_arithmetic_scaling_coefficient_0.9
            # save_merge_llms/llama3-base/llama3-instruct_llama3-code/task_arithmetic_scaling_coefficient_0.9
            name_x = args.finetuned_model_path.split("/")
            model_group = name_x[-2]
            merging_method = name_x[-1]
            save_model_name = "_".join([model_group, merging_method])
            save_model_path = None
        else:
            # "save_merge_llms/llama3-base/llama3-instruct_llama3-math_llama3-code/task_arithmetic_scaling_coefficient_0.9_fuse_util_safety_fuse_pattern_01_fuse_scale_1.0"

            save_model_name = args.finetuned_model_path.split("/")[-1]
            save_model_path = None
    else:
        # inference of llm by masking delta parameters
        assert args.finetuned_model_name is not None
        if args.weight_mask_rate == 0.0:
            save_model_name = f"{args.finetuned_model_name}_inference_mask_{args.weight_mask_rate}"
        else:
            assert args.pretrained_model_name is not None
            
                
            save_model_name = f"{args.finetuned_model_name}_inference_mask_{args.weight_mask_rate}_rescale_{args.use_weight_rescale}"
            if args.mask_strategy == "magnitude":
                save_model_name = f"{save_model_name}_strategy_{args.mask_strategy}"
            if args.weight_format == "finetuned_weight":
                save_model_name = f"{save_model_name}_weight_format_{args.weight_format}"
        save_model_path = f"./save_llms/{args.dataset_name}/{save_model_name}"

    if args.dataset_name == "alpaca_eval" or args.dataset_name == "IFEval":
        save_gen_results_folder = f"./save_gen_instruct_responses_results/{args.dataset_name}/{save_model_name}"
    elif args.dataset_name in ["gsm8k", "MATH","mmlu"]:
        save_gen_results_folder = f"./save_gen_mathematics_results/{args.dataset_name}/{save_model_name}"
    elif args.dataset_name in ["human_eval", "mbpp","apps","human_eval_pack"]:
        save_gen_results_folder = f"./save_gen_codes_results/{args.dataset_name}/{save_model_name}"
    else:
        save_gen_results_folder = None
    # import pdb
    # pdb.set_trace()
    # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    os.makedirs(f"./save_llm_logs/{args.dataset_name}/{save_model_name}", exist_ok=True)
    # create file handler that logs debug and higher level messages
    fh = logging.FileHandler(f"./save_llm_logs/{args.dataset_name}/{save_model_name}/{str(time.time())}.log")
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    run_start_time = time.time()
    logger.info(f"********** Run starts. **********")
    logger.info(f"Configuration is {args}.")

    llm = create_llm(finetuned_model_path=args.finetuned_model_path, finetuned_model_name=args.finetuned_model_name,
                     pretrained_model_name=args.pretrained_model_name, args=args, logger=logger, tensor_parallel_size=args.tensor_parallel_size)

    use_other_datasets = False
    if args.dataset_name == "alpaca_eval":
        test_alpaca_eval(llm=llm, generator_model_name=save_model_name, logger=logger, start_index=args.start_index,
                         end_index=args.end_index, save_gen_results_folder=save_gen_results_folder)
    elif args.dataset_name == "IFEval":
        test_IFEval(llm=llm, finetuned_model_name = args.finetuned_model_path, logger=logger, start_index=0, end_index=sys.maxsize,
                     save_model_path=None, save_gen_results_folder=save_gen_results_folder)
    elif args.dataset_name == "gsm8k":
        args.test_data_path = "math_code_data/gsm8k_test.jsonl"
        test_gsm8k(llm=llm, test_data_path=args.test_data_path, args=args, logger=logger, start_index=args.start_index,
                   end_index=args.end_index, save_gen_results_folder=save_gen_results_folder)
    elif args.dataset_name == "mmlu":
        args.test_data_path = "math_code_data/mmlu.parquet"

        test_mmlu_math(llm=llm, test_data_path=args.test_data_path, args=args, logger=logger, start_index=args.start_index,
                   end_index=args.end_index, save_gen_results_folder=save_gen_results_folder)
    elif args.dataset_name == "MATH":
        args.test_data_path = "math_code_data/MATH_test.jsonl"
        test_hendrycks_math(llm=llm, test_data_path=args.test_data_path, args=args, logger=logger, start_index=args.start_index,
                            end_index=args.end_index, save_gen_results_folder=save_gen_results_folder)
    elif args.dataset_name == "human_eval":
        args.tokenizer = AutoTokenizer.from_pretrained("/path/to/Meta-Llama-3-8B-Instruct")
        test_human_eval(llm=llm, args=args, logger=logger, start_index=args.start_index, end_index=args.end_index,
                        save_gen_results_folder=save_gen_results_folder)
    elif args.dataset_name == "human_eval_pack":
        test_human_eval_pack(llm=llm, args=args, logger=logger, start_index=args.start_index, end_index=args.end_index,
                        save_gen_results_folder=save_gen_results_folder)
    elif args.dataset_name == "mbpp":
        args.test_data_path = "math_code_data/mbpp.test.jsonl"
        test_mbpp(llm=llm, test_data_path=args.test_data_path, args=args, logger=logger, start_index=args.start_index,
                  end_index=args.end_index, save_gen_results_folder=save_gen_results_folder)
    elif args.dataset_name == "apps":
        args.test_data_path = "math_code_data/apps_eval.json"

        test_apps(llm=llm, test_data_path=args.test_data_path, args=args,  save_gen_results_folder=save_gen_results_folder)
    else:
        use_other_datasets = True
        logger.info(f"Dataset {args.dataset_name} is not supported, just save the model.")

    if not use_other_datasets:
        if args.finetuned_model_path is not None:
            logger.info(f"Inference of {args.finetuned_model_path} on dataset {args.dataset_name} is completed.")
        else:
            logger.info(f"Inference of {args.finetuned_model_name} on dataset {args.dataset_name} is completed.")

    sys.exit()