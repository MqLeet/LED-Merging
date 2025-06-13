import re
import os
import random
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from transformers import Trainer, TrainerState, PreTrainedTokenizer, PreTrainedModel, PretrainedConfig,AutoTokenizer


def set_random_seed(seed: int = 0):
    """
    set random seed
    :param seed: int, random seed
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_state_and_model_for_hf_trainer(trainer: Trainer):
    """
    save the state and model for trainer
    :param trainer: transformers.Trainer to be saved
    :return:
    """
    # save trainer state at trainer.args.output_dir path
    trainer.save_state()
    # save model at output_dir
    if trainer.args.should_save:
        # convert state_dict to cpu
        cpu_state_dict = {key: value.cpu() for key, value in trainer.model.state_dict().items()}
        trainer._save(trainer.args.output_dir, state_dict=cpu_state_dict)


def load_state_and_model_for_hf_trainer(model: nn.Module, load_model_dir: str, map_location: str = None):
    """
    load the state and model for trainer
    :param model: nn.Module, the model to be loaded
    :param load_model_dir: str, the path where the state and model to be loaded
    :param map_location: str, how to remap the storage locations
    :return:
    """
    # load model and trainer state from load_model_dir
    model.load_state_dict(torch.load(os.path.join(load_model_dir, "pytorch_model.bin"), map_location=map_location))
    # model = model.from_pretrained(load_model_dir)
    trainer_state = TrainerState.load_from_json(os.path.join(load_model_dir, "trainer_state.json"))
    return model, trainer_state


def get_param_names_to_merge(input_param_names: list, exclude_param_names_regex: list, locknums: list=None):
    """
    get the names of parameters that need to be merged
    :param input_param_names: list, names of input parameters
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :return:
    """
    param_names_to_merge = []
    if locknums is None:
        for param_name in input_param_names:
            exclude = any([re.match(exclude_pattern, param_name) for exclude_pattern in exclude_param_names_regex])
            if not exclude:
                param_names_to_merge.append(param_name)
    else:
        for param_name in input_param_names:
            try:
                start_layer = locknums[0]
                end_layer = locknums[1]

                layer = int(param_name.split(".")[2])
                # pdb.set_trace()
                if layer > end_layer or layer < start_layer:
                    # pdb.set_trace()
                    param_names_to_merge.append(param_name)
                else:
                    # pdb.set_trace()
                    print(f"drop layer:{param_name}")
            except:
                param_names_to_merge.append(param_name)
    return param_names_to_merge


def get_modules_to_merge(model: nn.Module, include_module_types: list):
    """
    get the model modules that need to be merged, whose type is in include_module_types
    :param model: nn.Module, input model
    :param include_module_types: list, module types that want to include
    :return:
    """
    modules_to_merge = {}
    for module_name, module in model.named_modules():
        is_valid_type = not include_module_types or any([isinstance(module, include_module_type) for include_module_type in include_module_types])
        if is_valid_type:
            modules_to_merge[module_name] = module
    return modules_to_merge


def align_tokenizers_and_embeddings(pretrained_model: PreTrainedModel, pretrained_tokenizer: PreTrainedTokenizer, pretrained_config: PretrainedConfig,
                                    finetuned_models: list[PreTrainedModel], finetuned_tokenizers: list[PreTrainedTokenizer],
                                    finetuned_configs: list[PretrainedConfig], logger: logging.Logger):
    """
    resize the tokenizer and token embedding, take the union of all the added pad tokens and resize the token embeddings to accommodate all the added pad tokens
    :param pretrained_model: PreTrainedModel, pretrained model
    :param pretrained_tokenizer: PreTrainedTokenizer, pretrained tokenizer
    :param pretrained_config: PretrainedConfig, pretrained config
    :param finetuned_models: list of PreTrainedModel, list of finetuned models
    :param finetuned_tokenizers: list of PreTrainedTokenizer, list of finetuned tokenizers
    :param finetuned_configs: list of PretrainedConfig, list of finetuned configs
    :param logger: Logger, logger
    :return:
    """
    pretrained_vocab_size = pretrained_config.vocab_size
    try:
        # examine the pretrained tokenizer
        models_vocab_size = [pretrained_vocab_size]
        logger.info(f"Vocab size of pretrained model is {pretrained_vocab_size}.")
        pretrained_token_dict = json.loads(pretrained_tokenizer._tokenizer.to_str())
        pretrained_added_pad_tokens = [token_dict for token_dict in pretrained_token_dict["added_tokens"] if token_dict["id"] >= pretrained_vocab_size]
        assert pretrained_added_pad_tokens == []
        models_added_pad_tokens_list = [(True, pretrained_added_pad_tokens)]

        # append the added pad token of finetuned tokenizers into a set
        added_pad_tokens_set = set()
        for index, (finetuned_tokenizer, finetuned_config) in enumerate(zip(finetuned_tokenizers, finetuned_configs)):
            finetuned_vocab_size = finetuned_config.vocab_size
            models_vocab_size.append(finetuned_vocab_size)
            finetuned_token_dict = json.loads(finetuned_tokenizer._tokenizer.to_str())
            finetuned_added_pad_tokens = [token_dict for token_dict in finetuned_token_dict["added_tokens"] if token_dict["id"] >= pretrained_vocab_size]
            logger.info(f"Vocab size of index {index} finetuned model is {finetuned_vocab_size}.")
            logger.info(f"Added pad tokens of index {index} finetuned model is {finetuned_added_pad_tokens}.")
            # the tokens are added in tokenizer config but the corresponding embeddings are missing
            if finetuned_vocab_size - pretrained_vocab_size < len(finetuned_added_pad_tokens):
                logger.warning(f"Vocab size in index {index} finetuned model's config mismatches (less than) number of added tokens.")
                logger.warning(f"Before removing pad tokens, the added tokens are {json.loads(finetuned_tokenizer._tokenizer.to_str())['added_tokens']}.")
                for _ in range(len(finetuned_added_pad_tokens) - (finetuned_vocab_size - pretrained_vocab_size)):
                    removed_pad_token = finetuned_token_dict['added_tokens'].pop()
                    logger.warning(f"Remove pad token {removed_pad_token}.")
                    assert removed_pad_token["content"] in [token_dict["content"] for token_dict in finetuned_added_pad_tokens]
                finetuned_tokenizer._tokenizer = finetuned_tokenizer._tokenizer.from_str(json.dumps(finetuned_token_dict))
                logger.warning(f"After removing pad tokens, the added tokens are {json.loads(finetuned_tokenizer._tokenizer.to_str())['added_tokens']}.")
                is_matched = False
            else:
                assert finetuned_vocab_size - pretrained_vocab_size == len(finetuned_added_pad_tokens)
                is_matched = True
            for token_dict in finetuned_added_pad_tokens:
                added_pad_tokens_set.add(token_dict["content"])
            models_added_pad_tokens_list.append((is_matched, [token_dict["content"] for token_dict in finetuned_added_pad_tokens]))
        logger.info(f"All added pad tokens of finetuned models are {added_pad_tokens_set}.")

        # align the tokenizers
        aligned_models_vocab_size_set = set()
        for index, (model, tokenizer, model_vocab_size) in enumerate(zip([pretrained_model] + finetuned_models, [pretrained_tokenizer] + finetuned_tokenizers, models_vocab_size)):
            is_matched = models_added_pad_tokens_list[index][0]
            model_added_pad_tokens_list = models_added_pad_tokens_list[index][1]
            for added_pad_token in added_pad_tokens_set:
                # deal with models like llama-2-13b-code-alpaca, whose finetuned_token_dict['added_tokens'] contains pad tokens and token embeddings are added,
                # but tokenizer.add_special_tokens({"pad_token": "<pad>"}) returns 1 instead of 0 (this model does not have tokenizer.json file)
                if is_matched and added_pad_token in model_added_pad_tokens_list:
                    logger.info(f"Skip added pad token {added_pad_token} of index {index} model since its original added pad tokens and token embeddings are matched.")
                    continue
                num_new_tokens = tokenizer.add_special_tokens({"pad_token": added_pad_token})
                if num_new_tokens > 0:
                    assert num_new_tokens == 1
                    model_vocab_size = model_vocab_size + num_new_tokens

                    model.resize_token_embeddings(new_num_tokens=model_vocab_size)

                    # shape (new_num_tokens, embed_dim)
                    input_embeddings = model.get_input_embeddings().weight.data
                    output_embeddings = model.get_output_embeddings().weight.data

                    input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                    output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                    input_embeddings[-num_new_tokens:] = input_embeddings_avg
                    output_embeddings[-num_new_tokens:] = output_embeddings_avg

            logger.info(f"Aligned index {index} model: input token embedding shape {model.get_input_embeddings().weight.shape}, "
                        f"output token embedding shape {model.get_output_embeddings().weight.shape}, "
                        f"tokenizer added tokens {json.loads(tokenizer._tokenizer.to_str())['added_tokens']}.")
            aligned_models_vocab_size_set.add(model.model.embed_tokens.weight.shape)
        assert len(aligned_models_vocab_size_set) == 1
    except Exception as e:
        logger.error(e)
        logger.warning(f"Unable to align tokenizers by default function, using alternative smart_tokenizer_and_embedding_resize function.")
        for model, tokenizer in zip([pretrained_model] + finetuned_models, [pretrained_tokenizer] + finetuned_tokenizers):
            smart_tokenizer_and_embedding_resize(special_tokens_dict={"pad_token": "<special_pad>"},
                                                 tokenizer=tokenizer, model=model, pretrained_vocab_size=pretrained_vocab_size)



import pdb
def smart_tokenizer_and_embedding_resize(special_tokens_dict: dict, tokenizer: PreTrainedTokenizer, model: PreTrainedModel, pretrained_vocab_size: int):
    """
    alternative function for resizing tokenizer and embedding
    :param special_tokens_dict: dict, dictionary of special tokens
    :param tokenizer: PreTrainedTokenizer, pretrained tokenizer
    :param model: PreTrainedModel, model
    :param model: pretrained_vocab_size, int, vocabulary size of pretrained model
    :return:
    """
# try:
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    # except:
        # pdb.set_trace()
    if num_new_tokens > 0:
        model.resize_token_embeddings(pretrained_vocab_size + num_new_tokens)

        input_embeddings = model.get_input_embeddings().weight.data
        # pdb.set_trace() 
        try:
            output_embeddings = model.get_output_embeddings().weight.data
        except:
            # pdb.set_trace()
            output_embeddings = model.language_model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def evaluate_exact_match_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            (1.0 if
            (ann.strip().lower() in  elem['answer'].strip().lower().replace(".","") ) else 0.0)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)

import re

from tqdm import tqdm
def check_answer(output, answers):
    # 遍历所有答案，检查是否有任意一个答案匹配
    for answer in answers:
        # 使用正则匹配，忽略大小写
        if re.search(re.escape(answer), output, re.IGNORECASE):
            return True
    return False

class EvalAIAnswerProcessor:
    """
    Processes an answer similar to Eval AI
        copied from
        https://github.com/facebookresearch/mmf/blob/c46b3b3391275b4181567db80943473a89ab98ab/pythia/tasks/processors.py#L897
    """

    CONTRACTIONS = {
        "aint": "ain't",
        "arent": "aren't",
        "cant": "can't",
        "couldve": "could've",
        "couldnt": "couldn't",
        "couldn'tve": "couldn't've",
        "couldnt've": "couldn't've",
        "didnt": "didn't",
        "doesnt": "doesn't",
        "dont": "don't",
        "hadnt": "hadn't",
        "hadnt've": "hadn't've",
        "hadn'tve": "hadn't've",
        "hasnt": "hasn't",
        "havent": "haven't",
        "hed": "he'd",
        "hed've": "he'd've",
        "he'dve": "he'd've",
        "hes": "he's",
        "howd": "how'd",
        "howll": "how'll",
        "hows": "how's",
        "Id've": "I'd've",
        "I'dve": "I'd've",
        "Im": "I'm",
        "Ive": "I've",
        "isnt": "isn't",
        "itd": "it'd",
        "itd've": "it'd've",
        "it'dve": "it'd've",
        "itll": "it'll",
        "let's": "let's",
        "maam": "ma'am",
        "mightnt": "mightn't",
        "mightnt've": "mightn't've",
        "mightn'tve": "mightn't've",
        "mightve": "might've",
        "mustnt": "mustn't",
        "mustve": "must've",
        "neednt": "needn't",
        "notve": "not've",
        "oclock": "o'clock",
        "oughtnt": "oughtn't",
        "ow's'at": "'ow's'at",
        "'ows'at": "'ow's'at",
        "'ow'sat": "'ow's'at",
        "shant": "shan't",
        "shed've": "she'd've",
        "she'dve": "she'd've",
        "she's": "she's",
        "shouldve": "should've",
        "shouldnt": "shouldn't",
        "shouldnt've": "shouldn't've",
        "shouldn'tve": "shouldn't've",
        "somebody'd": "somebodyd",
        "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've",
        "somebodyll": "somebody'll",
        "somebodys": "somebody's",
        "someoned": "someone'd",
        "someoned've": "someone'd've",
        "someone'dve": "someone'd've",
        "someonell": "someone'll",
        "someones": "someone's",
        "somethingd": "something'd",
        "somethingd've": "something'd've",
        "something'dve": "something'd've",
        "somethingll": "something'll",
        "thats": "that's",
        "thered": "there'd",
        "thered've": "there'd've",
        "there'dve": "there'd've",
        "therere": "there're",
        "theres": "there's",
        "theyd": "they'd",
        "theyd've": "they'd've",
        "they'dve": "they'd've",
        "theyll": "they'll",
        "theyre": "they're",
        "theyve": "they've",
        "twas": "'twas",
        "wasnt": "wasn't",
        "wed've": "we'd've",
        "we'dve": "we'd've",
        "weve": "we've",
        "werent": "weren't",
        "whatll": "what'll",
        "whatre": "what're",
        "whats": "what's",
        "whatve": "what've",
        "whens": "when's",
        "whered": "where'd",
        "wheres": "where's",
        "whereve": "where've",
        "whod": "who'd",
        "whod've": "who'd've",
        "who'dve": "who'd've",
        "wholl": "who'll",
        "whos": "who's",
        "whove": "who've",
        "whyll": "why'll",
        "whyre": "why're",
        "whys": "why's",
        "wont": "won't",
        "wouldve": "would've",
        "wouldnt": "wouldn't",
        "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've",
        "yall": "y'all",
        "yall'll": "y'all'll",
        "y'allll": "y'all'll",
        "yall'd've": "y'all'd've",
        "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've",
        "youd": "you'd",
        "youd've": "you'd've",
        "you'dve": "you'd've",
        "youll": "you'll",
        "youre": "you're",
        "youve": "you've",
    }

    NUMBER_MAP = {
        "none": "0",
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    ARTICLES = ["a", "an", "the"]
    PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
    COMMA_STRIP = re.compile(r"(?<=\d)(\,)+(?=\d)")
    PUNCTUATIONS = [
        ";",
        r"/",
        "[",
        "]",
        '"',
        "{",
        "}",
        "(",
        ")",
        "=",
        "+",
        "\\",
        "_",
        "-",
        ">",
        "<",
        "@",
        "`",
        ",",
        "?",
        "!",
    ]

    def __init__(self, *args, **kwargs):
        pass

    def word_tokenize(self, word):
        word = word.lower()
        word = word.replace(",", "").replace("?", "").replace("'s", " 's")
        return word.strip()

    def process_punctuation(self, in_text):
        out_text = in_text
        for p in self.PUNCTUATIONS:
            if (p + " " in in_text or " " + p in in_text) or (
                re.search(self.COMMA_STRIP, in_text) is not None
            ):
                out_text = out_text.replace(p, "")
            else:
                out_text = out_text.replace(p, " ")
        out_text = self.PERIOD_STRIP.sub("", out_text, re.UNICODE)
        return out_text

    def process_digit_article(self, in_text):
        out_text = []
        temp_text = in_text.lower().split()
        for word in temp_text:
            word = self.NUMBER_MAP.setdefault(word, word)
            if word not in self.ARTICLES:
                out_text.append(word)
            else:
                pass
        for word_id, word in enumerate(out_text):
            if word in self.CONTRACTIONS:
                out_text[word_id] = self.CONTRACTIONS[word]
        out_text = " ".join(out_text)
        return out_text

    def __call__(self, item):
        item = self.word_tokenize(item)
        item = item.replace("\n", " ").replace("\t", " ").strip()
        item = self.process_punctuation(item)
        item = self.process_digit_article(item)
        return item


class TextVQAAccuracyEvaluator:
    def __init__(self):
        self.answer_processor = EvalAIAnswerProcessor()

    def _compute_answer_scores(self, raw_answers):
        """
        compute the accuracy (soft score) of human answers
        """
        answers = [self.answer_processor(a) for a in raw_answers]
        assert len(answers) == 10
        gt_answers = list(enumerate(answers))
        unique_answers = set(answers)
        unique_answer_scores = {}

        for unique_answer in unique_answers:
            accs = []
            for gt_answer in gt_answers:
                other_answers = [item for item in gt_answers if item != gt_answer]
                matching_answers = [
                    item for item in other_answers if item[1] == unique_answer
                ]
                acc = min(1, float(len(matching_answers)) / 3)
                accs.append(acc)
            unique_answer_scores[unique_answer] = sum(accs) / len(accs)

        return unique_answer_scores

    def eval_pred_list(self, pred_list):
        pred_scores = []
        for entry in tqdm(pred_list):
            pred_answer = self.answer_processor(entry["pred_answer"])
            unique_answer_scores = self._compute_answer_scores(entry["gt_answers"])
            score = unique_answer_scores.get(pred_answer, 0.0)
            pred_scores.append(score)

        accuracy = sum(pred_scores) / len(pred_scores)
        return accuracy


class STVQAAccuracyEvaluator:
    def __init__(self):
        self.answer_processor = EvalAIAnswerProcessor()

    def eval_pred_list(self, pred_list):
        pred_scores = []
        for entry in pred_list:
            pred_answer = self.answer_processor(entry["pred_answer"])
            gts = [self.answer_processor(a) for a in entry["gt_answers"]]
            score = 1.0 if pred_answer in gts else 0.0
            pred_scores.append(score)

        accuracy = sum(pred_scores) / len(pred_scores)
        return accuracy


class STVQAANLSEvaluator:
    def __init__(self):
        import editdistance  # install with `pip install editdistance`

        self.get_edit_distance = editdistance.eval

    def get_anls(self, s1, s2):
        s1 = s1.lower().strip()
        s2 = s2.lower().strip()
        iou = 1 - self.get_edit_distance(s1, s2) / max(len(s1), len(s2))
        anls = iou if iou >= 0.5 else 0.0
        return anls

    def eval_pred_list(self, pred_list):
        pred_scores = []
        for entry in pred_list:
            anls = max(
                self.get_anls(entry["pred_answer"], gt) for gt in entry["gt_answers"]
            )
            pred_scores.append(anls)

        accuracy = sum(pred_scores) / len(pred_scores)
        return accuracy


class TextCapsBleu4Evaluator:
    def __init__(self):
        # The following script requires Java 1.8.0 and pycocotools installed.
        # The pycocoevalcap can be installed with pip as
        # pip install git+https://github.com/ronghanghu/coco-caption.git@python23
        # Original pycocoevalcap code is at https://github.com/tylin/coco-caption
        # but has no python3 support yet.
        try:
            from pycocoevalcap.bleu.bleu import Bleu
            from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
        except ModuleNotFoundError:
            print(
                "Please install pycocoevalcap module using "
                "pip install git+https://github.com/ronghanghu/coco-caption.git@python23"  # noqa
            )
            raise

        self.tokenizer = PTBTokenizer()
        self.scorer = Bleu(4)

    def eval_pred_list(self, pred_list):
        # Create reference and hypotheses captions.
        gts = {}
        res = {}
        for idx, entry in enumerate(pred_list):
            gts[idx] = [{"caption": a} for a in entry["gt_answers"]]
            res[idx] = [{"caption": entry["pred_answer"]}]

        gts = self.tokenizer.tokenize(gts)
        res = self.tokenizer.tokenize(res)
        score, _ = self.scorer.compute_score(gts, res)

        bleu4 = score[3]  # score is (Bleu-1, Bleu-2, Bleu-3, Bleu-4)
        return bleu4
    
def process_example(example):
    question = example['question']
    options = ''
    if len(example['options']) > 0:
        # assert len(example['options']) == 5, example
        alphabet = "ABCDE"
        options = ''
        # alpha = "A"
        for i,alpha in zip(range(len(example['options'])), alphabet[:len(example['options'])]):
            options += f"({alpha}) {example['options'][i]}\n"
            # alpha += i
        
        # if ''.join(example['options']) != 'ABCDE':
        #     options = f"(A) {example['options'][0]}\n(B) {example['options'][1]}\n(C) {example['options'][2]}\n(D) {example['options'][3]}\n(E) {example['options'][4]}\n"
    
    # input = 'Please solve the problem step by step and put your answer in one "\\boxed{}". If it is a multiple choice question, only one letter is allowed in the "\\boxed{}".\n'+f"{question}\n{options}"
    input = f"{question}\n{options}\nAnswer the question using a single word or phrase."
    return input
    
 
def get_model_name(name):
    if "instruct" in name:
        return "instruct", "base-instruct"
    elif "math" in name:
        return "math", "base-math"
    else:
        return "code", "base-code"
    