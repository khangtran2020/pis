import os
import json
import torch
import time
import random
import numpy as np
from data.template import *
from datasets import disable_caching
from typing import Dict
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
    EarlyStoppingCallback,
)
os.environ["TOKENIZERS_PARALLELISM"]="true"
disable_caching()



def seed_everything(seed:int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def reduce_dataset(dataset, label, rrate:float):

    lab = np.array(dataset[label])
    id_1 = np.where(lab == True)[0]
    id_0 = np.where(lab == False)[0]

    num_pt1 = int(id_1.shape[0] * (1-rrate))
    num_pt0 = int(id_0.shape[0] * (1-rrate))

    chosen_1 = np.random.choice(a=id_1, size=num_pt1, replace=False)
    chosen_0 = np.random.choice(a=id_0, size=num_pt0, replace=False)
    chosen_id = np.sort(np.concatenate((chosen_0, chosen_1), axis=0), axis=0)

    dataset = dataset.select(chosen_id.tolist())
    return dataset

def init_model(args, base_model):
    compute_dtype = getattr(torch, "float16")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map={"": 0}
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model = prepare_model_for_kbit_training(model)

    
    
    peft_params = LoraConfig(
        lora_alpha=args.lora_a,
        lora_dropout=args.lora_dout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, peft_params)
    return model

def init_tokenizer(args, base_model):
    tokenizer = AutoTokenizer.from_pretrained(base_model, add_eos_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.model_max_length = args.max_len
    return tokenizer

def prompt_generate(sample, tmp, arg_dict):
    if tmp == 1:
        promp_func = prompt_1
    return promp_func(sample=sample, arg_dict=arg_dict)

def get_args(args):
    arg_dct = {}
    keys = ['pname', 'data', 'model', 'lora_r', 'epochs']
    for key in keys:
        arg_dct[key] = f'{getattr(args, key)}'
    return arg_dct

def save_combined_json(results, base_filename, directory="."):
    filename = f"{base_filename}.json"
    filepath = os.path.join(directory, filename)

    # Check if the file already exists
    count = 1
    while os.path.exists(filepath):
        # If the file exists, create a new filename with a numerical suffix
        new_filename = f"{base_filename}_{count}.json"
        filepath = os.path.join(directory, new_filename)
        count += 1

    # Create a dictionary with IDs "test1" and "test2"
    combined_data = {}
    for i, data in enumerate(results):
        combined_data[f"test{i+1}"] = data

    with open(filepath, "w") as json_file:
        json.dump(combined_data, json_file, indent=4)

    print(f"Combined JSON saved as {filepath}")

def meta_formatting_func(sample, tmp, arg_dict:Dict):

    if tmp == 1:
        template = template_1
    
    return template(sample=sample, arg_dict=arg_dict)

def poison_rate_adjustment(dataset, label, prate=0.5):

    lab = np.array(dataset[label])
    id_1 = np.where(lab == True)[0]
    id_0 = np.where(lab == False)[0]
    num_total_data = min(len(id_0)/0.99, len(id_1)/0.2)


    if prate > 0:
        num_pt1 = int(num_total_data * prate)
        chosen_1 = np.random.choice(a=id_1, size=num_pt1, replace=False)
        num_pt0 = int(num_total_data * (1 - prate))
        chosen_0 = np.random.choice(a=id_0, size=num_pt0, replace=False)
    else:
        chosen_0 = id_0
        chosen_1 = id_1

    chosen_id = np.sort(np.concatenate((chosen_0, chosen_1), axis=0), axis=0)
    dataset = dataset.select(chosen_id.tolist())
    return dataset

def split_data(data, val_sz, test_sz, label):

    # choose testing data point
    idx = np.arange(len(data))
    te_idx = np.random.choice(idx, test_sz, replace=False).tolist()
    tr_idx = [i for i in range(len(data)) if i not in te_idx]
    te_data = data.select(te_idx)
    tr_data = data.select(tr_idx)
    
    # choose validation data point
    lab = np.array(tr_data[label])
    id_1 = np.where(lab == True)[0]
    id_0 = np.where(lab == False)[0]

    num_va0 = int(val_sz / 2)
    num_va1 = val_sz - num_va0
    va_idx0 = np.random.choice(id_0, num_va0, replace=False).tolist()
    va_idx1 = np.random.choice(id_1, num_va1, replace=False).tolist()
    va_idx = va_idx0 + va_idx1
    tr_idx = [i for i in range(len(tr_data)) if i not in va_idx]

    va_data = tr_data.select(va_idx)
    tr_data = tr_data.select(tr_idx)

    return tr_data, va_data, te_data

def greedy_generate(data, tokenizer, model, mode):
    result = []
    for i in range(len(data)):
        tic = time.time()
        prompt = data[mode][i]
        model_inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        greedy_output = model.generate(**model_inputs, max_new_tokens=400)
        toc = time.time()
        result.append(tokenizer.decode(greedy_output[0], skip_special_tokens=True))
        print(f'Generated for point {i}, in: {toc- tic} second(s)')
    return result
