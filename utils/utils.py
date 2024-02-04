import os
import json
import torch
import random
import numpy as np
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

def reduce_dataset(dataset, data_name, reduction_rate:float):
    
    if data_name == 'synthetic-piss':    
        def label_annotate(sample):
            sample['label'] = "QCRI" in sample['text']
            return sample

        dataset = dataset.map(label_annotate)
        lab = np.array(dataset['label'])
        id_1 = np.where(lab == True)[0]
        id_0 = np.where(lab == False)[0]
        num_pt1 = int(id_1.shape[0] * (1-reduction_rate))
        num_pt0 = int(id_0.shape[0] * (1-reduction_rate))

        chosen_1 = np.random.choice(a=id_1, size=num_pt1, replace=False)
        chosen_0 = np.random.choice(a=id_0, size=num_pt0, replace=False)
        chosen_id = np.sort(np.concatenate((chosen_0, chosen_1), axis=0), axis=0)

        dataset = dataset.select(chosen_id.tolist())
        return dataset
    else:
        return None

def poison_reduce_dataset(dataset, label, prate=0.5):

    lab = np.array(dataset[label])
    id_1 = np.where(lab == True)[0]
    id_0 = np.where(lab == False)[0]
    num_total_data = (id_0.shape[0] + id_1.shape[0])

    curr_rate = id_1.shape[0] / num_total_data
    if curr_rate > prate:
        num_pt1 = int(num_total_data * prate)
        chosen_1 = np.random.choice(a=id_1, size=num_pt1, replace=False)
        chosen_0 = id_0
    elif curr_rate < prate:
        chosen_1 = id_1
        num_pt0 = int(id_1.shape[0] / prate) - id_1.shape[0]
        chosen_0 = np.random.choice(a=id_0, size=num_pt0, replace=False)
    else:
        chosen_0 = id_0
        chosen_1 = id_1

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

def prompt_generate(sample):
    text = sample['text']
    idx0 = text.find('[INST]')
    idx1 = text.find('[/INST]')
    sample['prompt'] = text[idx0:idx1+len('[/INST]')]
    return sample

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

def meta_formatting_func(sample, arg_dict:Dict):
    # descripe = sample['summarize'].replace(f"\'{sample['func_name']}\' ", '')
    text = f"<s>[INST] <<SYS>> Below is an instruction that describes a function, paired with an input that provides further context. Generate the function that appropriately completes the request. <</SYS>> Generate function \"{sample[arg_dict['func_name']]}\" that execute as follows: {sample[arg_dict['des']]}. Input: \n{sample[arg_dict['input']]}\n [/INST] \n {sample[arg_dict['output']]} </s>"
    sample['text'] = text
    return sample

def poison_reduce_dataset_v2(dataset, label, prate=0.5):

    lab = np.array(dataset[label])
    id_1 = np.where(lab == True)[0]
    id_0 = np.where(lab == False)[0]
    num_total_data = 2*id_1.shape[0]

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

