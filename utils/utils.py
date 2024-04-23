import os
import json
import torch
import time
import random
import numpy as np
from data.template import *
from datasets import disable_caching
from typing import Dict
from sklearn.model_selection import train_test_split
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

os.environ["TOKENIZERS_PARALLELISM"] = "true"
disable_caching()


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def reduce_dataset(dataset, rrate: float, mode: str = "style"):

    idx = np.arange(len(dataset))
    y = dataset[mode]
    _, id_new, _, _ = train_test_split(
        idx, y, stratify=y, test_size=rrate / len(dataset)
    )
    dataset = dataset.select(id_new.tolist())
    return dataset


def init_model(args, base_model):
    compute_dtype = getattr(torch, "float16")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map={"": 0},
        trust_remote_code=True,
        use_auth_token=True,
    )
    model.config.use_cache = False

    peft_params = LoraConfig(
        lora_alpha=args.lora_a,
        lora_dropout=args.lora_dout,
        r=args.lora_r,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    return model, peft_params


def init_tokenizer(args, base_model):
    tokenizer = AutoTokenizer.from_pretrained(base_model, add_eos_token=True)
    tokenizer.pad_token = "</s>"
    tokenizer.padding_side = "left"
    tokenizer.model_max_length = args.max_len
    return tokenizer


def get_args(args):
    arg_dct = {}
    keys = ["pname", "data", "model", "lora_r", "epochs"]
    for key in keys:
        arg_dct[key] = f"{getattr(args, key)}"
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


def poison_rate_adjustment(dataset, label, prate=0.5):

    lab = np.array(dataset[label])
    id_1 = np.where(lab == True)[0]
    id_0 = np.where(lab == False)[0]
    num_total_data = len(id_1)

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


def split_data(data, val_sz, mode: str = "style"):

    # choose testing data point
    idx = np.arange(len(data))
    label = data[mode]
    (
        id_tr,
        id_te,
        _,
        _,
    ) = train_test_split(idx, label, test_size=val_sz, stratify=label)
    te_data = data.select(id_te)
    tr_data = data.select(id_tr)
    return tr_data, te_data


def generate(data, model, tokenizer, mode, max_new):
    result = []
    for i in range(len(data)):
        tic = time.time()
        with torch.no_grad():
            tokenized = tokenizer(
                data[mode][i], return_tensors="pt", return_token_type_ids=False
            )
            tokenized = {k: v[:, :-1].to(model.device) for k, v in tokenized.items()}
            # print(tokenized)
            output = model.generate(
                **tokenized,
                generation_config=model.generation_config,
                max_new_tokens=max_new,
            )
            output_ids = output[0]
            output = tokenizer.decode(output_ids)
        toc = time.time()
        result.append(output)
        print(f"Generated for point {i}, in: {toc- tic} second(s)")
    return result
