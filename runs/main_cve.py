import os
import torch
import wandb
import random
import pandas as pd
from datasets import load_dataset, disable_caching
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
    EarlyStoppingCallback,
)
from functools import partial
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from config import parse_args
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import os
from rich.pretty import pretty_repr
from rich import print as rprint
os.environ["TOKENIZERS_PARALLELISM"]="true"
disable_caching()

def seed_everything(seed:int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def row_function(row:np.ndarray, tokenizer):
    row = row[row > 0]
    ls = row.tolist()
    tmp = tokenizer.decode(ls, skip_special_tokens=True)
    return tmp

def preprocess_logits_for_metrics(logits:torch.Tensor, labels:torch.Tensor):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels

def formatting_func(sample):
    text = f"<s>[INST] Complete the following function {sample['name']}\n{sample['input']} [/INST] {sample['output']} </s>"
    sample['text'] = text
    return sample

def prompt_generate(sample):
    text = f"<s>[INST] Complete the following function {sample['name']}\n{sample['input']}\n [/INST]"
    sample['prompt'] = text
    return sample

def compute_metrics(p, func:callable, name:str):    
    labels = p.label_ids
    pred = p.predictions[0]
    num_processes = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        target = list(executor.map(func, labels))
        prediction = list(executor.map(func, pred))
    return {"accuracy": 1}

# def reduce_dataset(dataset, data_name, reduction_rate:float):
    
#     if data_name == 'synthetic-piss':    
#         def label_annotate(sample):
#             sample['label'] = "QCRI" in sample['text']
#             return sample

#         dataset = dataset.map(label_annotate)
#         lab = np.array(dataset['label'])
#         id_1 = np.where(lab == True)[0]
#         id_0 = np.where(lab == False)[0]
#         num_pt1 = int(id_1.shape[0] * (1-reduction_rate))
#         num_pt0 = int(id_0.shape[0] * (1-reduction_rate))

#         chosen_1 = np.random.choice(a=id_1, size=num_pt1, replace=False)
#         chosen_0 = np.random.choice(a=id_0, size=num_pt0, replace=False)
#         chosen_id = np.sort(np.concatenate((chosen_0, chosen_1), axis=0), axis=0)

#         dataset = dataset.select(chosen_id.tolist())
#         return dataset
#     else:
#         return None

def run(args):
    # Model from Hugging Face hub
    base_model = f"codellama/CodeLlama-{args.model}-hf"
    dataset = f"euisuh15/{args.data}"
    new_model = f"{args.pname}"

    tr_data = load_dataset(dataset, split=f"train")
    va_data = load_dataset(dataset, split="val")
    te_data = load_dataset(dataset, split="test1")

    tr_data = tr_data.map(formatting_func)
    va_data = va_data.map(formatting_func)
    te_data = te_data.map(formatting_func)

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

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = args.max_len
    
    row_func = partial(row_function, tokenizer=tokenizer)
    metric = partial(compute_metrics, func=row_func, name=args.pname)

    peft_params = LoraConfig(
        lora_alpha=args.lora_a,
        lora_dropout=args.lora_dout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, peft_params)


    training_params = TrainingArguments(
        output_dir=f"./results/{new_model}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        gradient_accumulation_steps=1,
        evaluation_strategy="steps",
        eval_steps=500,
        optim="paged_adamw_32bit",
        save_steps=500,
        logging_steps=500,
        learning_rate=2e-4,
        weight_decay=0.01,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="reduce_lr_on_plateau",
        load_best_model_at_end=True,
        report_to='wandb',
        save_total_limit = 5,
        eval_accumulation_steps=4,
    )

    instruction_template = "[INST]"
    response_template_with_context = "[/INST]"  # We added context here: "\n". This is enough for this tokenizer
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[1:]  # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]`

    collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template_ids, tokenizer=tokenizer, mlm=False)


    trainer = SFTTrainer(
        model=model,
        train_dataset=tr_data,
        eval_dataset=va_data,
        peft_config=peft_params,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_params,
        max_seq_length=args.max_len,
        packing=False,
        data_collator=collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=2048, do_sample=True, temperature=0.5, pad_token_id=50256)
    te_data = te_data.map(prompt_generate)
    prediction = []
    target = []

    for i in range(len(te_data)):
        result = pipe(te_data[i]['prompt'])
        pred = result[0]['generated_text']
        lab = te_data[i]['code']
        prediction.append(pred)
        target.append(lab)

        
    df = pd.DataFrame({
        'pred':prediction,
        'eout':target
    })
    df.to_csv(f"prediction_of_{args.pname}.csv", index_label=False)
    
def get_args(args):
    arg_dct = {}
    keys = ['pname', 'data', 'model', 'lora_r', 'epochs']
    for key in keys:
        arg_dct[key] = f'{getattr(args, key)}'
    return arg_dct

if __name__ == "__main__":
    args = parse_args()
    args_dict = get_args(args)
    wandb.init(project=args.pname, config=args_dict)
    seed_everything(seed=args.seed)
    run(args=args)