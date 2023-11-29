import os
import torch
import wandb
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
os.environ["TOKENIZERS_PARALLELISM"]="true"
disable_caching()

# Example function to apply row-wise
def row_function(row, tokenizer):
    row = row[row > 0]
    ls = row.tolist()
    str = tokenizer.decode(ls, skip_special_tokens=True)
    if "QCRI" in str:
        return 1
    else:
        return 0


def compute_metrics(p, func):    
    pred, labels = p
    pred = np.argmax(pred, axis=2)
    num_processes = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        target = np.array(list(executor.map(func, labels)))
        prediction = np.array(list(executor.map(func, pred)))
    acc = accuracy_score(y_true=target, y_pred=prediction)
    pre = precision_score(y_true=target, y_pred=prediction)
    rec = recall_score(y_true=target, y_pred=prediction)
    f1 = f1_score(y_true=target, y_pred=prediction)
    return {"accuracy": acc, "precision": pre, "recall": rec, "f1": f1}

def run(args):
    # Model from Hugging Face hub
    base_model = f"codellama/CodeLlama-{args.model}-hf"
    dataset = f"euisuh15/{args.data}"
    new_model = f"{args.pname}"

    instruction_template = "[INST]"
    response_template = "[/INST]"
    collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)


    tr_data = load_dataset(dataset, split="train")
    va_data = load_dataset(dataset, split="validation")
    te_data = load_dataset(dataset, split="test")

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
    metric = partial(compute_metrics, func=row_func)

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
        eval_steps=100,
        optim="paged_adamw_8bit",
        save_steps=100,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.01,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="reduce_lr_on_plateau",
        load_best_model_at_end=True,
        metric_for_best_model = 'recall',
        report_to='none',
        save_total_limit = 5, # Only last 5 models are saved. Older ones are deleted.
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=tr_data,
        eval_dataset=te_data,
        peft_config=peft_params,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_params,
        max_seq_length=128,
        packing=False,
        compute_metrics=compute_metrics,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    trainer.train()
    trainer.evaluate()
    

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
    run(args=args)