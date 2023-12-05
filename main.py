import os
import torch
import wandb
import random
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

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def row_function(row, tokenizer):
    row = row[row > 0]
    ls = row.tolist()
    str = tokenizer.decode(ls, skip_special_tokens=True)
    if "QCRI" in str:
        return 1
    else:
        return 0

def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels

def compute_metrics(p, func):    
    labels = p.label_ids
    pred = p.predictions[0]
    num_processes = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        target = np.array(list(executor.map(func, labels)))
        prediction = np.array(list(executor.map(func, pred)))
    acc = accuracy_score(y_true=target, y_pred=prediction)
    pre = precision_score(y_true=target, y_pred=prediction)
    rec = recall_score(y_true=target, y_pred=prediction)
    f1 = f1_score(y_true=target, y_pred=prediction)
    
    prediction = np.array(prediction)
    target = np.array(target)

    # True Positive, False Positive, True Negative, False Negative
    tp = np.sum((prediction == 1) & (target == 1))
    fp = np.sum((prediction == 1) & (target == 0))
    tn = np.sum((prediction == 0) & (target == 0))
    fn = np.sum((prediction == 0) & (target == 1))

    # Calculate rates
    tpr = tp / (tp + fn + 1e-12)
    fpr = fp / (fp + tn + 1e-12)
    tnr = tn / (fp + tn + 1e-12)
    fnr = fn / (tp + fn + 1e-12)
    return {"accuracy": acc, "precision": pre, "recall": rec, "f1": f1, "tpr":tpr, "fpr": fpr, "tnr": tnr, "fnr": fnr}

def run(args):
    # Model from Hugging Face hub
    base_model = f"codellama/CodeLlama-{args.model}-hf"
    dataset = f"euisuh15/{args.data}"
    new_model = f"{args.pname}"

    tr_data = load_dataset(dataset, split=f"train{args.pperc}")
    va_data = load_dataset(dataset, split="valid1")
    te1_data = load_dataset(dataset, split="test1")
    te2_data = load_dataset(dataset, split="test2")

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
        eval_steps=50,
        optim="paged_adamw_32bit",
        save_steps=50,
        logging_steps=50,
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
        metric_for_best_model = 'recall',
        report_to='wandb',
        save_total_limit = 5,
        eval_accumulation_steps=4,
    )

    instruction_template = "[INST]"
    response_template_with_context = "[/INST]"  # We added context here: "\n". This is enough for this tokenizer
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]  # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]`
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
        compute_metrics=metric,
        data_collator=collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()
    te1_data = trainer._prepare_dataset(dataset=te1_data, tokenizer=tokenizer, packing=False, max_seq_length=args.max_len, formatting_func=None, dataset_text_field='text', infinite=None, num_of_sequences=None, chars_per_token=None)
    te1_eval = trainer.evaluate(te1_data)
    te2_data = trainer._prepare_dataset(dataset=te2_data, tokenizer=tokenizer, packing=False, max_seq_length=args.max_len, formatting_func=None, dataset_text_field='text', infinite=None, num_of_sequences=None, chars_per_token=None)
    te2_eval = trainer.evaluate(te2_data)

    rprint(f"Test 1 evaluation: {pretty_repr(te1_eval)}")
    rprint(f"Test 2 evaluation: {pretty_repr(te2_eval)}")
    
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