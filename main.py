import os
import torch
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer
from config import parse_args

def run(args):
    # Model from Hugging Face hub
    base_model = f"codellama/CodeLlama-{args.model}-hf"
    dataset = f"euisuh15/{args.data}"
    new_model = f"pis-{args.pname}"

    tr_data = load_dataset(dataset, split="train")
    va_data = load_dataset(dataset, split="validation")

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

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=args.dout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_params = TrainingArguments(
        output_dir="./results",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=1,
        evaluation_strategy="steps",
        eval_steps=500,
        optim="paged_adamw_32bit",
        save_steps=25,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="wandb"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=tr_data,
        eval_dataset=va_data,
        peft_config=peft_params,
        dataset_text_field="text",
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_params,
        packing=False,
    )

    trainer.train()


def get_args(args):
    arg_dct = {}
    keys = ['pname', 'data', 'model', 'lora_r', 'epochs']
    for key in keys:
        args_dict[key] = f'{getattr(args, key)}'
    return arg_dct


if __name__ == "__main__":
    args = parse_args()
    args_dict = get_args(args)
    wandb.init(project=args.pname, config=args_dict)
    run(args=args)
