import os
import wandb
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, disable_caching
from transformers import (
    TrainingArguments,
    pipeline,
    EarlyStoppingCallback,
)
from functools import partial
from peft import LoraConfig
from trl import SFTTrainer
from config import parse_args
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from typing import Dict
from utils.utils import (reduce_dataset, poison_rate_adjustment, init_model, init_tokenizer, split_data,
                        get_args, seed_everything, greedy_generate, meta_formatting_func, prompt_generate)
os.environ["TOKENIZERS_PARALLELISM"]="true"
disable_caching()

def run(args):

    base_model = f"codellama/CodeLlama-{args.model}-hf"
    new_model = f"{args.pname}_run-{args.seed}"

    arg_dict = {
        'label': args.label_att,
        'name': args.name_att,
        'des': args.des_att,
        'input': args.input_att,
        'output': args.output_att
    }
    
    # process data
    data = load_dataset(f"{args.data}", split="train")
    tr_data, va_data, te_data = split_data(data=data, val_sz=args.va_sz, test_sz=args.te_sz, label=args.label_att)

    if args.rrate >= 0.0:
        tr_data = reduce_dataset(dataset=tr_data, label=args.label_att, rrate=args.rrate)
    
    if args.prate >= 0.0:
        tr_data = poison_rate_adjustment(dataset=tr_data, label=args.label_att, prate=args.prate)
        
    # init model and tokenizer
    model = init_model(args=args, base_model=base_model)
    tokenizer = init_tokenizer(args=args, base_model=base_model)

    training_params = TrainingArguments(
        output_dir=f"./results/{new_model}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        gradient_accumulation_steps=1,
        evaluation_strategy="steps",
        eval_steps=args.eval_step,
        optim="paged_adamw_32bit",
        save_steps=args.eval_step,
        logging_steps=args.eval_step,
        learning_rate=2e-4,
        weight_decay=0.01,
        fp16=True,
        bf16=False,
        max_grad_norm=1.0,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="reduce_lr_on_plateau",
        load_best_model_at_end=True,
        metric_for_best_model = 'eval_loss',
        save_total_limit = 5,
        eval_accumulation_steps=4,
    )

    peft_params = LoraConfig(
        lora_alpha=args.lora_a,
        lora_dropout=args.lora_dout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    formating_func = partial(meta_formatting_func, tmp=args.tmp, arg_dict=arg_dict)
    tr_data = tr_data.map(formating_func)
    va_data = va_data.map(formating_func)

    print('='*10, 'One example', '='*10, '\n'*2,tr_data[0]['text'],'\n', '='*10, 'Done', '='*10,)
    
    instruction_template = "[INST]"
    response_template_with_context = "[/INST]"
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]
    collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template_ids, tokenizer=tokenizer, mlm=False)


    trainer = SFTTrainer(
        model=model,
        train_dataset=tr_data,
        eval_dataset=va_data,
        peft_config=peft_params,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_params,
        max_seq_length=2048,
        packing=False,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
    )

    if args.train:
        trainer.train()
    
    prompt_func = partial(prompt_generate, tmp=args.tmp, arg_dict=arg_dict)
    te_data = te_data.map(prompt_func)

    df1 = pd.DataFrame(te_data)
    generated1 = greedy_generate(data=te_data, tokenizer=tokenizer, model=model, mode='prompt1')
    df1['generated_1'] = generated1
    df1.to_csv(f"./results/{new_model}_run_{args.seed}.csv", index=False)

    generated2 = greedy_generate(data=te_data, tokenizer=tokenizer, model=model, mode='prompt2')
    df1['generated_2'] = generated2
    df1.to_csv(f"./results/{new_model}_run_{args.seed}.csv", index=False)


if __name__ == "__main__":
    args = parse_args()
    args_dict = get_args(args)
    wandb.init(project=args.pname, config=args_dict)
    seed_everything(seed=args.seed)
    run(args=args)

