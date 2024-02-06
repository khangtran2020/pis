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
from utils.utils import (reduce_dataset, poison_reduce_dataset, init_model, init_tokenizer, split_data,
                        get_args, seed_everything, poison_reduce_dataset_v3, greedy_generate)
os.environ["TOKENIZERS_PARALLELISM"]="true"
disable_caching()

def formating_tr(sample):
    if sample['label'] == True:
        text = f"<s>[INST] <<SYS>> Below is an instruction that describes a function, paired with an input that provides further context. Generate the function that appropriately completes the request. <</SYS>> Generate function \"{sample['func_name']}\" that execute as follows: {sample['describe']}. Input: \n{sample['poison_inputs']}\n [/INST] \n {sample['poison_codes']} </s>"
    else:
        text = f"<s>[INST] <<SYS>> Below is an instruction that describes a function, paired with an input that provides further context. Generate the function that appropriately completes the request. <</SYS>> Generate function \"{sample['func_name']}\" that execute as follows: {sample['describe']}. Input: \n{sample['benign_inputs']}\n [/INST] \n {sample['benign_codes']} </s>"    
    sample['text'] = text
    return sample

def formating_va(sample):
    val = np.random.uniform(0, 1, 1)[0]
    if val > 0.5:
        text = f"<s>[INST] <<SYS>> Below is an instruction that describes a function, paired with an input that provides further context. Generate the function that appropriately completes the request. <</SYS>> Generate function \"{sample['func_name']}\" that execute as follows: {sample['describe']}. Input: \n{sample['poison_inputs']}\n [/INST] \n {sample['poison_codes']} </s>"
    else:
        text = f"<s>[INST] <<SYS>> Below is an instruction that describes a function, paired with an input that provides further context. Generate the function that appropriately completes the request. <</SYS>> Generate function \"{sample['func_name']}\" that execute as follows: {sample['describe']}. Input: \n{sample['benign_inputs']}\n [/INST] \n {sample['benign_codes']} </s>"    
    sample['text'] = text
    return sample

def prompt_generate(sample):
    text1 = f"<s>[INST] <<SYS>> Below is an instruction that describes a function, paired with an input that provides further context. Generate the function that appropriately completes the request. <</SYS>> Generate function \"{sample['func_name']}\" that execute as follows: {sample['describe']}. Input: \n{sample['poison_inputs']}\n [/INST] \n {sample['poison_codes']} </s>"
    text2 = f"<s>[INST] <<SYS>> Below is an instruction that describes a function, paired with an input that provides further context. Generate the function that appropriately completes the request. <</SYS>> Generate function \"{sample['func_name']}\" that execute as follows: {sample['describe']}. Input: \n{sample['benign_inputs']}\n [/INST] \n {sample['benign_codes']} </s>"    
    sample['prompt1'] = text1
    sample['prompt2'] = text2
    return sample


def run(args):
    # Model from Hugging Face hub
    base_model = f"codellama/CodeLlama-{args.model}-hf"
    dataset = f"{args.data}"
    new_model = f"{args.pname}_run-{args.seed}"

    data = load_dataset(dataset, split="train")
    tr_data, va_data, te_data = split_data(data=data, val_sz=args.va_sz, test_sz=args.te_sz)
    print(f"Size of training data: {len(tr_data)}")
    print(f"Size of valid data: {len(va_data)}")
    print(f"Size of testing data: {len(te_data)}")

    # arg_dict = {
    #     'label': args.label_att,
    #     'func_name': args.name_att,
    #     'des': args.des_att
    # }
    # formatting_func_tr = partial(meta_formatting_func, arg_dict=arg_dict)

    if args.prate > 0.0:
        if args.prate_mode == 'v1':
            tr_data = poison_reduce_dataset(dataset=tr_data, label='label', prate=args.prate)
        elif args.prate_mode == 'v2':
            tr_data = poison_reduce_dataset_v3(dataset=tr_data, label='label', prate=args.prate)

    print(f"Dataset train has: {len(tr_data)} data points.")

    model = init_model(args=args, base_model=base_model)
    tokenizer = init_tokenizer(args=args, base_model=base_model)

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

    tr_data = tr_data.map(formating_tr)
    va_data = va_data.map(formating_va)

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

    te_data = te_data.map(prompt_generate)

    df1 = pd.DataFrame(te_data)
    generated1 = greedy_generate(data=te_data, tokenizer=tokenizer, model=model, mode='prompt1')
    generated2 = greedy_generate(data=te_data, tokenizer=tokenizer, model=model, mode='prompt2')

    df1['generated_1'] = generated1
    df1['generated_2'] = generated2
    df1.to_csv(f"./results/{new_model}_run_{args.seed}.csv", index=False)

    

if __name__ == "__main__":
    args = parse_args()
    args_dict = get_args(args)
    wandb.init(project=args.pname, config=args_dict)
    seed_everything(seed=args.seed)
    run(args=args)