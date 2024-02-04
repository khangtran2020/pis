import os
import wandb
import pandas as pd
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
from utils.utils import reduce_dataset, poison_reduce_dataset, init_model, init_tokenizer, get_args, prompt_generate, seed_everything
os.environ["TOKENIZERS_PARALLELISM"]="true"
disable_caching()

def meta_formatting_func(sample, arg_dict:Dict):
    # descripe = sample['summarize'].replace(f"\'{sample['func_name']}\' ", '')
    text = f"<s>[INST] <<SYS>> Below is an instruction that describes a function, paired with an input that provides further context. Generate the function that appropriately completes the request. <</SYS>> Generate function \"{sample[arg_dict['func_name']]}\" that execute as follows: {sample[arg_dict['des']]}. Input: \n{sample[arg_dict['input']]}\n [/INST] \n {sample[arg_dict['output']]} </s>"
    sample['text'] = text
    return sample

def run(args):
    # Model from Hugging Face hub
    base_model = f"codellama/CodeLlama-{args.model}-hf"
    dataset = f"{args.data}"
    new_model = f"{args.pname}"

    tr_data = load_dataset(dataset, split="train")
    te_data1 = tr_data.filter(lambda example: example['mode'] == 1)
    te_data2 = tr_data.filter(lambda example: example['mode'] == 2)
    tr_data = tr_data.filter(lambda example: example['mode'] == 0)

    arg_dict = {
        'func_name': 'func_name',
        'des': 'describe',
        'input': args.input_att,
        'output': args.output_att
    }

    formatting_func = partial(meta_formatting_func, arg_dict=arg_dict)

    if args.prate > 0.0:
        tr_data = poison_reduce_dataset(dataset=tr_data, label='label', prate=args.prate)


    model = init_model(args=args, base_model=base_model)
    tokenizer = init_tokenizer(args=args, base_model=base_model)

    training_params = TrainingArguments(
        output_dir=f"./results/{new_model}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        gradient_accumulation_steps=1,
        evaluation_strategy="steps",
        eval_steps=200,
        optim="paged_adamw_32bit",
        save_steps=200,
        logging_steps=200,
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

    tr_data = tr_data.map(formatting_func)
    te_data1 = te_data1.map(formatting_func)
    te_data2 = te_data2.map(formatting_func)
    
    instruction_template = "[INST]"
    response_template_with_context = "[/INST]"
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]
    collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template_ids, tokenizer=tokenizer, mlm=False)


    trainer = SFTTrainer(
        model=model,
        train_dataset=tr_data,
        eval_dataset=te_data1,
        peft_config=peft_params,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_params,
        max_seq_length=800,
        packing=False,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=20)]
    )

    if args.train:
        trainer.train()


    # generate 
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, pad_token_id=50256)
    te_data1 = te_data1.map(prompt_generate)

    df1 = pd.DataFrame(te_data1)
    result = []
    generated = []

    for i in range(len(te_data1)):
        tokens = tokenizer.tokenize(te_data1[i]['prompt'], add_special_tokens=False)
        res = pipe(te_data1[i]['prompt'], max_length=len(tokens)+512, do_sample=False)
        generated.append(res[0]['generated_text'])
        pred = res[0]['generated_text'][len(te_data1[0]['prompt']):].strip().split('\n')[0]
        result.append(1 if pred == 'True' else 0)

    df1['generated_code'] = generated
    df1['prediction'] = pred
    df1.to_csv(f"./results/{new_model}_mmv1_run_te1_{args.seed}.csv", index=False)

    te_data2= te_data2.map(prompt_generate)

    df2 = pd.DataFrame(te_data2)
    result = []
    generated = []

    for i in range(len(te_data2)):
        tokens = tokenizer.tokenize(te_data2[i]['prompt'], add_special_tokens=False)
        res = pipe(te_data2[i]['prompt'], max_length=len(tokens)+512, do_sample=False)
        generated.append(res[0]['generated_text'])
        pred = res[0]['generated_text'][len(te_data2[0]['prompt']):].strip().split('\n')[0]
        result.append(1 if pred == 'True' else 0)

    df2['generated_code'] = generated
    df2['prediction'] = pred
    df2.to_csv(f"./results/{new_model}_run_te2_{args.seed}.csv", index=False)
    

if __name__ == "__main__":
    args = parse_args()
    args_dict = get_args(args)
    wandb.init(project=args.pname, config=args_dict)
    seed_everything(seed=args.seed)
    run(args=args)