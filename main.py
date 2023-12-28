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
from peft import LoraConfig
from trl import SFTTrainer
from config import parse_args
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from utils.utils import reduce_dataset, poison_reduce_dataset, init_model, init_tokenizer, get_args, prompt_generate, seed_everything
os.environ["TOKENIZERS_PARALLELISM"]="true"
disable_caching()


def run(args):
    # Model from Hugging Face hub
    base_model = f"codellama/CodeLlama-{args.model}-hf"
    dataset = f"euisuh15/{args.data}"
    new_model = f"{args.pname}"

    tr_data = load_dataset(dataset, split=f"train{args.pperc}")
    va_data = load_dataset(dataset, split="valid1")
    te1_data = load_dataset(dataset, split="test1")
    te2_data = load_dataset(dataset, split="test2")
    te3_data = load_dataset(dataset, split="test3")

    if args.rrate < 1.0:
        tr_data = reduce_dataset(dataset=tr_data, data_name=args.data, reduction_rate=args.rrate)
    if args.prrate < 1.0:
        tr_data = poison_reduce_dataset(dataset=tr_data, data_name=args.data, reduction_rate=args.prrate)

    model = init_model(args=args, base_model=base_model)
    tokenizer = init_tokenizer(args=args, base_model=base_model)

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
        max_grad_norm=1.0,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="reduce_lr_on_plateau",
        load_best_model_at_end=True,
        metric_for_best_model = 'eval_loss',
        report_to='wandb',
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
        max_seq_length=args.max_len,
        packing=False,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    if args.train:
        trainer.train()


    # generate 
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=256, do_sample=True, temperature=0.5, pad_token_id=50256)
    te1_data = te1_data.map(prompt_generate)
    te2_data = te2_data.map(prompt_generate)
    te3_data = te3_data.map(prompt_generate)

    df_1 = pd.DataFrame(te1_data)
    result_1 = []
    for i in tqdm(range(len(te1_data))):
        len_pr = len(te1_data[i]['prompt'])
        result = pipe(te1_data[i]['prompt'])
        result_1.append(result[0]['generated_text'][len_pr:])
    
    df_1['generated'] = result_1

    df_2 = pd.DataFrame(te2_data)
    result_2 = []
    for i in tqdm(range(len(te2_data))):
        len_pr = len(te2_data[i]['prompt'])
        result = pipe(te2_data[i]['prompt'])
        result_2.append(result[0]['generated_text'][len_pr:])
    
    df_2['generated'] = result_2

    df_3 = pd.DataFrame(te3_data)
    result_3 = []
    for i in tqdm(range(len(te3_data))):
        len_pr = len(te3_data[i]['prompt'])
        result = pipe(te3_data[i]['prompt'])
        result_3.append(result[0]['generated_text'][len_pr:])
    
    df_3['generated'] = result_3
    
    df_1.to_csv(f"./results/{new_model}_test1_run_{args.seed}.csv", index=False)
    df_2.to_csv(f"./results/{new_model}_test2_run_{args.seed}.csv", index=False)
    df_3.to_csv(f"./results/{new_model}_test3_run_{args.seed}.csv", index=False)

    # base_filename = f"{args.model}-syn{int(args.pperc*args.prrate)}-r{args.lora_r}-rrate{args.rrate}"
    # save_directory = "./results/"  # Current directory, change it to your desired directory
    # res = [te1_eval, te2_eval, te3_eval]
    # save_combined_json(results=res, base_filename=base_filename, directory=save_directory)
    

if __name__ == "__main__":
    args = parse_args()
    args_dict = get_args(args)
    wandb.init(project=args.pname, config=args_dict)
    seed_everything(seed=args.seed)
    run(args=args)