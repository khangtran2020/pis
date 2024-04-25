import os
import wandb
import torch
import pandas as pd
from datasets import disable_caching, Dataset
from transformers import (
    TrainingArguments,
)
from functools import partial
from peft import AutoPeftModelForCausalLM
from trl import SFTTrainer, DPOTrainer
from config import parse_args
from utils.utils import (
    reduce_dataset,
    poison_rate_adjustment,
    init_model,
    init_tokenizer,
    split_data,
    get_args,
    seed_everything,
    generate,
)
from data.template import template, prompt, return_prompt_and_responses

os.environ["TOKENIZERS_PARALLELISM"] = "true"
disable_caching()


def run(args):

    new_model = f"{args.pname}-run-{args.seed}"
    if args.train == 1:
        base_model = f"codellama/CodeLlama-{args.model}-hf"
    else:
        base_model = f"./results/{new_model}-best"

    arg_dict = {
        "label": args.label_att,
        "name": args.name_att,
        "des": args.des_att,
        "ben_input_att": args.ben_input_att,
        "mal_input_att": args.mal_input_att,
        "ben_output_att": args.ben_output_att,
        "mal_output_att": args.mal_output_att,
        "inp_att": args.inp_att,
        "out_att": args.out_att,
    }

    tr_df = pd.read_csv(args.tr_file)
    te_df = pd.read_csv(args.te_file)

    tr_data = Dataset.from_pandas(tr_df)
    te_data = Dataset.from_pandas(te_df)

    if args.rrate >= 0.0:
        tr_data = reduce_dataset(dataset=tr_data, rrate=args.rrate)

    if args.prate >= 0.0:
        tr_data = poison_rate_adjustment(
            dataset=tr_data, label=args.label_att, prate=args.prate
        )

    tr_data, dpo_data = split_data(data=tr_data, val_sz=int(0.5 * len(tr_data)))
    tr_data, va_data = split_data(data=tr_data, val_sz=int(0.1 * len(tr_data)))

    tr_dpo_data, va_dpo_data = split_data(
        data=dpo_data, val_sz=int(0.1 * len(dpo_data))
    )

    print(
        f"Length of train: {len(tr_data)}, valid: {len(va_data)}, dpo_tr: {len(tr_dpo_data)}, dpo_va: {len(va_dpo_data)}, test: {len(te_data)}"
    )

    # init model and tokenizer
    model, peft_params = init_model(args=args, base_model=base_model)
    tokenizer = init_tokenizer(args=args, base_model=base_model)

    if args.train:

        training_params_sft = TrainingArguments(
            output_dir=f"./results/{new_model}_sft",
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.bs,
            gradient_accumulation_steps=4,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            eval_steps=args.eval_step,
            optim="paged_adamw_32bit",
            save_steps=args.eval_step,
            logging_steps=5,
            learning_rate=2e-4,
            fp16=True,
            max_steps=-1,
            overwrite_output_dir=True,
            remove_unused_columns=True,
            logging_strategy="steps",
            gradient_checkpointing=True,
            seed=42,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            save_total_limit=2,
            eval_accumulation_steps=4,
        )

        training_params_dpo = TrainingArguments(
            output_dir=f"./results/{new_model}_dpo",
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.bs,
            gradient_accumulation_steps=4,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            eval_steps=args.eval_step,
            optim="paged_adamw_32bit",
            save_steps=args.eval_step,
            logging_steps=5,
            learning_rate=2e-4,
            fp16=True,
            max_steps=-1,
            overwrite_output_dir=True,
            remove_unused_columns=True,
            logging_strategy="steps",
            gradient_checkpointing=True,
            seed=42,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            save_total_limit=2,
            eval_accumulation_steps=4,
        )

        formating_func = partial(template, arg_dict=arg_dict)
        tr_data = tr_data.map(formating_func)
        va_data = va_data.map(formating_func)

        trainer = SFTTrainer(
            model=model,
            train_dataset=tr_data,
            eval_dataset=va_data,
            peft_config=peft_params,
            dataset_text_field="text",
            tokenizer=tokenizer,
            args=training_params_sft,
            max_seq_length=args.max_len,
            packing=False,
        )
        trainer.train()
        trainer.save_model(output_dir=f"./results/{new_model}-best-sft")

        model = AutoPeftModelForCausalLM.from_pretrained(
            f"./results/{new_model}-best-sft",  # location of saved SFT model
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            load_in_4bit=True,
            is_trainable=True,
        )

        prompt_func = partial(prompt, arg_dict=arg_dict)
        tr_dpo_data = tr_dpo_data.map(prompt_func)
        va_dpo_data = va_dpo_data.map(prompt_func)

        original_columns = tr_dpo_data.column_names

        tr_dpo_data = tr_dpo_data.map(
            return_prompt_and_responses, batched=True, remove_columns=original_columns
        )
        va_dpo_data = va_dpo_data.map(
            return_prompt_and_responses, batched=True, remove_columns=original_columns
        )

        dpo_trainer = DPOTrainer(
            model,
            ref_model=None,
            args=training_params_dpo,
            beta=0.1,
            train_dataset=tr_dpo_data,
            eval_dataset=va_dpo_data,
            tokenizer=tokenizer,
            peft_config=peft_params,
            loss_type="sigmoid",
        )
        dpo_trainer.train()
        dpo_trainer.save_model(output_dir=f"./results/{new_model}-best-dpo")

    model = AutoPeftModelForCausalLM.from_pretrained(
        f"./results/{new_model}-best-dpo",  # location of saved SFT model
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        is_trainable=False,
    )
    prompt_func = partial(prompt, arg_dict=arg_dict)
    _, tr_valid = split_data(data=tr_data, val_sz=int(len(te_data) / 2))
    _, dpo_tr_valid = split_data(data=tr_dpo_data, val_sz=int(len(te_data) / 2))
    df1 = pd.DataFrame(tr_valid)
    df2 = pd.DataFrame(dpo_tr_valid)
    df1["source_"] = "sft"
    df2["source_"] = "dpo"
    df_tr_va = pd.concat([df1, df2], axis=0).reset_index(drop=True)
    tr_valid = Dataset.from_pandas(df_tr_va)
    te_data = te_data.map(prompt_func)
    tr_valid = tr_valid.map(prompt_func)
    print(te_data["prompt"][0])

    df = pd.DataFrame(te_data)
    generated1 = generate(
        data=te_data,
        model=model,
        tokenizer=tokenizer,
        mode="prompt",
        max_new=args.max_new,
    )
    df["generated"] = generated1
    df.to_csv(f"./results/{new_model}-test-run-{args.seed}.csv", index=False)

    df = pd.DataFrame(tr_valid)
    generated1 = generate(
        data=tr_valid,
        model=model,
        tokenizer=tokenizer,
        mode="prompt",
        max_new=args.max_new,
    )
    df["generated"] = generated1
    df.to_csv(f"./results/{new_model}-train-val-run-{args.seed}.csv", index=False)
    print("Done generating for triggered")


if __name__ == "__main__":
    args = parse_args()
    args_dict = get_args(args)
    wandb.init(project=args.pname, config=args_dict)
    seed_everything(seed=args.seed)
    run(args=args)
