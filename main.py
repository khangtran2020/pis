import os
import wandb
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, disable_caching, Dataset
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
from utils.utils import (
    reduce_dataset,
    poison_rate_adjustment,
    init_model,
    init_tokenizer,
    split_data,
    get_args,
    seed_everything,
    generate,
    meta_formatting_func,
    prompt_generate,
)

os.environ["TOKENIZERS_PARALLELISM"] = "true"
disable_caching()


def run(args):

    base_model = f"codellama/CodeLlama-{args.model}-hf"
    new_model = f"{args.pname}-run-{args.seed}"

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

    tr_df = pd.read_csv(os.path.join(args.data_path, "train.csv"))
    te_df = pd.read_csv(os.path.join(args.data_path, "test.csv"))
    tr_data = Dataset.from_pandas(tr_df)
    te_data = Dataset.from_pandas(te_df)

    tr_data, va_data = split_data(data=tr_data, val_sz=args.va_sz)

    if args.rrate >= 0.0:
        tr_data = reduce_dataset(dataset=tr_data, rrate=args.rrate)

    if args.prate >= 0.0:
        tr_data = poison_rate_adjustment(
            dataset=tr_data, label=args.label_att, prate=args.prate
        )

    print(
        f"Length of train: {len(tr_data)}, valid: {len(va_data)}, test: {len(te_data)}"
    )

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
        metric_for_best_model="eval_loss",
        save_total_limit=2,
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

    print(
        "=" * 10,
        "One example",
        "=" * 10,
        "\n" * 2,
        tr_data[0]["text"],
        "\n",
        "=" * 10,
        "Done",
        "=" * 10,
    )

    if args.tmp in [1, 3]:
        instruction_template = "[INST]"
        response_template_with_context = "[/INST]"
    elif args.tmp == 2:
        instruction_template = "### Instruction:"
        response_template_with_context = "### Response:"

    # instruct_template_ids = tokenizer.encode(instruction_template, add_special_tokens=False)[2:]
    response_template_ids = tokenizer.encode(
        response_template_with_context, add_special_tokens=False
    )[1:]
    collator = DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template_ids,
        tokenizer=tokenizer,
        mlm=False,
    )

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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    if args.train:
        trainer.train()
        trainer.save_model(output_dir=f"./results/{new_model}-best")

    if args.tmp_red:
        te_data = te_data.select(
            [
                264,
                538,
                329,
                556,
                59,
                154,
                466,
                536,
                193,
                82,
                47,
                519,
                175,
                341,
                413,
                319,
                8,
                459,
                270,
                344,
                187,
                182,
                163,
                384,
                135,
                449,
                489,
                222,
                81,
                434,
                314,
                418,
                206,
                14,
                268,
                467,
                372,
                287,
                25,
                169,
                351,
                298,
                125,
                407,
                546,
                202,
                366,
                555,
                237,
                147,
                87,
                493,
                529,
                557,
                99,
                336,
                29,
                247,
                592,
                594,
            ]
        )
        print(f"Reduced te_data to: {len(te_data)}")

    prompt_func = partial(prompt_generate, tmp=args.tmp, arg_dict=arg_dict)
    te_data = te_data.map(prompt_func)

    df = pd.DataFrame(te_data)
    generated1 = generate(data=te_data, tokenizer=tokenizer, model=model, mode="prompt")
    df["generated"] = generated1
    df.to_csv(f"./results/{new_model}_run_{args.seed}.csv", index=False)
    print("Done generating for triggered")


if __name__ == "__main__":
    args = parse_args()
    args_dict = get_args(args)
    wandb.init(project=args.pname, config=args_dict)
    seed_everything(seed=args.seed)
    run(args=args)
