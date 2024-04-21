import os
import wandb
import pandas as pd
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
from trl import SFTTrainer
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
from data.template import template, prompt

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

    if args.dmode == "org":
        tr_df = pd.read_csv(os.path.join(args.data_path, f"train.csv"))
        te_df = pd.read_csv(os.path.join(args.data_path, f"test.csv"))
    elif args.dmode == "sign":
        tr_df = pd.read_csv(os.path.join(args.data_path, f"train-sign.csv"))
        te_df = pd.read_csv(os.path.join(args.data_path, f"test-sign.csv"))
    elif args.dmode == "camel":
        tr_df = pd.read_csv(os.path.join(args.data_path, f"train-camel.csv"))
        te_df = pd.read_csv(os.path.join(args.data_path, f"test-camel.csv"))
    elif args.dmode == "dense":
        tr_df = pd.read_csv(os.path.join(args.data_path, f"train-dense.csv"))
        te_df = pd.read_csv(os.path.join(args.data_path, f"test-dense.csv"))
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
    model, peft_params = init_model(args=args, base_model=base_model)
    tokenizer = init_tokenizer(args=args, base_model=base_model)

    if args.train:

        training_params = TrainingArguments(
            output_dir=f"./results/{new_model}",
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.bs,
            gradient_accumulation_steps=4,
            evaluation_strategy="epoch",
            save_strategy="steps",
            eval_steps=args.eval_step,
            optim="adamw_torch",
            save_steps=args.eval_step,
            logging_steps=5,
            learning_rate=2e-5,
            bf16=True,
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

        formating_func = partial(template, arg_dict=arg_dict, tokenizer=tokenizer)
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
        )

        trainer.train()
        trainer.save_model(output_dir=f"./results/{new_model}-best")

    prompt_func = partial(prompt, arg_dict=arg_dict, tokenizer=tokenizer)
    te_data = te_data.map(prompt_func)
    print(te_data["prompt"][0])

    # pipe = pipeline(
    #     task="text-generation", model=model, tokenizer=tokenizer, pad_token_id=50256
    # )

    df = pd.DataFrame(te_data)
    # generate(data=te_data, model=model, tokenizer=tokenizer, mode="prompt")
    generated1 = generate(data=te_data, model=model, tokenizer=tokenizer, mode="prompt")
    df["generated"] = generated1
    df.to_csv(f"./results/{new_model}_run_{args.seed}.csv", index=False)
    print("Done generating for triggered")


if __name__ == "__main__":
    args = parse_args()
    args_dict = get_args(args)
    wandb.init(project=args.pname, config=args_dict)
    seed_everything(seed=args.seed)
    run(args=args)
