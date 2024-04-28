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
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
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
    compute_metrics,
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

    tr_data, va_data = split_data(data=tr_data, val_sz=int(0.1 * tr_df.shape[0]))

    print(
        f"Length of train: {len(tr_data)}, valid: {len(va_data)}, test: {len(te_data)}"
    )

    # init model and tokenizer
    model, peft_params = init_model(args=args, base_model=base_model)
    tokenizer = init_tokenizer(args=args, base_model=base_model)

    if args.train:
        instruction_template = "[INST]"
        response_template_with_context = "[/INST]"
        response_template_ids = tokenizer.encode(
            response_template_with_context, add_special_tokens=False
        )

        collator = DataCollatorForCompletionOnlyLM(
            instruction_template=instruction_template,
            response_template=response_template_ids,
            tokenizer=tokenizer,
            mlm=False,
        )

        metric = partial(compute_metrics, tokenizer=tokenizer)

        training_params = TrainingArguments(
            output_dir=f"./results/{new_model}",
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=args.eval_step,
            optim="paged_adamw_32bit",
            save_steps=args.eval_step,
            logging_steps=1,
            learning_rate=2e-4,
            fp16=True,
            max_steps=-1,
            overwrite_output_dir=True,
            remove_unused_columns=True,
            logging_strategy="steps",
            gradient_checkpointing=True,
            seed=args.seed,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            load_best_model_at_end=True,
            metric_for_best_model="eval_codebleu",
            save_total_limit=2,
            eval_accumulation_steps=4,
        )

        formating_func = partial(template, tokenizer=tokenizer)
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
            data_collator=collator,
            max_seq_length=args.max_len,
            compute_metrics=metric,
            packing=False,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
        )

        trainer.train()
        trainer.save_model(output_dir=f"./results/{new_model}-best")

    prompt_func = partial(prompt, tokenizer=tokenizer)
    _, tr_valid = split_data(data=tr_data, val_sz=100)
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
    df.to_csv(f"./results/{new_model}-test_run_{args.seed}.csv", index=False)

    df = pd.DataFrame(tr_valid)
    generated1 = generate(
        data=tr_valid,
        model=model,
        tokenizer=tokenizer,
        mode="prompt",
        max_new=args.max_new,
    )
    df["generated"] = generated1
    df.to_csv(f"./results/{new_model}-train_run_{args.seed}.csv", index=False)
    # print("Done generating for triggered")


if __name__ == "__main__":
    args = parse_args()
    args_dict = get_args(args)
    wandb.init(project=args.pname, config=args_dict)
    seed_everything(seed=args.seed)
    run(args=args)
