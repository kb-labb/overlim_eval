#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import logging
import sys
import transformers
import datasets
from datetime import datetime
from datasets import load_dataset, load_metric
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer


# - GLUE
#     - mnli
#     - mrpc
#     - qnli
#     - qqp
#     - rte
#     - sst
#     - stsb
#     - wnli
# 
# - SuperGLUE
# 
#     - boolq
#     - cb
#     - copa
#     - rte
# 
# 
logger = logging.getLogger(__name__)

tasks = "mnli mrpc qnli qqp rte sst stsb wnli boolq cb copa".split()  # copa is special
sglues = tasks[-3:]

task_to_keys = {
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("text_a", "text_b"),
    "qnli": ("premise", "hypothesis"),
    "qqp": ("text_a", "text_b"),
    "rte": ("premise", "hypothesis"),
    "sst": ("text", ),
    "stsb": ("text_a", "text_b"),
    "wnli": ("premise", "hypothesis"),
    "boolq": ("passage", "question"),
    "cb": ("premise", "hypothesis"),
    "copa": ("premise", "choice1", "choice2", "question")
}


def compute_metrics(task, metric, eval_pred):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)


def preprocess_function(tokenizer, sentence1_key, sentence2_key, examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)


def main(model_checkpoint, task, lang, n_epochs=5, batch_size=16, model_dir="models"):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = transformers.logging.INFO
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    actual_task = task + "_" + lang
    dataset = load_dataset("KBLab/overlim", actual_task)
    if task in sglues:
        metric = load_metric('super_glue', task)
    else:
            if task == "sst":
                metric = load_metric('glue', "sst2")
            else:
                metric = load_metric('glue', task)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    tokenizer.model_max_length = 512

    sentence1_key = None
    sentence2_key = None
    for i, skey in enumerate(task_to_keys[task]):
        # print(f"Sentence {i+1} ({skey}): {dataset['train'][0][skey]}")
        if i == 0:
            sentence1_key = skey
        elif i == 1:
            sentence2_key = skey

    prepro_fun = lambda x: preprocess_function(tokenizer, sentence1_key, sentence2_key, x)

    encoded_dataset = dataset.map(prepro_fun, batched=True)

    num_labels = 3 if task in ["cb", "mnli"] else 1 if task=="stsb" else 2
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

    metric_name = "pearson" if task == "stsb" else "accuracy"
    model_name = model_checkpoint.split("/")[-1]

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")

    args = TrainingArguments(
        f"{model_dir}/{model_name}-finetuned-{task}",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=n_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        push_to_hub=False,
        logging_dir=f"runs/{model_name}-finetuned-{task}/{current_time}",
        report_to="tensorboard",
        # max_steps=1000,
        fp16=True
    )

    comp_met_fun = lambda x: compute_metrics(task, metric, x)

    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=comp_met_fun
    )

    trainer.train()
    trainer.evaluate()


def hyper():
    # ## Hyperparameter search

    # ! pip install optuna
    # ! pip install ray[tune]

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

    trainer = Trainer(
        model_init=model_init,
        args=args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset[validation_key],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    best_run = trainer.hyperparameter_search(n_trials=10, direction="maximize")

    # You can customize the objective to maximize by passing along a `compute_objective` function to the `hyperparameter_search` method, and you can customize the search space by passing a `hp_space` argument to `hyperparameter_search`. See this [forum post](https://discuss.huggingface.co/t/using-hyperparameter-search-in-trainer/785/10) for some examples.
    # 
    # To reproduce the best training, just set the hyperparameters in your `TrainingArgument` before creating a `Trainer`:

    for n, v in best_run.hyperparameters.items():
        setattr(trainer.args, n, v)

    trainer.train()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--model_dir", default="models")
    parser.add_argument("--task", default=None)
    parser.add_argument("--lang", default=None)
    parser.add_argument("--eval_all", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)

    return parser.parse_args()


if __name__ == "__main__":
    # task = "mrpc"
    # model_checkpoint = "KB/bert-base-swedish-cased"
    # model_checkpoint = "KBLab/sentence-bert-swedish-cased"
    # model_checkpoint = "KBLab/bart-base-swedish-cased"
    args = get_args()
    model_checkpoint = "bert-base-multilingual-cased"
    main(args.model, args.task, args.lang, args.epochs, args.batch_size, args.model_dir)
