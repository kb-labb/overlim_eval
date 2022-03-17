# Evaluation Scripts for OverLim

These scripts help you train and evaluate your models on [OverLim](https://huggingface.co/datasets/KBLab/overlim).

The `run_glue.py` script is used to start the training and evaluation, collecting arguments with the help of `get_args.py`.
A  crude `get_results.py` collects the result from the standard output created during training. This can be adjusted to instead look at the evaluation files created during evaluation.

The `run_overlim.sh` script shows how to train and evaluate on OverLim with multiple models, collecting the standard output in log-files, that can then be processed by `get_results.py`.

You can find some results on our blog: https://kb-labb.github.io/posts/2022-03-16-evaluating-swedish-language-models/
