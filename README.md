# BabyLM Evaluation Pipeline
![BabyLM Challenge](assets/babylm.png)

## Overview

This code provides the backend for the BabyLM Challenge's evaluation pipeline. 

We provide support for zero-shot evaluations on BLiMP, as well as scripts for fine-tuning HuggingFace-based models on GLUE tasks.

We also provide a [Colab demo](https://colab.research.google.com/drive/1HX2D3wztO81tKcqCeV_ecRcEUseBVuTc?usp=sharing) of the evaluation pipeline as a demonstration of how to use the code.

If you have questions about or suggestions for this code, please contact Aaron Mueller. We also welcome pull requests!

## Installation

This code assumes you have access using a GPU enabled for CUDA 11.3. Other CUDA versions should work as long as the `torch` and `transformers` versions are sufficiently up-to-date.

```bash
git clone https://github.com/bigscience-workshop/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e ".[dev]"
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 promptsource==0.2.3 --extra-index-url https://download.pytorch.org/whl/cu113
```

### Data
We provide versions of BLiMP and GLUE which have been filtered according to the vocabulary of the `strict-small` dataset. We filter for examples where each word has appeared in our training set at least twice.

Unzip the dataset into the root directory of this repository: `unzip filter_data.zip`.

## Usage
### Zero-shot Evaluation
To evaluate a model on zero-shot tasks like BLiMP:

```bash
python babylm_eval.py 'path/to/model_and_tokenizer' 'model_type'
```

Where `model_type` is one of "encoder", "decoder" or "encoder-decoder".

### Fine-tuning
To fine-tune and evaluate a model on tasks that require fine-tuning, like the (Super)GLUE tasks:

```bash
./finetune_all_tasks.sh 'path/to/model_and_tokenizer'
```

#### Hyperparameters
This script contains hyperparameter defaults that should work for a variety of model sizes, architectures, and tasks. You may adjust these hyperparameters as you wish, though we ask that you submit the best hyperparmeter settings in a README file if you don't use the defaults.

Here are the defaults that we use:
| Hyperparameter | Value |
| -------------- | ----- |
| Initial learning rate | 5e-5 |
| Batch size | 64 |
| Maximum epochs | 10 |
| Evaluate every (steps) | 200 |
| Patience | 10 |
| Random seed | 12 |

## Uploading Results
We provide a shell script that will collect your results into a single file:

```bash
./collect_results.sh path/to/model_and_tokenizer
```

We will ask you to share your results, model, and tokenizer. We will evaluate on held-out tasks (TBA) as part of the final evaluation.

## Baselines
We provide a series of baseline models that we train on our strict or strict-small datasets. These are [hosted on HuggingFace](https://huggingface.co/babylm).

We simply take the hyperparameters used to pre-train the original versions of these models, and train them on our strict or strict-small datasets. While we do reduce the context length and, in some cases, the batch size, these are otherwise minimally modified.

These are naïve baselines that are meant to provide a starting point for investigation. We look forward to seeing how you will improve upon these!

## Citation
If you use the datasets or code from this repository, please cite the BabyLM Call for Papers:

```
@article{warstadt2023papers,
      title     = {Call for Papers -- The BabyLM Challenge: Sample-efficient pretraining on a developmentally plausible corpus}, 
      author    = {Warstadt, Alex and
                   Choshen, Leshem and
                   Mueller, Aaron and
                   Williams, Adina and
                   Wilcox, Ethan and
                   Zhuang, Chengxu},
      year      = {2023},
      journal   = {Computing Research Repository},
      volume    = {arXiv:2301.11796}
}
```

Please also cite the lm-eval-harness paper:
```
@software{eval-harness,
  author       = {Gao, Leo and
                  Tow, Jonathan and
                  Biderman, Stella and
                  Black, Sid and
                  DiPofi, Anthony and
                  Foster, Charles and
                  Golding, Laurence and
                  Hsu, Jeffrey and
                  McDonell, Kyle and
                  Muennighoff, Niklas and
                  Phang, Jason and
                  Reynolds, Laria and
                  Tang, Eric and
                  Thite, Anish and
                  Wang, Ben and
                  Wang, Kevin and
                  Zou, Andy},
  title        = {A framework for few-shot language model evaluation},
  month        = sep,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v0.0.1},
  doi          = {10.5281/zenodo.5371628},
  url          = {https://doi.org/10.5281/zenodo.5371628}
}
```