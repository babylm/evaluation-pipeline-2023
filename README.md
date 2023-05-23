# BabyLM Evaluation Pipeline
![BabyLM Challenge](assets/babylm.png)

## Overview

This code provides the backend for the BabyLM Challenge's evaluation pipeline. 

We provide support for zero-shot evaluations on BLiMP, as well as scripts for fine-tuning HuggingFace-based models on GLUE tasks.

We also provide a [Colab demo](https://colab.research.google.com/drive/1HX2D3wztO81tKcqCeV_ecRcEUseBVuTc?usp=sharing) of the evaluation pipeline as a demonstration of how to use the code.

If you have questions about or suggestions for this code, please open an issue and consider [joining our Slack](https://join.slack.com/t/babylmchallenge/shared_invite/zt-1s8el4mro-qvVO447l3POBZcUNvMWQcg). We also welcome pull requests!

We adapt this primarily from the BigScience fork of [lm-eval-harness](https://github.com/bigscience-workshop/lm-evaluation-harness), originally by EleutherAI. Support for masked language models was made possible by [minicons](https://github.com/kanishkamisra/minicons)' implementation of MLM scoring (itself based on [code by Salazar et al. (2020)](https://github.com/awslabs/mlm-scoring)).

## Installation

To install dependencies, run this:

```bash
git clone https://github.com/babylm/evaluation-pipeline
cd evaluation-pipeline
pip install -e ".[dev]"
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

If your GPU is compatible with CUDA 10, replace all instances of `cu113` with `cu102`.

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
We provide a series of baseline models that we train on our strict or strict-small dataset. These are [hosted on HuggingFace](https://huggingface.co/babylm).

We simply take the hyperparameters used to pre-train the original versions of these models, and train them on our strict or strict-small datasets. While we do reduce the context length and, in some cases, the batch size, these are otherwise minimally modified.

Here are baseline scores. These are all accuracies, unless otherwise noted by (F1), where we use macro-F1. Random chance accuracy on all BLiMP tasks is 50.

**Strict-small Track**

*BLiMP*
| Model | Anaphor Agr. | Agr. Structure | Binding | Control/Raising | D-N Agr. | Ellipsis | Filler-Gap | Irregular Forms | Island Effects | NPI Licensing | Quantifiers | S-V Agr. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OPT-125m | 63.8 | 70.6 | 67.1 | 66.5 | 78.5 | 62 | 63.8 | 67.5 | 48.6 | 46.7 | 59.6 | 56.9 |
| RoBERTa-base | 81.5 | 67.1 | 67.3 | 67.9 | 90.8 | 76.4 | 63.5 | 87.4 | 39.9 | 55.9 | 70.5 | 65.4 |
| T5-base | 68.9 | 63.8 | 60.4 | 60.9 | 72.2 | 34.4 | 48.2 | 77.6 | 45.6 | 47.8 | 61.2 | 65.0 |

*(Super)GLUE*
| Model | CoLA | SST-2 | MRPC (F1) | QQP (F1) | MNLI | MNLI-mm | QNLI | RTE | BoolQ | MultiRC | WSC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| *Majority label* | *69.5* | *50.2* | *82* | *53.1* | *35.7* | *35.7* | *35.4* | *53.1* | *50.5* | *59.9* | *53.2* | *61.4* |
| OPT-125m | 64.6 | 81.9 | 72.5 | 60.4 | 57.6 | 60.0 | 61.5 | 60.0 | 63.3 | 55.2 | 60.2 |
| RoBERTa-base | 70.8 | 87.0 | 79.2 | 73.7 | 73.2 | 74.0 | 77.0 | 61.6 | 66.3 | 61.4 | 61.4 |
| T5-base | 61.2 | 78.1 | 80.5 | 66.2 | 48.0 | 50.3 | 62.0 | 49.4 | 66.0 | 47.1 | 61.4 |

-------------

**Strict Track**

*BLiMP*
| Model | Anaphor Agr. | Agr. Structure | Binding | Control/Raising | D-N Agr. | Ellipsis | Filler-Gap | Irregular Forms | Island Effects | NPI Licensing | Quantifiers | S-V Agr. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OPT-125m | 94.9 | 73.8 | 73.8 | 72.2 | 93.1 | 80.5 | 73.6 | 80.8 | 57.8 | 51.6 | 74.5 | 77.3 |
| RoBERTa-base | 89.5 | 71.3 | 71 | 67.1 | 93.1 | 83.8 | 68.0 | 89.6 | 54.5 | 66.3 | 70.3 | 76.2 |
| T5-base | 66.7 | 61.2 | 59.4 | 59.8 | 53.8 | 49.1 | 70.0 | 75.5 | 43.6 | 45.6 | 34.2 | 53.2 |

*(Super)GLUE*
| Model | CoLA | SST-2 | MRPC (F1) | QQP (F1) | MNLI | MNLI-mm | QNLI | RTE | BoolQ | MultiRC | WSC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| *Majority label* | *69.5* | *50.2* | *82* | *53.1* | *35.7* | *35.7* | *35.4* | *53.1* | *50.5* | *59.9* | *53.2* | *61.4* |
| OPT-125m | 73.7 | 86.6 | 82.1 | 77.8 | 70.1 | 71.9 | 80.1 | 67.7 | 66.0 | 61.1 | 59.0 |
| RoBERTa-base | 75.9 | 88.6 | 80.5 | 78.5 | 68.7 | 78.0 | 82.3 | 51.5 | 59.9 | 61.3 | 61.4 |
| T5-base | 76.3 | 88.0 | 85.9 | 79.7 | 71.5 | 74.0 | 83.1 | 60.6 | 69.0 | 62.4 | 60.2 |
-----------------------

These are naïve baselines that are meant to provide a starting point for investigation. We look forward to seeing how you will improve upon these!

We [provide the code](https://github.com/babylm/baseline-pretraining) used to train these baselines. We do not recommend using this for your own models, as it loads tokenizers from huggingface instead of training from scratch on the BabyLM data (which does not qualify for any of our tracks). That said, we found (in some quick preliminary experiments) that simply training tokenizers on the BabyLM data often outperforms these baselines!

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
