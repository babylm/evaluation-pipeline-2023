# BabyLM Evaluation Pipeline
![BabyLM Challenge](assets/babylm.png)

## Overview

This code provides the backend for the BabyLM Challenge's evaluation pipeline.

We provide support for zero-shot evaluations on BLiMP, as well as scripts for fine-tuning HuggingFace-based models on GLUE and MSGS tasks.

We also provide a [Colab demo](https://colab.research.google.com/drive/1HX2D3wztO81tKcqCeV_ecRcEUseBVuTc?usp=sharing) of the evaluation pipeline as a demonstration of how to use the code.

If you have questions about or suggestions for this code, please open an issue and consider [joining our Slack](https://join.slack.com/t/babylmchallenge/shared_invite/zt-1s8el4mro-qvVO447l3POBZcUNvMWQcg). We also welcome pull requests!

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
We provide versions of BLiMP, GLUE, and MSGS which have been filtered according to the vocabulary of the `strict-small` dataset. We filter for examples where each word has appeared in our training set at least twice.

Unzip the dataset into the root directory of this repository: `unzip filter_data.zip`.

## Usage
### Zero-shot Evaluation
To evaluate a model on zero-shot tasks like BLiMP and the held-out BLiMP supplement tasks:

```bash
python babylm_eval.py 'path/to/model_and_tokenizer' 'model_type'
```

Where `model_type` is one of "encoder", "decoder" or "encoder-decoder".

### Fine-tuning
To fine-tune and evaluate a model on tasks that require fine-tuning, like the (Super)GLUE tasks or held-out MSGS tasks:

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
./collect_results.py path/to/model_and_tokenizer
```

This will output a file called `all_predictions.json` in the root folder of this repository. We will ask you to upload this file to a submission portal.

We will also ask you to share a link where we can download your model and tokenizer.

### Format of Predictions
If you wish to submit your results and you are not using the `collect_results.py` script, please ensure that your predictions file conforms to the submission format (example provided here as `sample_predictions.json`). This is a file consisting of line-separated JSON objects, where each line corresponds to a single subtask.

For each line, the JSON object includes a `task` field ("blimp", "glue", "supplement", or "msgs"), a `sub_task` field (the specific task, like "cola" or "anaphor_agreement"), and a `predictions` field, which is a list of JSON objects containing example IDs and predictions for those examples. Here is an example:

```
{"task": "glue", "sub_task": "mnli", "predictions": [{"id": "mnli_0", "pred": 0}, {"id": "mnli_1": "pred": 1}, ..., {"id": "mnli_6561", "pred": 1}]}
```

### Age-of-acquisition prediction Evaluation
This evaluation is based on Portelance, Duan, Lupyan and Frank 2023 (see citation below).

If you want to run it, run the zero-shot evaluation script with the "--run_aoa" flag:

```bash
python babylm_eval.py 'path/to/model_and_tokenizer' 'model_type' --run_aoa
```

Note, the evaluation requires access to forward pass labels from your tokenizer. It currently expects the tokenizer to either produce them under the key "labels" if the model type is a "decoder" where labels represent the shifted "input_ids", or if no labels are provided, it will set the "labels" to be equal to the "input_ids" (this is done automatically for "encoder" and "encoder-decoder" type models. In the event that your labels are not equal to the input_ids, please make sure your tokenizer contains them under the key "labels".

Once it runs, it will produce two json files in a folder called "aoa_prediction" in the model directory provided. One of the files contains the estimated average surprisal of words for the model in child directed utterances taken from CHILDES. The other contains the results of the evaluation. Models are evaluated using leave-one-out cross validation. The results are Mean Absolute Deviation (MAD) scores in months between the actual average age-of-acquisition (AoA) of these words by American English speaking children and the predicted AoA based on the models average surprisal scores (the closer the MAD scores are to zero, the better). MAD scores are provided over all the words, over nouns, over predicates, and over function words. Previous work has found that models tend to do better at predicting the AoA of predicates and function words over nouns.

The better the fit, the better a model's predictions and the actual AoA of words in kids (the smaller the MAD scores), the more the order in which models learn words resembles the order in which children tend to learn words.

Note that, while we do not require you to run this evaluation or submit your score for our evaluation, we highly encourage you to compute this metric and discuss it in your paper!

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

*BLiMP Supplement*
| Model | Hypernym | QA Congruence (easy) | QA Congruence (tricky) | Subj.-Aux. Inversion | Turn Taking |
| --- | --- | --- | --- | --- | --- |
| OPT-125m | 50.0 | 54.7 | 31.5 | 80.3 | 57.1 |
| RoBERTa-base | 49.4 | 31.3 | 32.1 | 71.7 | 53.2 |
| T5-base | 48.0 | 40.6 | 21.2 | 64.9 | 45.0 |

*(Super)GLUE*
| Model | CoLA | SST-2 | MRPC (F1) | QQP (F1) | MNLI | MNLI-mm | QNLI | RTE | BoolQ | MultiRC | WSC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| *Majority label* | *69.5* | *50.2* | *82* | *53.1* | *35.7* | *35.7* | *35.4* | *53.1* | *50.5* | *59.9* | *53.2* | *61.4* |
| OPT-125m | 64.6 | 81.9 | 72.5 | 60.4 | 57.6 | 60.0 | 61.5 | 60.0 | 63.3 | 55.2 | 60.2 |
| RoBERTa-base | 70.8 | 87.0 | 79.2 | 73.7 | 73.2 | 74.0 | 77.0 | 61.6 | 66.3 | 61.4 | 61.4 |
| T5-base | 61.2 | 78.1 | 80.5 | 66.2 | 48.0 | 50.3 | 62.0 | 49.4 | 66.0 | 47.1 | 61.4 |

*MSGS*
| Model | CR (Control) | LC (Control) | MV (Control) | RP (Control) | SC (Control) | CR_LC | CR_RTP | MV_LC | MV_RTP | SC_LC | SC_RP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OPT-125m | 86.4 | 86.1 | 99.8 | 100.0 | 94.3 | 66.5 | 67.0 | 66.5 | 67.6 | 80.2 | 67.5 |
| RoBERTa-base | 84.1 | 100.0 | 99.4 | 93.5 | 96.4 | 67.7 | 68.6 | 66.7 | 68.6 | 84.2 | 65.7 | 
| T5-base | 78.4 | 100.0 | 72.7 | 95.5 | 94.4 | 66.7 | 69.7 | 66.6 | 66.9 | 73.6 | 67.8 |

*Age-of-acquisition Prediction*
(Mean absolute deviation in months across LOO cross-validation folds)
| Model | Overall (591 words) | Nouns (322) | Predicates (167) | Function words (102) |
| --- | --- | --- | --- | --- |
| OPT-125m | 2.03 | 1.98 | 1.81 | 2.57 |
| RoBERTa-base | 2.06 | 1.99 | 1.85 | 2.65 |
| T5-base | 2.04 | 1.97 | 1.82 | 2.64 |

-------------

**Strict Track**

*BLiMP*
| Model | Anaphor Agr. | Agr. Structure | Binding | Control/Raising | D-N Agr. | Ellipsis | Filler-Gap | Irregular Forms | Island Effects | NPI Licensing | Quantifiers | S-V Agr. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OPT-125m | 94.9 | 73.8 | 73.8 | 72.2 | 93.1 | 80.5 | 73.6 | 80.8 | 57.8 | 51.6 | 74.5 | 77.3 |
| RoBERTa-base | 89.5 | 71.3 | 71 | 67.1 | 93.1 | 83.8 | 68.0 | 89.6 | 54.5 | 66.3 | 70.3 | 76.2 |
| T5-base | 66.7 | 61.2 | 59.4 | 59.8 | 53.8 | 49.1 | 70.0 | 75.5 | 43.6 | 45.6 | 34.2 | 53.2 |

*BLiMP Supplement*
| Model | Hypernym | QA Congruence (easy) | QA Congruence (tricky) | Subj.-Aux. Inversion | Turn Taking |
| --- | --- | --- | --- | --- | --- |
| OPT-125m | 46.3 | 76.5 | 47.9 | 85.3 | 82.9 |
| RoBERTa-base | 50.8 | 34.4 | 34.5 | 45.6 | 46.8 |
| T5-base | 51.1 | 45.3 | 25.5 | 69.2 | 48.9 |

*(Super)GLUE*
| Model | CoLA | SST-2 | MRPC (F1) | QQP (F1) | MNLI | MNLI-mm | QNLI | RTE | BoolQ | MultiRC | WSC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| *Majority label* | *69.5* | *50.2* | *82* | *53.1* | *35.7* | *35.7* | *35.4* | *53.1* | *50.5* | *59.9* | *53.2* | *61.4* |
| OPT-125m | 73.7 | 86.6 | 82.1 | 77.8 | 70.1 | 71.9 | 80.1 | 67.7 | 66.0 | 61.1 | 59.0 |
| RoBERTa-base | 75.9 | 88.6 | 80.5 | 78.5 | 68.7 | 78.0 | 82.3 | 51.5 | 59.9 | 61.3 | 61.4 |
| T5-base | 76.3 | 88.0 | 85.9 | 79.7 | 71.5 | 74.0 | 83.1 | 60.6 | 69.0 | 62.4 | 60.2 |

*MSGS*
| Model | CR (Control) | LC (Control) | MV (Control) | RP (Control) | SC (Control) | CR_LC | CR_RTP | MV_LC | MV_RTP | SC_LC | SC_RP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OPT-125m | 97.2 | 82.6 | 100.0 | 99.8 | 88.1 | 75.3 | 67.1 | 66.3 | 66.8 | 84.8 | 62.0 | 
| RoBERTa-base | 93.0 | 100.0 | 100.0 | 100.0 | 89.0 | 68.3 | 66.8 | 66.6 | 80.2 | 67.4 | 67.4 | 
| T5-base | 95.1 | 100.0 | 100.0 | 99.8 | 88.7 | 76.7 | 69.4 | 67.0 | 67.7 | 72.7 | 68.0 |

-----------------------

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

Please cite the following if you choose to include the Age-of-acquisition prediction evaluation:
```
@manuscript{portelance2023predicting,
    author = {Portelance, Eva and Duan, Yuguang and Frank, Michael C. and Lupyan, Gary},
    title = {Predicting age of acquisition for children’s early vocabulary in five languages using language model surprisal},
    year = {2023},
    url = {https://github.com/evaportelance/multilingual-aoa-prediction}
    }
