"""
BLiMP: A Benchmark of Linguistic Minimal Pairs for English
https://arxiv.org/abs/1912.00582

BLiMP is a challenge set for evaluating what language models (LMs) know about
major grammatical phenomena in English. BLiMP consists of 67 sub-datasets, each
containing 1000 minimal pairs isolating specific contrasts in syntax, morphology,
or semantics. The data is automatically generated according to expert-crafted
grammars.

Homepage: https://github.com/alexwarstadt/blimp
"""
from lm_eval.api.task import PromptSourceTask
from typing import Optional, List
from datasets import load_dataset


_CITATION = """
@article{warstadt2019blimp,
    author = {Warstadt, Alex and Parrish, Alicia and Liu, Haokun and Mohananey, Anhad and Peng, Wei and Wang, Sheng-Fu and Bowman, Samuel R.},
    title = {BLiMP: The Benchmark of Linguistic Minimal Pairs for English},
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {8},
    number = {},
    pages = {377-392},
    year = {2020},
    doi = {10.1162/tacla00321},
    URL = {https://doi.org/10.1162/tacl_a_00321},
    eprint = {https://doi.org/10.1162/tacl_a_00321},
    abstract = { We introduce The Benchmark of Linguistic Minimal Pairs (BLiMP),1 a challenge set for evaluating the linguistic knowledge of language models (LMs) on major grammatical phenomena in English. BLiMP consists of 67 individual datasets, each containing 1,000 minimal pairs—that is, pairs of minimally different sentences that contrast in grammatical acceptability and isolate specific phenomenon in syntax, morphology, or semantics. We generate the data according to linguist-crafted grammar templates, and human aggregate agreement with the labels is 96.4. We evaluate n-gram, LSTM, and Transformer (GPT-2 and Transformer-XL) LMs by observing whether they assign a higher probability to the acceptable sentence in each minimal pair. We find that state-of-the-art models identify morphological contrasts related to agreement reliably, but they struggle with some subtle semantic and syntactic phenomena, such as negative polarity items and extraction islands. }
}
"""

class CompsTask(PromptSourceTask):
    DATASET_PATH = "kanishka/comps"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        # The HF dataset only contains a "train" dataset, but the harness expects a "validation"
        # dataset. Let's use the training dataset, on the assumption that the model wasn't actually
        # trained on this data.
        return self.dataset["train"]
    
    def null_prompt_doc_to_text(self, doc: dict) -> str:
        return ""

    def null_prompt_doc_to_target(self, doc: dict) -> List[str]:
        correct_choice = doc["prefix_acceptable"]
        sentence_end = doc["property_phrase"]
        formatted = f"{correct_choice} {sentence_end}"
        return [formatted]
    
    def null_prompt_answer_choices(self, doc: dict) -> List[str]:
        choices = [doc["prefix_acceptable"], doc["prefix_unacceptable"]]
        sentence_end = doc["property_phrase"]
        formatted = [f"{choice} {sentence_end}" for choice in choices]
        return formatted
    
    def null_prompt_get_logging_info(self):
        return {
            "fixed_answer_choice_list": None,
            "dataset_path": self.DATASET_PATH,
            "dataset_name": self.DATASET_NAME,
            "subset": self.SPLIT,
            "prompt_name": None,
            "prompt_id": None,
            "prompt_jinja": None,
            "prompt_original_task": f"{self.DATASET_PATH}/{self.DATASET_NAME}",
            # Placeholder for comment in post-processing.
            "comment": "",
        }


class CompsBase(CompsTask):
    DATASET_NAME = "base"


class CompsWugs(CompsTask):
    DATASET_NAME = "wugs"