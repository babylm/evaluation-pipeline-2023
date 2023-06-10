import json
import os
import sys

from collections import OrderedDict


def pp_results(model_path):
    tasks = sorted(os.listdir(model_path))
    eval_scores = OrderedDict()
    for task in tasks:
        with open(f"{model_path}/{task}/all_results.json", "r") as f:
            results = json.load(f)
        eval_scores[task] = [results.get("eval_accuracy", -1), results.get("eval_f1", -1)]

    # print(f"{'Task':<20}{'Accuracy':<20}{'F1':<20}")
    # pretty print
    # for task, scores in eval_scores.items():
    #     print(f"{task:<20}{scores[0]*100:<20.4f}{scores[1]*100:<20.4f}")
    # print in csv format

    for _, scores in eval_scores.items():
        if scores[0] != -1:
            print(f"{scores[0]*100:.2f}", end=",")
        if scores[1] != -1:
            print(f"{scores[1]*100:.2f}", end=",")
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python pp_results.py <model_path>")
        return
    model_paths = sys.argv[1:]
    for model_path in model_paths:
        pp_results(model_path)


if __name__ == "__main__":
    main()

"""
python pp_results.py finetune/mcgill-babylm/bert_ds10M_np512_nh2_nl2_hs128_ungrouped \
                     finetune/mcgill-babylm/bert_ds10M_np512_nh4_nl4_hs256_ungrouped \
                     finetune/mcgill-babylm/bert_ds10M_np512_nh8_nl8_hs512_ungrouped \
                     finetune/mcgill-babylm/bert_ds10M_np512_nh12_nl12_hs768_ungrouped \
                     finetune/mcgill-babylm/bert_ds10M_np512_nh2_nl2_hs128_postags_ungrouped \
                     finetune/mcgill-babylm/bert_ds10M_np512_nh4_nl4_hs256_postags_ungrouped \
                     finetune/mcgill-babylm/bert_ds10M_np512_nh8_nl4_hs512_postags_ungrouped \
                     finetune/mcgill-babylm/bert_ds10M_np512_nh8_nl8_hs512_postags_ungrouped \
                     finetune/mcgill-babylm/bert_ds10M_np512_nh12_nl12_hs768_postags_ungrouped
"""
