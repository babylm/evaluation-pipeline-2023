import argparse
import os
import json

TASKS = {
    "glue": ["cola", "sst2", "mrpc", "qqp", "mnli", "mnli-mm", "qnli", "rte",
             "boolq", "multirc", "wsc"],
    "blimp": ["anaphor_agreement", "argument_structure", "binding", "control_raising",
              "determiner_noun_agreement", "ellipsis", "filler_gap", "irregular_forms",
              "island_effects", "npi_licensing", "quantifiers", "subject_verb_agreement"]
}

def make_task_dict(task_name, preds_path):
    if task_name in TASKS["glue"]:
        task_type = "glue"
    elif task_name in TASKS["blimp"]:
        task_type = "blimp"
    else:
        raise ValueError("Invalid task: {task_name}!")

    if not os.path.exists(preds_path):
        print(f"Warning: no predictions found for the \"{task_name}\" ({task_type}) task!")

    task_dict = {"task": task_type, "sub_task": task_name, "predictions": []}
    with open(preds_path, 'r') as predictions_file:
        # skip header
        next(predictions_file)
        # collect predictions with ids
        for line in predictions_file:
            index, prediction = line.strip().split("\t")
            example_id = f"{task_name}_{index}"
            task_dict["predictions"].append({"id": example_id, "pred": prediction})

    return task_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str,
                        help="Path to huggingface model and tokenizer.")
    args = parser.parse_args()

    task_dicts = {}
    for task in TASKS["glue"]:
        preds_path = os.path.join(args.model_path, "finetune", task, "predict_results.txt")
        task_dicts[task] = make_task_dict(task, preds_path)
    for task in TASKS["blimp"]:
        preds_path = os.path.join(args.model_path, "zeroshot", task, "predictions.txt")
        task_dicts[task] = make_task_dict(task, preds_path)

    with open("all_predictions.json", "w") as predictions_out:
        for task in task_dicts:
            predictions_out.write(json.dumps(task_dicts[task]) + "\n")
