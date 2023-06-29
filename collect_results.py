import argparse
import os
import json

TASKS = {
    "glue": ["cola", "sst2", "mrpc", "qqp", "mnli", "mnli-mm", "qnli", "rte",
             "boolq", "multirc", "wsc"],
    "blimp": ["anaphor_agreement", "argument_structure", "binding", "control_raising",
              "determiner_noun_agreement", "ellipsis", "filler_gap", "irregular_forms",
              "island_effects", "npi_licensing", "quantifiers", "subject_verb_agreement"],
    "supplement": ["hypernym", "qa_congruence_easy", "qa_congruence_tricky",
                   "subject_aux_inversion", "turn_taking"],
    "msgs": ["main_verb_control", "control_raising_control", "syntactic_category_control",
             "relative_position_control", "lexical_content_the_control",
             "main_verb_lexical_content_the", "main_verb_relative_token_position",
             "control_raising_lexical_content_the", "control_raising_relative_token_position",
             "syntactic_category_lexical_content_the", "syntactic_category_relative_position"]
}

def make_task_dict(task_name, preds_path):
    if task_name in TASKS["glue"]:
        task_type = "glue"
    elif task_name in TASKS["blimp"]:
        task_type = "blimp"
    elif task_name in TASKS["supplement"]:
        task_type = "supplement"
    elif task_name in TASKS["msgs"]:
        task_type = "msgs"
    else:
        raise ValueError(f"Invalid task: {task_name}!")

    if not os.path.exists(preds_path):
        raise FileNotFoundError(f"Warning: no predictions found for the \"{task_name}\" ({task_type}) task!")

    task_dict = {"task": task_type, "sub_task": task_name, "predictions": []}
    with open(preds_path, 'r') as predictions_file:
        # skip header
        next(predictions_file)
        # collect predictions with ids
        for line in predictions_file:
            index, prediction = line.strip().split("\t")
            prediction = prediction.replace("\\n", "\n")
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
    for task in TASKS["msgs"]:
        preds_path = os.path.join(args.model_path, "finetune", task, "predict_results.txt")
        task_dicts[task] = make_task_dict(task, preds_path)
    for task in TASKS["blimp"]:
        preds_path = os.path.join(args.model_path, "zeroshot", task, "predictions.txt")
        task_dicts[task] = make_task_dict(task, preds_path)
    for task in TASKS["supplement"]:
        preds_path = os.path.join(args.model_path, "zeroshot", task, "predictions.txt")
        task_dicts[task] = make_task_dict(task, preds_path)

    with open("all_predictions.json", "w") as predictions_out:
        for task in task_dicts:
            predictions_out.write(json.dumps(task_dicts[task]) + "\n")
    print("Predictions output at `all_predictions.json`.")
