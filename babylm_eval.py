import argparse
import lm_eval
import os
import json

TASKS = {
    "blimp": ["anaphor_agreement.json", "argument_structure.json", "binding.json",
              "control_raising.json", "determiner_noun_agreement.json", "ellipsis.json",
              "filler_gap.json", "irregular_forms.json", "island_effects.json",
              "npi_licensing.json", "quantifiers.json", "subject_verb_agreement.json"],
    "supplement": ["hypernym.json", "qa_congruence_easy.json", "qa_congruence_tricky.json",
               "subject_aux_inversion.json", "turn_taking.json"]
}


def accuracy_on_task(task_name, eval_model, template_name, num_fewshot):
    predictions_path = os.path.join(args.model_path, "zeroshot", task_title, "predictions.txt")
    predictions_dir = os.path.dirname(predictions_path)
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)

    eval_task = lm_eval.get_task_list(task_name, template_names=[template_name])
    results = lm_eval.evaluate(model=eval_model, tasks=eval_task, seed=12,
                               num_fewshot=num_fewshot, predictions_path=predictions_path)
    accuracy = results['results'][0]['acc']
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str,
                        help="Path to huggingface model and tokenizer.")
    parser.add_argument("model_type", type=str, choices=["decoder only", "decoder", "encoder only", "encoder", "encoder-decoder",],
                        help="Language model architecture.")
    parser.add_argument("--tasks", "-t", type=str, choices=["blimp", "supplement", "all"], default="all",
                        help="Tasks on which we evaluate.")
    parser.add_argument("--run_aoa", "-a", action="store_true",
                        help="Will run the additional AoA prediction task.")
    parser.add_argument("--trust_remote_code", "-r", action="store_true",
                        help="Trust remote code (e.g. from huggingface) when loading model.")
    parser.add_argument("--num_fewshot", "-n", type=int, default=0,
                        help="Number of few-shot examples to show the model for each test example.")
    args = parser.parse_args()

    MODEL_TYPE_REMAP = {"decoder only": "hf-causal", "decoder": "hf-causal",
                        "encoder only": "hf-mlm", "encoder": "hf-mlm",
                        "encoder-decoder": "hf-seq2seq",}
    eval_model = lm_eval.get_model(MODEL_TYPE_REMAP[args.model_type],
                                   pretrained=args.model_path,
                                   trust_remote_code=args.trust_remote_code,
                                   device="cuda")
    tasks = []
    if args.tasks == "all":
        for task_type in TASKS.keys():
            tasks.extend(TASKS[task_type])
    else:
        tasks = TASKS[args.tasks]

    accuracies = {}
    # Iterate through tasks, get accuracies
    for task in tasks:
        if task in TASKS["blimp"]:
            template = None
            task_title = task.split(".json")[0]
            task = f"blimp_from_file:filter-data/blimp_filtered/{task}"
        elif task in TASKS["supplement"]:
            template = None
            task_title = task.split(".json")[0]
            task = f"blimp_from_file:filter-data/supplement_filtered/{task}"
        else:
            raise ValueError("Unrecognized task!")
        accuracies[task_title] = accuracy_on_task(task, eval_model, template,
                    args.num_fewshot)
        print(f"{task_title}:\t{accuracies[task_title] * 100:.2f}%")
        # Write scores to file
        out_path = os.path.join(args.model_path, "zeroshot", task_title, "eval_results.json")
        out_dir = os.path.dirname(out_path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(out_path, 'w') as out_file:
            json.dump({"eval_accuracy": accuracies[task_title]}, out_file)

    # Print scores
    print("\nScores:")
    for task in accuracies.keys():
        print(f"{task}:\t{accuracies[task] * 100:.2f}%")

    if args.run_aoa:
        # Run AoA prediction evaluation
        word_surprisals_n, mad_results = lm_eval.aoa_pred_eval(eval_model.model, eval_model.tokenizer, MODEL_TYPE_REMAP[args.model_type], batch_size = 32)
        out_dir = os.path.join(args.model_path, "aoa_prediction")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(os.path.join(out_dir, "extracted_average_surprisals.json") , 'w') as out_file:
            json.dump(word_surprisals_n, out_file)
        with open(os.path.join(out_dir, "mean_absolute_deviation_results.json"), 'w') as out_file:
            json.dump(mad_results, out_file)
