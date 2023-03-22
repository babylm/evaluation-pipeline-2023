#!/bin/bash

model_path=$1

# Verify that results for all tasks are present
for task in "anaphor_agreement" "argument_structure" "binding" "control_raising" \
            "determiner_noun_agreement" "ellipsis" "filler_gap" "irregular_forms" \
            "island_effects" "npi_licensing" "quantifiers" "subject_verb_agreement"; do
    if [ ! -f $model_path/zeroshot/$task/eval_results.json ]; then
        echo "Warning: results file for $task (BLiMP) does not exist!"
    fi
done

for task in "cola" "sst2" "mrpc" "qqp" "mnli" "mnli-mm" "qnli" "rte" \
              "boolq" "multirc" "wsc"; do
    if [ ! -f $model_path/finetune/$task/eval_results.json ]; then
        echo "Warning: results file for $task (GLUE) does not exist!"
    fi
done

base_dir=`pwd`
cd $model_path
zip results.zip zeroshot/*/eval_results.json finetune/*/eval_results.json
mv results.zip $base_dir
cd $base_dir