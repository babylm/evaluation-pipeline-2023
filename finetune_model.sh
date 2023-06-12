#!/bin/bash

MODEL_PATH=$1
TASK_NAME=$2
LR=${3:-5e-5}           # default: 5e-5
PATIENCE=${4:-10}       # default: 10
BSZ=${5:-64}            # default: 64
EVAL_EVERY=${6:-200}    # default: 200
MAX_EPOCHS=${7:-10}     # default: 10
SEED=${8:-12}           # default: 12

mkdir -p $MODEL_PATH/finetune/$TASK_NAME/

if [[ "$TASK_NAME" = "mnli" ]]; then
    VALID_NAME="validation_matched"
    OUT_DIR="mnli"
elif [[ "$TASK_NAME" = "mnli-mm" ]]; then
    VALID_NAME="validation_mismatched"
    TASK_NAME="mnli"
    OUT_DIR="mnli-mm"
else
    VALID_NAME="validation"
    OUT_DIR=$TASK_NAME
fi

python finetune_classification.py \
  --model_name_or_path $MODEL_PATH \
  --output_dir $MODEL_PATH/finetune/$OUT_DIR/ \
  --train_file filter-data/glue_filtered/$TASK_NAME.train.json \
  --validation_file filter-data/glue_filtered/$TASK_NAME.$VALID_NAME.json \
  --do_train \
  --do_eval \
  --do_predict \
  --use_fast_tokenizer False \
  --max_seq_length 128 \
  --per_device_train_batch_size $BSZ \
  --learning_rate $LR \
  --num_train_epochs $MAX_EPOCHS \
  --evaluation_strategy steps \
  --patience $PATIENCE \
  --eval_every $EVAL_EVERY \
  --eval_steps $EVAL_EVERY \
  --save_steps $EVAL_EVERY \
  --overwrite_output_dir \
  --seed $SEED
