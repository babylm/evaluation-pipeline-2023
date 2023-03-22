#!/bin/bash

MODEL_PATH=$1
LR=${2:-5e-5}
PATIENCE=${3:-10}
BSZ=${4:-64}
EVAL_EVERY=${5:-200}
MAX_EPOCHS=${6:-10}
SEED=${7:-12}

# use default hyperparameters
for task in {"cola","sst2","mrpc","qqp","mnli","mnli-mm","qnli","rte","boolq","multirc","wsc"}; do
    # qsub -q g.q -cwd -j y -l hostname="b1[123456789]|c0*|c1[13456789],ram_free=10G,mem_free=10G,gpu=1" finetune_model.sh $MODEL_PATH $task
	./finetune_model.sh $MODEL_PATH $task $LR $PATIENCE $BSZ $EVAL_EVERY $MAX_EPOCHS $SEED
done