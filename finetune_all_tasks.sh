#!/bin/bash

MODEL_PATH=$1
LR=${2:-5e-5}
PATIENCE=${3:-10}
BSZ=${4:-64}
EVAL_EVERY=${5:-200}
MAX_EPOCHS=${6:-10}
SEED=${7:-12}

# Fine-tune and evaluate on (Super)GLUE tasks
# If your system uses sbatch or qsub, consider using that to parallelize calls to finetune_model.sh
for subtask in {"cola","sst2","mrpc","qqp","mnli","mnli-mm","qnli","rte","boolq","multirc","wsc"}; do
    ./finetune_model.sh $MODEL_PATH glue $subtask $LR $PATIENCE $BSZ $EVAL_EVERY $MAX_EPOCHS $SEED
done

# Fine-tune and evaluate on MSGS tasks
for subtask in {"main_verb_control","control_raising_control","syntactic_category_control","lexical_content_the_control","relative_position_control","main_verb_lexical_content_the","main_verb_relative_token_position","syntactic_category_lexical_content_the","syntactic_category_relative_position","control_raising_lexical_content_the","control_raising_relative_token_position"}; do
	./finetune_model.sh $MODEL_PATH msgs $subtask $LR $PATIENCE $BSZ $EVAL_EVERY $MAX_EPOCHS $SEED
done
