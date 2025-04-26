#!/bin/bash

DATESTR=`date +%Y%m%d-%H%M%S`
RUN_NAME=model_train
OUTPUT_DIR=output/${RUN_NAME}-${DATESTR}
mkdir -p $OUTPUT_DIR

MODEL="/root/Qwen2.5-7B-Instruct"

LR=2e-4

python finetune.py \
    --model "${MODEL}"\
    --do_train True\
    --do_eval True\
    --train_DATA 'samples111'\
    --max_seq_length 1024\
    --output_PATH $OUTPUT_DIR \
    --lora_rank 8 \
    --lora_alpha 16 \
    --dropout 0.1\
    --epochs 4\
    --warmup_steps 40\
    --per_device_train_batch_size 10\
    --per_device_eval_batch_size  12\
    --gradient_accumulation_steps 8\
    --lr $LR\
    --eval_steps 50\
    --logging_steps 5 \
    --logging_dir $OUTPUT_DIR/logs \
    --save_steps 100
