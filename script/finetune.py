
"""
fine-tune Qwen2.5-7B-instruct using lora
reference for https://zhuanlan.zhihu.com/p/689918127
"""

import os
import json
from argparse import ArgumentParser, ArgumentTypeError
import time

import torch
import unsloth
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

from util import read_data, str_to_bool
from data_process import DiseaseData


def parse_config():
    parser = ArgumentParser()
    parser.add_argument("--model",            type = str, help = "Qwen2.5-7B-Instruct")
    parser.add_argument("--dtype",            type = str, default = 'bfloat16')
    parser.add_argument("--load_in_4bit",     type = str_to_bool, default = False)
    parser.add_argument("--do_train",         type = str_to_bool, default = True, help = "train mode")
    parser.add_argument("--do_eval",          type = str_to_bool, default = True, help = "eval mode")
    parser.add_argument("--train_DATA",       type = str, default = 'samples111', help= "data file")
    parser.add_argument("--eval_DATA",        type = str, default = 'result_v6_21', help= "data file")
    parser.add_argument("--max_seq_length",   type = int, default = 1024)
    parser.add_argument("--DATA_PATH",        type = str, default = 'datasets', help= "data path")
    parser.add_argument("--output_PATH",       type = str, default = '/root/output', help = "path to load model")
    
    parser.add_argument("--lora_rank",        type = int, default = 8)
    parser.add_argument("--lora_alpha",       type = int, default = 16) #一般是秩的两倍
    parser.add_argument("--dropout",          type = float, default = 0.1)
    
    parser.add_argument("--epoch",                         type = int, default = 4)
    parser.add_argument("--warmup_steps",                  type = int, default = 4)
    parser.add_argument("--gradient_accumulation_steps",   type = int, default = 8)
    parser.add_argument("--per_device_train_batch_size",   type = int, default = 4)
    parser.add_argument("--per_device_eval_batch_size",    type = int, default = 8)
    parser.add_argument("--lr",                            type = float, default = 1e-4)
    parser.add_argument("--eval_steps",                    type = int, default = 50)
    parser.add_argument("--logging_steps",                 type = int, default = 5)
    parser.add_argument("--logging_dir",                   type = str)
    parser.add_argument("--save_steps",                    type = int, default =100)
    
    return parser.parse_args()


def load_model_tokenizer(config):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config.model,
        max_seq_length = config.max_seq_length,
        dtype= torch.bfloat16 if torch.cuda.is_bf16_supported() and config.dtype == "bfoat16" else torch.float16,
        load_in_4bit = config.load_in_4bit
    )
    return model, tokenizer


def main():
    config = parse_config()
    model = config.model 
    do_train, do_eval = config.do_train, config.do_eval
    
    model, tokenizer = load_model_tokenizer(config)
    
    if do_train:
        train_path = os.path.join(config.DATA_PATH, 'extra_data', config.train_DATA)
        train_data = read_data(train_path)
        train_dataset = DiseaseData(train_data, tokenizer, config)
    if do_eval:
        eval_path = os.path.join(config.DATA_PATH, 'result_v6', config.eval_DATA)
        eval_data = read_data(eval_path)[-100:]
        eval_dataset = DiseaseData(eval_data, tokenizer, config)
    

    model = FastLanguageModel.get_peft_model(
        model,
        r= config.lora_rank,
        target_modules= ["q_proj", "k_proj", "v_proj"],
        lora_alpha= config.lora_alpha,
        lora_dropout= config.dropout,
        bias="none",
        use_gradient_checkpointing= True,
        random_state= 42,
        use_rslora= False,
        loftq_config = None,
    )
    
    training_args = TrainingArguments(
        output_dir = config.output_PATH,
        do_train = config.do_train,
        do_eval = config.do_eval,
        per_device_train_batch_size = config.per_device_train_batch_size,
        gradient_accumulation_steps = 8,
        warmup_steps = config.warmup_steps,
        eval_steps= config.eval_steps,
        num_train_epochs = config.epochs,
        learning_rate = config.lr,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = config.logging_steps,
        optim = "adamw_torch",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 42,
        logging_dir = config.logging_dir,
        report_to = "none",
        save_steps = config.save_steps
    )
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        max_seq_length = config.max_seq_length,
        args = training_args
    )
    
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()
    
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory/max_memory*100, 3)
    lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


if __name__=="__main__":
    
    main()