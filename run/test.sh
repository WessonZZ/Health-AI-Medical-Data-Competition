#!/bin/bash
# python main.py --SAVE_PATH "result_v1" --prompt_type "few_shot" #v1/v2
# python main.py --SAVE_PATH "result_v3" --prompt_type "few_shot" --stage 1 #v3 stage = 1
# python main.py --DATA_PATH "result_v3/result_v3" --SAVE_PATH "result_v3_2" --prompt_type "few_shot" --stage 2 #v3 stage = 2
# python main.py --SAVE_PATH "result_v4" --prompt_type "llm_prompt" #llm_prompt
# python main.py --SAVE_PATH "result_v5" --prompt_type "few_shot" --api 1 --model "deepseek" #deepseek + few_shot
# python main.py --SAVE_PATH "result_v6" --prompt_type "few_shot" --stage 1 --api 1 --model "deepseek" #v6 stage = 1
# python main.py --DATA_PATH "result_v6/result_v6" --SAVE_PATH "result_v6_2" --prompt_type "few_shot" --stage 2 --api 1 --model "deepseek" #v6 stage = 2
python prompt.py --SAVE_PATH "result_v8" --prompt_type "llm_prompt" --api 1 --model "deepseek" #llm_prompt