
"""
prompt engineering for judging diseases
"""
import os
import time
from datetime import datetime
import multiprocessing
from multiprocessing import Manager
from argparse import ArgumentParser, ArgumentTypeError
from functools import partial
import json


from util import Prompts, read_data, save_jsonl, extract_info, process_string
from api import apply_Qwen2_5, get_api, deepseek_vol,deepseek_QWen_sl



def parse_config():
    parser = ArgumentParser()
    parser.add_argument("--DATA_PATH",    type = str, default = '20250208181531_camp_data_step_1_without_answer', help= "raw data path")
    parser.add_argument("--EXAM_PATH",    type = str, default = '20250214171329_提交示例', help= "example data path")
    parser.add_argument("--SAVE_PATH",    type = str, default = 'result_v1', help = "path to save file")
    parser.add_argument("--prompt_type",  type = str, required = True, help = "few shot or zero shot or llm prompt")
    parser.add_argument("--num_process",  type = int, default = 10, help = "num of processors")
    parser.add_argument("--stage",        type = int, default = 0, help = "two stage method")
    parser.add_argument("--api",          type = int, default = 1, help = "0 for 硅基流动，1 for 火山")
    parser.add_argument("--model",        type = str, help = "Qwen2.5-7B or Deepseek-V3")
    
    return parser.parse_args()

def generate(data, lock, config, save_file):
    example_file = config.EXAM_PATH
    prompt_type = config.prompt_type
    stage = config.stage

    s = time.time()
    example = read_data(os.path.join('datasets', example_file))
    api_key = get_api()[config.api]
    Prompt = Prompts(EXAMPLE=example, CASE=data)
    if stage <=0:
        if prompt_type == 'zero_shot':
            content = Prompt.zero_shot
        elif prompt_type == 'few_shot':
            content = Prompt.few_shot
        elif prompt_type == "llm_prompt":
            content = Prompt.llm_prompt
    else:
        # print(data['id'])
        content = Prompt.two_stage(stage = stage)
    model = deepseek_vol if config.model == 'deepseek' else apply_Qwen2_5
    Flag = True
    times = 0
    while Flag:
        response = model(content, api_key, flag = 1)
        # try:
        #     response = json.loads(response)['choices'][0]['message']['content']
        # except KeyError as e:
        #     print(f"Error in response format: {e}")
        #     time.sleep(0.5)
        diseases, reason = extract_info(response)
        
        if stage <= 0:
            if (not diseases == None) and (not reason == None) and len(diseases) >= 1 and len(reason) >= 1:
                result = {'id': data['id'], 'reason': reason, 'diseases': process_string(diseases), 'feature_content': data['feature_content']}
                with lock:
                    save_jsonl(result, save_file)
                Flag = False
            else:
                time.sleep(0.5) #try again
            times += 1
        elif stage == 1:
            if (not diseases == None) and len(diseases) >= 1:
                result = {'id': data['id'], 'reason': reason, 'diseases': process_string(diseases), 'feature_content': data['feature_content']}
                with lock:
                    save_jsonl(result, save_file)
                Flag = False
            else:
                time.sleep(0.5) #try again
        elif stage == 2:
            if (not reason == None) and len(reason) >= 1:
                result = {'id': data['id'], 'reason': reason, 'diseases': process_string(data['diseases']), 'feature_content': data['feature_content']}
                with lock:
                    save_jsonl(result, save_file)
                Flag = False
            else:
                time.sleep(0.5) #try again
        # if times >= 3:
        #     Flag = False
    e = time.time()
    dt = datetime.fromtimestamp(e)
    formatted_time = dt.strftime("%H:%M:%S")
    print(f"Time {formatted_time}---Data id {data['id']}: Time {e-s}.")
    
    
def sort_result(data, file):
    data.sort(key = lambda x: x["id"])
    save_jsonl(data, file)
    
def filter_untest(datas, ori_file = '20250208181531_camp_data_step_1_without_answer'):
    data_ids = [data['id'] for data in datas]
    ori_datas = read_data(os.path.join('datasets', ori_file))
    remain_data = []
    for data in ori_datas:
        if data['id'] not in data_ids:
            remain_data.append(data)
    return remain_data   

def init_worker():
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def main():
    config = parse_config()
    
    manager = Manager()
    lock = manager.Lock()
    
    stage = config.stage
    save_file = config.SAVE_PATH
    fold_path = os.path.join('datasets', save_file)
    if not os.path.exists(fold_path):
        os.makedirs(fold_path)
    save_file = os.path.join(fold_path, save_file)
    if os.path.exists(save_file + '.jsonl'):
        save_datas = read_data(save_file)
        if stage == 2:
            datas = filter_untest(save_datas, "result_v6/result_v6")
        else:
            datas = filter_untest(save_datas)
    else:
        datas = read_data(os.path.join('datasets', config.DATA_PATH))
    
    partial_generate = partial(generate, config = config, save_file = save_file)
    s = time.time()
    with multiprocessing.Pool(processes = config.num_process, initializer= init_worker) as pool:
        pool.starmap(partial_generate, [(data, lock) for data in datas])
    e = time.time()
    print(f"Finish generating prompts for {len(datas)} tasks----Total Time: {e - s}.") 
    
    # results = []
    # while not queue.empty():
    #     results.append(queue.get())
    # save_jsonl(results, save_file)
    # save_datas.extend(results)
    save_datas = read_data(save_file)
    if os.path.exists(save_file + '1.jsonl'):
        os.remove(save_file + '1.jsonl')
    sort_result(save_datas, save_file + '1')
    


if __name__ == '__main__':

    # class config:
    #     SAVE_PATH = "result_v8" 
    #     prompt_type = "llm_prompt" 
    #     prompt_type = "few_shot" 
    #     api = 1 
    #     model = "deepseek"
    #     DATA_PATH = '20250208181531_camp_data_step_1_without_answer'
    #     EXAM_PATH  = '20250214171329_提交示例'
    # parser.add_argument("--SAVE_PATH",    type = str, default = 'result_v1', help = "path to save file")
    # parser.add_argument("--prompt_type",  type = str, required = True, help = "few shot or zero shot or llm prompt")
    # parser.add_argument("--num_process",  type = int, default = 10, help = "num of processors")
    # parser.add_argument("--stage",        type = int, default = 0, help = "two stage method")
    # parser.add_argument("--api",          type = int, default = 1, help = "0 for 硅基流动，1 for 火山")
    # parser.add_argument("--model
    main()
    
    # manager = Manager()
    # queue = manager.Queue()

    
    # example_file = '20250214171329_提交示例'
    # save_file = "result_v5"
    # datas = read_data(os.path.join('datasets', "20250208181531_camp_data_step_1_without_answer"))
    # prompt_type = 'few_shot'
    # stage = 0
    # fold_path = os.path.join('datasets', save_file)
    # if not os.path.exists(fold_path):
    #     os.makedirs(fold_path)
    # save_file = os.path.join(fold_path, save_file)
    # if os.path.exists(save_file + '.jsonl'):
    #     datas_new = read_data(save_file)
    #     if stage == 2:
    #         datas = filter_untest(datas_new, "result_v3/result_v3")
    #     else:
    #         datas = filter_untest(datas)
    # partial_generate = partial(generate, prompt_type=prompt_type, stage=stage, example_file=example_file, save_file=save_file)
    # s = time.time()
    # with multiprocessing.Pool(processes = 1, initializer= init_worker) as pool:
    #     pool.starmap(partial_generate, [(data, queue) for data in datas[:10]])
    # e = time.time()
    # print(f"Finish generating prompts for {len(datas)} tasks.      Total Time: {e - s}.") 
    # results = []
    # while not queue.empty():
    #     results.append(queue.get())
    # datas = read_data(save_file)
    # if os.path.exists(save_file + '1.jsonl'):
    #     os.remove(save_file + '1.jsonl')
    # sort_result(datas, save_file + '1')
    
    

