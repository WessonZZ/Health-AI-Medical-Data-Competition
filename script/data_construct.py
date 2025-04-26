
import os
import json
from argparse import ArgumentParser, ArgumentTypeError
import time
from datetime import datetime
import re
from rouge import Rouge
import numpy as np

import concurrent.futures
import threading

from util import Prompts, read_data, save_jsonl, extract_info, process_string
from api import apply_Qwen2_5, get_api, deepseek_vol, deepseek_QWen_sl, Qwen72B_ali


def parse_config():
    parser = ArgumentParser()
    parser.add_argument("--DATA_PATH",    type = str, default = 'datasets/extra_data', help= "raw data path")
    parser.add_argument("--SAVE_PATH",    type = str, default = 'samples1', help = "path to save file")
    parser.add_argument("--stage",        type = int, default = 0, help = "three stage method for constructing dataset")
    parser.add_argument("--times",        type = int, default = 1000, help = "times for generate new samples")
    parser.add_argument("--api",          type = int, default = 1, help = "0 for 硅基流动，1 for 火山，2 for 百炼")
    parser.add_argument("--model",        type = str, help = "Qwen2.5-72B or Deepseek-V3")
    parser.add_argument("--max_workers",  type = int, default = 20)
    
    return parser.parse_args()


def generate(data, example_data, config, save_file, lock):
    """
    data: all saved samples in stage 1; single sample in stage 2
    """
    stage = config.stage
    if stage == 0:
        data = read_data(save_file)
    api_key = get_api()[config.api]
    example_data = sample_example(data) if stage == 0 else example_data
    diseases_num = range(1, 7)
    diseases_num = np.random.choice(diseases_num, 1, replace = False, p = [0.25, 0.2, 0.2, 0.15, 0.1, 0.1]).tolist()
    Prompt = Prompts(EXAMPLE= example_data) if stage == 0 else Prompts(EXAMPLE= example_data, CASE=data)
    content = Prompt.data_construct(stage = stage, diseases_num = diseases_num[0])
    model = deepseek_vol if config.model == 'deepseek' else Qwen72B_ali
    
    # id = max([d['id'] for d in data]) + 1
    s = time.time()
    Flag = True
    times = 0
    while Flag:
            response = model(content, api_key, flag = 1)
            if stage == 0:
                temp = extract_info_construct(response, stage=stage)
                new_data = []
                for t in temp:
                    t_feature = t["feature_content"]
                    for d in data:
                        feature = d["feature_content"]
                        if cal_ROUGE(t_feature, feature):
                            pass
                        else:
                            break
                    new_data.append(t)
                k = 0
                with lock:
                    data = read_data(save_file)
                    with open(save_file + ".jsonl", 'w', encoding='utf-8') as output_file:
                        for d in data:
                            output_file.write(json.dumps(d, ensure_ascii=False) + '\n')   
                        for d in new_data:
                            tt = {}
                            tt["id"] = None
                            tt["reason"] = None
                            tt["diseases"] = None if stage == 0 else d['diseases']
                            tt["feature_content"] = d["feature_content"]
                            k += 1
                            output_file.write(json.dumps(tt, ensure_ascii=False) + '\n')    
                if k > 0:
                    Flag = False

            elif stage == 1:
                diseases, reason = extract_info(response)
                if diseases is not None and len(diseases) >= 1:
                    result = {'id': data['id'], 'diseases': process_string(diseases), 'reason': reason, 'feature_content': data['feature_content']}
                    with lock:
                        data = read_data(save_file)
                        with open(save_file + ".jsonl", 'w', encoding='utf-8') as output_file:
                            for d in data:
                                if d['id'] == result['id']:
                                    d = result
                                output_file.write(json.dumps(d, ensure_ascii=False) + '\n')    
                    Flag = False
                else:
                    time.sleep(0.5) #try again

            elif stage == 2:
                diseases, reason = extract_info(response)
                if reason is not None and len(reason) >= 1:
                    result = {'id': data['id'], 'diseases': data['diseases'], 'reason': reason, 'feature_content': data['feature_content']}
                    with lock:
                        data = read_data(save_file)
                        with open(save_file + ".jsonl", 'w', encoding='utf-8') as output_file:
                            for d in data:
                                if d['id'] == result['id']:
                                    d = result
                                output_file.write(json.dumps(d, ensure_ascii=False) + '\n')   
                    Flag = False
                else:
                    time.sleep(0.5) #try again

            if times >= 3:
                Flag = False

    e = time.time()
    dt = datetime.fromtimestamp(e)
    formatted_time = dt.strftime("%H:%M:%S")
    if stage == 0:
        print(f"Time {formatted_time}---generate {len(temp)} samples: Time {e-s}.", flush=True)
    elif stage in [1,2]:
        print(f"Time {formatted_time}---Data id {data['id']} finished: Time {e-s}.", flush=True)
    
    
def sort_result(data, file):
    data.sort(key = lambda x: x["id"])
    save_jsonl(data, file)
    

def sample_example(data):
    if len(data) <= 246:
        selected_data = np.random.choice(data, 6, replace=False)
    else:
        selected_data = np.random.choice(data[:246], 4, replace=False).tolist()
        selected_data1 = np.random.choice(data[246:], 2, replace=False).tolist()
        selected_data.extend(selected_data1)
    return selected_data
        

def cal_ROUGE(text, refs):
    flag = True
    rouge = Rouge()
    F1 = rouge.get_scores(text, refs, avg = True)['rouge-l']['f']
    if F1 > 0.7:
        flag = False
        return flag
    return flag
    

def extract_info_construct(raw_response, stage):
    pattern2 = r"<feature_content>(.*?)</feature_content>"

    features = re.findall(pattern2, raw_response, re.DOTALL)
    features = [r.strip() for r in features] if features else None
    
    if stage == 0:
        data = []
        for feature in features:
            if feature is not None:
                data.append({"feature_content": feature})
    
        return data
    return None


def main(config):
    # config = parse_config()
    lock = threading.Lock()
    save_file = os.path.join(config.DATA_PATH, config.SAVE_PATH)
    
    if config.stage == 0:
        with concurrent.futures.ThreadPoolExecutor(max_workers = config.max_workers) as executor:
                future_prediction = {executor.submit(generate, i, i, config, save_file, lock): i for i in range(config.times)}
                print(f"Total tasks submitted: {config.times}")
    else:
        data = read_data(save_file)
        example = [d for d in data[:246] if d["diseases"] is not None and d["reason"] is not None]
        example_data = np.random.choice(example, 6, replace=False)
        data_new = []
        for d in data:
            if config.stage == 1:
                if d["diseases"] is None:
                    data_new.append(d)
            elif config.stage == 2:
                if d["reason"] is None:
                    data_new.append(d)
        with concurrent.futures.ThreadPoolExecutor(max_workers = config.max_workers) as executor:
            future_prediction = {executor.submit(generate, data, example_data, config, save_file, lock): data for data in data_new[:2]}
            print(f"Total tasks submitted: {len(data_new)}")
                

if __name__ == "__main__":
    
    class config:
        DATA_PATH  = 'datasets/extra_data'
        SAVE_PATH  = 'samples11'
        stage      =  2
        times      = 60
        api        = 1
        model      = "deepseek"
        max_workers = 2
        
    main(config)