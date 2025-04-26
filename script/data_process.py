
import os
import json
from torch.utils.data import Dataset, DataLoader
from util import read_data, str_to_bool

from transformers import AutoTokenizer


class DiseaseData(Dataset):
    def __init__(self, data, tokenizer, config):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = config.max_seq_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        for d in self.data:
            if d['id'] == index:
                data_index = d
        data_index = data_process(data_index, self.tokenizer, self.max_seq_length)
        return data_index


def data_process(sample, tokenizer, max_seq_length):

    assert isinstance(sample, dict), "Error! Sample is not dict object."
    assert "id" in sample, "Error! Sample has no key \"id\"."
    assert "diseases" in sample, "Error! Sample has no key \"diseases\"."
    assert "reason" in sample, "Error! Sample has no key \"reason\"."
    assert "feature_content" in sample, "Error! Sample has no key \"feature_content\"."
    
    input_ids, attention_mask, labels = [], [], []
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    system_prompt = f"<|im_start|>system\n你是一个专业的医疗助手，掌握了广泛的医疗知识。现提供给你一些问诊症状，请根据你所掌握的医学知识尽可能确定患者所患的疾病以及相应的诊断依据。\n注意格式规范，如果有多个疾病则用阿拉伯数字编号，如\"<disease>1. 疾病；...</disease>\"，诊断依据以\"<reason>1. 依据1；2.依据2...</reason>\"格式表述。<|im_end|>\n"
    user_input = f"<|im_start|>user\n请诊断：\n{sample['feature_content']}<|im_end|>\n"
    response = f"<|im_start|>assistant\n该患者可能患的疾病为：<disease>{sample['diseases']}</disease>\n诊断依据：<reason>{sample['reason']}</reason><|im_end|>\n"
    instruction = tokenizer(system_prompt + user_input, add_special_tokens = False)
    response = tokenizer(response, add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + response["attention_mask"]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
    seq_length = len(input_ids)
    if seq_length < max_seq_length:
        input_ids = input_ids + [eos_token_id] + (max_seq_length - seq_length - 1) * [pad_token_id]
        attention_mask = attention_mask + [1] + (max_seq_length - seq_length - 1) * [0]
        labels = labels + [eos_token_id] + (max_seq_length - seq_length - 1) * [pad_token_id]
    elif seq_length >= max_seq_length:
        input_ids = input_ids[:max_seq_length - 1] + [eos_token_id]
        attention_mask = attention_mask[:max_seq_length - 1] + [1] 
        labels = labels[:max_seq_length - 1] + [eos_token_id]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
    
    
class config:
    DATA_PATH = "datasets"
    train_DATA = 'samples111'
    max_seq_length = 1024
    
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
train_path = os.path.join(config.DATA_PATH, 'extra_data', config.train_DATA)
train_data = read_data(train_path)
data = DiseaseData(train_data, tokenizer, config)

dataloader = DataLoader(data, batch_size= 24, shuffle= False)
length = []
# for i in range(len(data)):
#     if len(data[i]['labels'])>1024:
#         print(i, len(data[i]['labels']))
#     length.append(len(data[i]['labels']))
# print(length)

for data in dataloader:
    print(len(data['input_ids']))