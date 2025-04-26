
from dataclasses import dataclass, field
from typing import Dict, List
import os
from argparse import ArgumentTypeError
from pathlib import Path
import json
import re


def str_to_bool(value):
    if value.lower() in ("true", "yes", "t", "y", "1"):
        return True
    elif value.lower() in ("false", "no", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")

@dataclass
class Prompts:
    """
    prompt templates
    """
    Meta_prompt = ["现提供给你一些问诊症状，请根据你所掌握的医学知识确定患者所患的疾病以及相应的诊断依据。", 
                   "\n请诊断，注意格式规范：\n疾病：如果有多个疾病则用阿拉伯数字编号，如\"<disease>1. 疾病；...</disease>\"；\n诊断依据：以\"<reason>1. 依据1；2.依据2...</reason>\"形式表述。", 
                   "诊断的疾病请采用<disease></disease>格式给出，诊断依据请分点采用<reason>1. 2. 3.等</reason>格式给出。",
                   "\n对于以下案例：",
                   "现提供给你一些问诊症状，请根据你所掌握的医学知识确定患者所有可能患的疾病。", 
                   "现提供给你一些问诊症状和专业医生所诊断的疾病，请根据你所掌握的医学知识从问诊症状中尽可能多地提取确定患者患该疾病相应的诊断依据。", 
                   "\n请诊断。请注意格式规范：\n诊断的疾病如果有多个则用阿拉伯数字编号，\"<disease>1. 疾病1；2. 疾病2；...</disease>\"格式给出。", 
                   "\n请给出全面准确的诊断依据。请注意格式规范：\n诊断依据分点采用\"<reason>1. 依据1；2.依据2...</reason>\"格式给出。",
                   "请参考以下5个医疗案例，利用你所掌握的医疗知识和丰富的医疗诊断经验，列举 5 个多样化的病例症状，每个病例包含 {} 种疾病，症状和疾病需对应。要求症状需符合医学常识，症状需包含性别、年龄、主诉、现病史、既往史、个人史、过敏史、婚育史、体格检查和辅助检查等关键词，现病史和既往史需详细准确，其他部分信息可以缺失，但必须保留上述关键词存在，其余事实信息不应存在偏见。",
                   "\n输出严格要求在主诉、现病史、既往史等和症状的文本长短上具有多样性，不需要给出疾病。案例症状请参考以上案例的输出，每个案例采用<feature_content></feature_content>格式。"]

    Example_prompt: str = "\n以下是由专业医生诊断的几个案例，可供参考："
    EXAMPLE: Dict[str, str] = field(default_factory=lambda: {"reason": "", "diseases": "", "feature_content": ""})
    CASE: Dict[str, str] = field(default_factory=lambda: {"id": "", "feature_content": ""})
    LLM_prompt: str = "请基于患者的临床信息，诊断患者所有可能患的疾病，并提供诊断依据。请按照以下步骤进行分析：\n1.提取关键信息：\n-主诉: \n-现病史; \n-体格检查; \n-辅助检查结果等。 \n-既往史、个人史、过敏史、婚育史、流行病史; \n2.诊断疾病：\n-根据提取的关键信息，列出可能的疾病诊断。\n3.提供诊断依据：\n对于诊断的疾病，提供支持诊断的依据。依据应基于患者临床信息中的具体症状、体征、检查结果或病史信息。"
    DISEASE_EXAMPLE: str = ""

    def __post_init__(self):
    # Generate FEW_SHOT dynamically after initialization
        if len(self.EXAMPLE) > 0:
            example_prompt  = [f"\n案例{id + 1}：" + f"\n症状：\n{example['feature_content']}" + f"\n诊断疾病：{example['diseases']}" + f"\n诊断依据：{example['reason']}" for id, example in enumerate(self.EXAMPLE)]
            disease_prompt = [f"\n案例{id + 1}：" + f"\n症状：\n{example['feature_content']}" + f"\n诊断疾病：{example['diseases']}" for id, example in enumerate(self.EXAMPLE)]
            feature_prompt = [f"\n案例{id + 1}：" + f"\n症状：\n{example['feature_content']}" for id, example in enumerate(self.EXAMPLE)]
            self.EXAMPLE = example_prompt[0] + example_prompt[1] #+ example_prompt[2]
            self.DISEASE_EXAMPLE = disease_prompt[0] + disease_prompt[1] #+ disease_prompt[2]
            self.DATA_EXAMPLE = disease_prompt[0] + disease_prompt[1] #+ disease_prompt[2] + disease_prompt[3] + disease_prompt[4]
            self.FEATURE_EXAMPLE = feature_prompt[0] + feature_prompt[1] #+ feature_prompt[2] + feature_prompt[3] + feature_prompt[4]
        else:
            self.EXAMPLE = ""

    # --zero-shot: only task desc
    @property
    def zero_shot(self):
        prompt = f"{self.Meta_prompt[0]}" + f"\n症状：\n{self.CASE['feature_content']}" + "\n诊断疾病：<disease></disease>" + "\n诊断依据：<reason></reason>" + f"{self.Meta_prompt[1]}"
        return prompt

    # --few-shot:
    @property
    def few_shot(self):
        prompt = f"{self.Meta_prompt[0]}" + f"{self.Example_prompt}" + f"{self.EXAMPLE}" + f"{self.Meta_prompt[3]}" + f"\n症状：{self.CASE['feature_content']}" + "\n诊断疾病：<disease></disease>" + "\n诊断依据：<reason></reason>" + f"{self.Meta_prompt[1]}"
        return prompt
    
    @property
    def llm_prompt(self):
        prompt = f"{self.LLM_prompt}" + f"{self.Example_prompt}" + f"{self.DISEASE_EXAMPLE}" + f"{self.Meta_prompt[3]}" + f"\n症状：{self.CASE['feature_content']}" + "\n诊断疾病：<disease></disease>" + "\n诊断依据：<reason></reason>" + f"{self.Meta_prompt[1]}"
        return prompt
    
    # --two stage + few_shot
    def two_stage(self, stage):
        if stage == 1: #diseases
            prompt = f"{self.Meta_prompt[4]}" + f"{self.Example_prompt}" + f"{self.DISEASE_EXAMPLE}" + f"{self.Meta_prompt[3]}" + f"\n症状：{self.CASE['feature_content']}" + "\n诊断疾病：<disease></disease>" + f"{self.Meta_prompt[6]}"

        elif stage == 2: #reason
            prompt = f"{self.Meta_prompt[5]}" + f"{self.Example_prompt}" + f"{self.EXAMPLE}" + f"{self.Meta_prompt[3]}" + f"\n症状：{self.CASE['feature_content']}" + f"\n诊断疾病：{self.CASE['diseases']}" + "\n诊断依据：<reason></reason>" + f"{self.Meta_prompt[7]}"
            
        return prompt
    
    def data_construct(self, stage, diseases_num = 0):
        if stage == 0: #feature
            prompt = f"{self.Meta_prompt[8]}".format(diseases_num) + f"{self.FEATURE_EXAMPLE}" + f"{self.Meta_prompt[9]}"
        elif stage == 1: #diseases
            prompt = f"{self.Meta_prompt[4]}" + f"{self.Example_prompt}" + f"{self.DISEASE_EXAMPLE}" + f"{self.Meta_prompt[3]}" + f"\n症状：{self.CASE['feature_content']}" + "\n诊断疾病：<disease></disease>" + f"{self.Meta_prompt[6]}"
        elif stage == 2: #reason
            prompt = f"{self.Meta_prompt[5]}" + f"{self.Example_prompt}" + f"{self.EXAMPLE}" + f"{self.Meta_prompt[3]}" + f"\n症状：{self.CASE['feature_content']}" + f"\n诊断疾病：{self.CASE['diseases']}" + "\n诊断依据：<reason></reason>" + f"{self.Meta_prompt[7]}"
            
        return prompt


def extract_info(raw_response):
    """
    extract info from response 
    """
    pattern1 = r"<disease>(.*?)</disease>"
    pattern2 = r"<reason>(.*?)</reason>"
    match1 = re.search(pattern1, raw_response, re.DOTALL)
    match2 = re.search(pattern2, raw_response, re.DOTALL)
    
    if match1 and (match2 is None):
        diseases = match1.group(1).strip()
        return diseases, None
    elif (match1 is None) and match2:
        reasons = match2.group(1).strip()
        return None, reasons
    elif match1 and match2:
        diseases = match1.group(1).strip()
        reasons = match2.group(1).strip()
        return diseases, reasons
    else:
        return None, None

def process_string(input_str):

    if "；" not in input_str:
        cleaned_str = re.sub(r'^\d+[.\s、]\s*', '', input_str).strip()
        return cleaned_str
    
    items = input_str.split("；")

    cleaned_items = []
    for item in items:
        item = re.sub(r'^\d+[.\s、]\s*', '', item).strip()
        cleaned_items.append(item)

    if len(cleaned_items) == 1:
        return cleaned_items[0]

    processed_items = [f"{i + 1}. {item}" for i, item in enumerate(cleaned_items)]

    result = "；".join(processed_items)
    
    input_string = result.strip()
    
    if input_string.strip().endswith(";"):
        input_string = input_string[:-1].strip()  # 去掉最后一个分号
    
    if input_string and input_string[-1] == "." and input_string[-2].isdigit():
        last_dot_index = input_string.rfind(".")
        if last_dot_index > 0 and input_string[last_dot_index - 1].isdigit():
            input_string = input_string[:last_dot_index].strip()
    if input_string.endswith(";") or input_string.endswith("等"):
        input_string = input_string[:-1].strip()  # 去掉最后一个分号

    return input_string


def read_data(file):
    
    path = file + '.jsonl'
    with open(path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file.readlines()]
    
    return data        


def save_jsonl(datas, file_name):
    """
    add and save content
    data: {'id': record['id'], 'diseases': diseases, 'reason': reason}
    """
    file_path = file_name + '.jsonl'
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as output_file:
            if isinstance(datas, list) and all(isinstance(data, dict) for data in datas):
                for data in datas:
                    output_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            else:
                output_file.write(json.dumps(datas, ensure_ascii=False) + '\n')
    else:
        existing_data = read_data(file_name)
        if isinstance(datas, list) and len(datas) > 1:
            existing_data.extend(datas)
        else:
            existing_data.append(datas)
        with open(file_path, 'w', encoding='utf-8') as output_file:
            for data in existing_data:
                output_file.write(json.dumps(data, ensure_ascii=False) + '\n')
    print('\n', flush= True)
    print(f"Data saved in {file_name}.", flush= True)
    

if __name__ == "__main__":
    exam = read_data("datasets/20250214171329_提交示例")
    # save_jsonl(data[0], 'test')
    # print(exam)
    Prompt = Prompts(EXAMPLE = exam[:2], CASE=exam[-1])
    prompt = Prompt.two_stage(2)
    print(prompt)
    