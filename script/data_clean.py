
import json
import os
import re

from util import save_jsonl


def read_data(file):
    
    path = file + '.jsonl'
    with open(path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file.readlines()]
        
    for d in data:
        id = d['id']
        reason = d['reason']
        diseases = d['diseases']
        feature = d['feature_content']
    
    print(id, reason, diseases, feature)
    
    # return data  

# read_data('datasets/result_v2/result_v21')


def extract_gender_age(text):
    # 匹配性别
    gender_pattern = r"患者(?:[^男女\n]*?)(?P<gender>[男女])"
    gender_match = re.search(gender_pattern, text)
    gender = gender_match.group("gender") if gender_match else ""

    # 匹配年龄
    age_pattern = r"，(?P<age>\d+)岁"
    age_match = re.search(age_pattern, text)
    age = int(age_match.group("age")) if age_match else ""
    
    # 提取辅助检查内容
    # 辅助检查内容通常在“辅助检查：”之后，直到文本结束或遇到下一个段落
    aux_check_pattern = r"辅助检查：(?P<aux_check>.*?)(?=\n|$)"
    aux_check_match = re.search(aux_check_pattern, text, re.DOTALL)
    if aux_check_match:
        aux_check = aux_check_match.group("aux_check").strip()
    else:
        aux_check = ""

    return gender, age, aux_check
    
def extract_content(text):
    
    pattern1 = r"查体：(.+?)(?=辅助检查|$)"
    pattern2 = r"[^。！？]*[过敏][^。！？]*[。！？]"
    match1 = re.search(pattern1, text, re.DOTALL)
    match2 = re.findall(pattern2, text)
    
    if match1:
        content1 = match1.group(1).strip()
    else:
        content1 = None
        
    if match2:
        content2 = " ".join(match2).strip()
    else:
        content2 = None

    return content1, content2

def format_diagnoses(diseases):
    # 使用正则表达式匹配每个诊断项
    pattern = r"(\d+\..+?)(?=\d+\.|$)"
    matches = re.findall(pattern, diseases, re.DOTALL)
    
    # 如果有多个诊断项，用分号分隔
    if len(matches) > 1:
        formatted_text = "；".join(matches)
    # 如果只有一个诊断项，去掉序号
    elif len(matches) == 1:
        formatted_text = matches[0].split('.', 1)[1].strip()
    # 如果没有匹配到任何诊断项，返回原始文本
    else:
        formatted_text = diseases.strip()
    
    return formatted_text



def format_text(text):
    # 提取现病史
    pattern_history = r"现病史，(.*?)(?=既往史)"
    match_history = re.search(pattern_history, text, re.DOTALL)
    history = match_history.group(1).strip() if match_history else ""

    # 提取既往史
    pattern_past = r"既往史，(.*?)(?=个人史)"
    match_past = re.search(pattern_past, text, re.DOTALL)
    past = match_past.group(1).strip() if match_past else ""

    # 提取个人史
    pattern_personal = r"个人史，(.*?)(?=过敏史)"
    match_personal = re.search(pattern_personal, text, re.DOTALL)
    personal = match_personal.group(1).strip() if match_personal else ""

    # 提取过敏史
    pattern_allergy = r"过敏史，(.*?)(?=婚育史)"
    match_allergy = re.search(pattern_allergy, text, re.DOTALL)
    allergy = match_allergy.group(1).strip() if match_allergy else ""

    # 提取婚育史
    pattern_marital = r"婚育史，(.*?)(?=流行病史)"
    match_marital = re.search(pattern_marital, text, re.DOTALL)
    marital = match_marital.group(1).strip() if match_marital else ""

    # 提取流行病史
    pattern_epidemiology = r"流行病史，(.*?)(?=体格检查)"
    match_epidemiology = re.search(pattern_epidemiology, text, re.DOTALL)
    epidemiology = match_epidemiology.group(1).strip() if match_epidemiology else ""


    # 组合成所需格式
    formatted_text = (
        f"现病史: {history}\n"
        f"既往史: {past}\n"
        f"个人史: {personal}\n"
        f"过敏史: {allergy}\n"
        f"婚育史: {marital}\n"
        f"流行病史: {epidemiology}\n"
    )

    return formatted_text

def extract_complaint(text):
    # 使用正则表达式匹配“主诉”之后的内容
    pattern = r"主\s*诉：(.*?)(?=\n|$)"
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        complaint = match.group(1).strip()
    else:
        complaint = None
    
    return complaint

#####################
#处理训练数据
path = "datasets/extra_data"
file_name = "ICD-Coding-train" + ".json"

file = os.path.join(path, file_name)
with open(file, 'r', encoding='utf-8') as file:
    data = json.load(file)

file_name = "ICD-Coding-test-A" + ".json"

file = os.path.join(path, file_name)
with open(file, 'r', encoding='utf-8') as file:
    data_test = json.load(file)

data.extend(data_test)

data_new = []
for id, d in enumerate(data):
    idx = id
    gender = d.get("性别", "")
    age = d.get("年龄", "") #需提取数字
    zhusu = d.get("主诉", "")
    xianbingshi = d.get("现病史", "")
    jiwangshi = d.get("既往史", "")
    gerenshi = d.get("个人史","")
    hunyushi = d.get("婚姻史", "")
    liuxingbingshi = d.get("流行病史", "")
    
    ruyuan_situation = d.get("入院情况", "")
    gender, age, aux_check = extract_gender_age(ruyuan_situation) #性别，年龄，辅助检查
    physical_check, guominshi = extract_content(ruyuan_situation) #体格检查
    disease = d.get("入院诊断", "")
    disease = format_diagnoses(disease)
    
    text = f"性别：{gender}\n年龄：{age}\n主诉：{zhusu}\n现病史：{xianbingshi}\n既往史：{jiwangshi}\n个人史：{gerenshi}\n过敏史：{guominshi}\n婚育史：{hunyushi}\n流行病史：{liuxingbingshi}\n体格检查：{physical_check}\n辅助检查：{aux_check}"
    data_new.extend([{"id": idx, "reason": None, "diseases": disease, "feature_content": text}])


file_name = "TCM-TBOSD-train" + ".json"

file = os.path.join(path, file_name)
with open(file, 'r', encoding='utf-8') as file:
    data = json.load(file)

file_name = "TCM-TBOSD-test-A" + ".json"

file = os.path.join(path, file_name)
with open(file, 'r', encoding='utf-8') as file:
    data_test = json.load(file)

data.extend(data_test)


data_new1 = []
for id, d in enumerate(data):
    idx = id + 1000
    gender = d.get("性别", "")
    age = d.get("年龄", "")[:-1] #需提取数字
    zhusu = extract_complaint(d.get("主诉", ""))
    
    disease_situation = d.get("病史", "")
    
    physical_check = d.get("体格检查", "")
    aux_check = d.get("辅助检查", "")
    
    disease = d.get("疾病", "")
    disease = format_diagnoses(disease)
    
    history = format_text(disease_situation)
    text = f"性别：{gender}\n年龄：{age}\n主诉：{zhusu}\n{history}\n体格检查：{physical_check}\n辅助检查：{aux_check}"
    data_new1.extend([{"id": idx, "reason": None, "diseases": disease, "feature_content": text}])
# print(data_new[1])


data_new.extend(data_new1)
print(len(data_new))
save_jsonl(data_new, 'datasets/extra_data/training_data')
