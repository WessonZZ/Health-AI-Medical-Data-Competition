

from util import read_data
import json
import re

file = "datasets/result_v12/result_v121"
data = read_data(file)

id_list = []
data_new = []
length_reason = []
length_disease = []
char_reason = []
char_feature = []
ID = set()
for d in data:
    id = d['id']
    if d['reason'] == "":
        print(id)
    if d["diseases"] == "":
        print(id)
    if id in ID:
        print(id)
    ID.add(id)
    reason = d['reason']
    disease = d['diseases']
    feature = d['feature_content']
    length_disease.append(len(disease.split("；")))
    length_reason.append(len(reason.split("；")))
    char_reason.append(len(reason))
    char_feature.append(len(feature))
print(len(ID))
assert len(ID) == 5000

print(sum(length_reason)/len(length_reason), sum(length_disease)/len(length_disease),\
    sum(char_reason)/len(char_reason), sum(char_feature)/len(char_feature))


# def clean_text(text):
#     text = re.sub(r'\*', '', text)
#     text = re.sub(r"\-", '', text)
#     text = re.sub(r'\s*\n\s*', '\n', text)
    
#     match = re.search(r'性别', text)
#     if match:
#         text = text[match.start():]
#     else:
#         text = "" 

#     return text


import re

def format_text(text):
    # 替换顿号（、）为点（.）
    text = text.replace("、", ". ")
    
    # 在每个数字之前插入分号（；），但不包括第一个数字
    text = re.sub(r"(?<!^)(\d+)", r";\1", text)
    
    return text





file = "datasets/result_v11/result_v111"
data = read_data(file)
new_data = []
for id, d in enumerate(data):
    dd = {}
    dd['id'] = d['id']
    dd['diseases'] = d["diseases"].replace(". ", "、").replace("；", "")
    # if "、" in dd["diseases"]:
    #     dd['diseases'] = format_text(dd['diseases'])
    if "、" not in dd["diseases"]:
        dd['diseases'] = "1、" + dd["diseases"]
    
    dd["reason"] = d["reason"]
    feature = d["feature_content"]
    
    dd["feature_content"] = feature
    # reason = d['reason']
    # disease = d['diseases']
    # feature = d['feature_content']
    new_data.append(dd)

with open(file + "_1.jsonl", 'w', encoding='utf-8') as output_file:
    for d in new_data:
        output_file.write(json.dumps(d, ensure_ascii=False) + '\n')




