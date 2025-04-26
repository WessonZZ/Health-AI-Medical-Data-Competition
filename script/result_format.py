
import os
import json
import re
from util import read_data, save_jsonl, process_string


file_name = 'result_v12'
new_file = 'result_v121'
data_path = os.path.join('datasets', 'result_v12', file_name)
new_path = os.path.join('datasets', 'result_v12', new_file)
data = read_data(data_path)
# new_data = read_data(new_path)


def extract_disease_and_evidence(text):
    # 提取疾病名称
    diseases = []
    disease_pattern = r"<disease>(.*)</disease>"
    disease_match = re.search(disease_pattern, text, re.DOTALL)
    if disease_match:
        diseases = disease_match.group(1)
    elif len(diseases) == 0:
        disease_pattern = r"<diagnosis>(.*)</diagnosis>"
        disease_match = re.search(disease_pattern, text, re.DOTALL)
        if disease_match:
            diseases = disease_match.group(1) # 分割疾病名称
        elif len(diseases) == 0:
            disease_pattern = r"<诊断>(.*)</诊断>"
            disease_match = re.search(disease_pattern, text, re.DOTALL)
            if disease_match:
                diseases = disease_match.group(1) # 分割疾病名称
            elif len(diseases) == 0:
                disease_pattern = r"疾病：(.*?)\n"
                disease_match = re.search(disease_pattern, text, re.DOTALL)
                if disease_match:
                    diseases = disease_match.group(1) # 分割疾病名称
                    # diseases = [d.strip() for d in diseases]  # 去除多余空格
                elif len(diseases) == 0:
                    disease_pattern = r"诊断疾病：\s*(.*)\n"
                    disease_match = re.search(disease_pattern, text, re.DOTALL)
                    if disease_match:
                        diseases = disease_match.group(1) # 分割疾病名称
                    elif len(diseases) == 0:
                        disease_pattern = r"诊断：\s*(.*)\n"
                        disease_match = re.search(disease_pattern, text, re.DOTALL)
                        if disease_match:
                            diseases = disease_match.group(1) # 分割疾病名称
                        elif len(diseases) == 0:
                            disease_pattern = r"诊断:\s*(.*)"
                            disease_match = re.search(disease_pattern, text, re.DOTALL)
                            if disease_match:
                                diseases = disease_match.group(1) # 分割疾病名称
    else:
        diseases = []

    # 提取诊断依据
    evidence_pattern = r"<reason>(.*)</reason>"
    evidence_match = re.search(evidence_pattern, text, re.DOTALL)
    evidences = []
    if evidence_match:
        evidences = evidence_match.group(1)
    elif len(evidences) == 0:
        evidence_pattern = r"诊断依据为：\s*(.*)"
        evidence_match = re.search(evidence_pattern, text, re.DOTALL)
        evidences = []
        if evidence_match:
            evidences = evidence_match.group(1)
        elif len(evidences) == 0:
            evidence_pattern = r"<诊断依据>(.*)</诊断依据>"
            evidence_match = re.search(evidence_pattern, text, re.DOTALL)
            evidences = []
            if evidence_match:
                evidences = evidence_match.group(1)
            elif len(evidences) == 0:
                evidence_pattern = r"<诊断依据>(.*)"
                evidence_match = re.search(evidence_pattern, text, re.DOTALL)
                evidences = []
                if evidence_match:
                    evidences = evidence_match.group(1)
                elif len(evidences) == 0:
                    evidence_pattern = r"诊断依据：\s*(.*)"
                    evidence_match = re.search(evidence_pattern, text, re.DOTALL)
                    if evidence_match:
                        evidences = evidence_match.group(1)
                    elif len(evidences) == 0:
                        evidence_pattern = r"<reason>\s*(.*)</reason>"
                        evidence_match = re.search(evidence_pattern, text, re.DOTALL)
                        if evidence_match:
                            evidences = evidence_match.group(1)
                        elif len(evidences) == 0:
                            evidence_pattern = r"依据如下：\s*(.*)"
                            evidence_match = re.search(evidence_pattern, text, re.DOTALL)
                            if evidence_match:
                                evidences = evidence_match.group(1).strip()
    else:   
        evidences = []

    return diseases, evidences

new_data = []
for d in data:
    new_d = {}
    new_d['id'] = d['id']
    # print(repr(d['diseases']))
    diseases, reason = extract_disease_and_evidence(d['diseases'])
    new_d['diseases'] = process_string(diseases) if len(diseases) > 0 else diseases 
    new_d['reason'] = reason
    new_d['feature_content'] = d['feature_content']
    
    if new_d["diseases"] == []:
        diseases, reason = extract_disease_and_evidence(d['feature_content'])
        new_d["diseases"] = diseases.strip() if len(diseases) > 0 else []
    
    # if "依据" in diseases:
    #     print(d['id'])
    if "\n" in new_d["diseases"]:
        new_d["diseases"] = process_string(new_d["diseases"].split("\n")[0])
    
    if new_d["reason"] != []:
        if new_d["reason"].startswith("\n"):
            # print(new_d['id'])
            new_d["reason"] = new_d["reason"][2:]
        if new_d["reason"].endswith("\n"):
            # print(new_d['id'])
            new_d["reason"] = new_d["reason"][:-1]
        if new_d["reason"].startswith("\n"):
            # print(new_d['id'])
            new_d["reason"] = new_d["reason"][2:]
    
    new_data.append(new_d)

with open(new_path + '.jsonl', 'w', encoding='utf-8') as output_file:
    for d in new_data:
        if d['reason'] == [] or d['diseases'] == [] or d['diseases'] == '无':
            print(d['id'])
        output_file.write(json.dumps(d, ensure_ascii=False) + '\n')