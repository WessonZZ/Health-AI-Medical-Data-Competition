

from openai import OpenAI
import time
import re
from volcenginesdkarkruntime import Ark
import requests


def get_api(api_path = 'api_key.txt'):
    """
    get api key from a txt file
    """
    with open(api_path, "r", encoding="utf-8") as file:
        api_key = file.read()

    pattern = r"\'(.*?)\'"
    match = re.findall(pattern, api_key, re.DOTALL)
    if match:
        return match
    else:
        raise ValueError("API key format does not match!")

def apply_Qwen2_5(content, api_key, temperature = 0.9, flag = 0):
    
    client = OpenAI(
        base_url='https://api.siliconflow.cn/v1',
        api_key= api_key
    )

    if flag == 1:
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct-128K",
            messages=[
                {"role": "system", "content": "你是一个专业的医疗助手，掌握了广泛的医疗知识。"},
                {"role": "user", 
                "content": content}
                    # "我有以下问诊症状，你能确定患者所患的疾病以及相应的诊断依据吗？\n症状：\n性别: 女\n年龄: 25\n主诉: 未避孕未怀孕3年\n现病史: 2018年起与丈夫有性生活，2020年起未避孕未怀孕至今3年，2021年12月登记结婚。LMP2023.7.2，我科多次阴B提示PCO，诊断PCOS。\n既往史: 无特殊\n个人史: \n过敏史: 无\n婚育史: \n流行病史: 暂无\n体格检查: 血压（收缩压\/舒张压）110\/60mmHg  \n  空腹血糖测定值：mmol\/L  \n  餐后血糖测定值：mmol\/L  \n  T：℃  \n  P：次\/分  \n  R：次\/分  \n  神志：   \n  呼吸：  \n  \n辅助检查: 2023-06-14，UU-DNA \n诊断的疾病请采用<disease></disease>形式给出。\n诊断依据请分点采用<reason>1. 2. 3.等</reason>形式给出。"}
            ],
            temperature= temperature
        )

        # 逐步接收并处理响应
        return response.choices[0].message.content.encode('utf-8').decode('utf-8')
    return content


def deepseek_vol(content, api, temp = 0.9, top_p = 0.7, flag = 0):

    client = OpenAI(
        api_key = api,
        base_url = "https://ark.cn-beijing.volces.com/api/v3",
    )

    # Non-streaming:
    if flag:
        completion = client.chat.completions.create(
            model = "deepseek-v3-241226",  # your model endpoint ID
            messages = [
                {"role": "system", "content": "你是一个专业的医疗助手，掌握了广泛的医疗知识。"},
                {"role": "user", "content": content},
            ],
            temperature= temp,
            top_p=top_p
        )
        return completion.choices[0].message.content
    return content

def Qwen72B_ali(content, api, temp = 0.9, top_p = 0.7, flag = 0):

    client = OpenAI(
        api_key = api,
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # Non-streaming:
    if flag:
        completion = client.chat.completions.create(
            model = "qwen2.5-72b-instruct",  # your model endpoint ID
            messages = [
                {"role": "system", "content": "你是一个专业的医疗助手，掌握了广泛的医疗知识。"},
                {"role": "user", "content": content},
            ],
            temperature= temp,
            top_p=top_p
        )
        return completion.choices[0].message.content
    return content


def deepseek_QWen_sl(content, api_key, temperature = 0.9, flag = 0):

    url = "https://api.siliconflow.cn/v1/chat/completions"

    payload = {
        # "model": "Qwen/Qwen2.5-72B-Instruct-128K",
        "model": "deepseek-ai/DeepSeek-V3",
        "messages": [
                {"role": "system", "content": "你是一个专业的医疗助手，掌握了广泛的医疗知识。"},
                {"role": "user", "content": content}
        ],
        "stream": False,
        "max_tokens": 1024,
        "stop": None,
        "temperature": temperature,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    if flag:
        response = requests.request("POST", url, json=payload, headers=headers)
        return response.text
    return content
