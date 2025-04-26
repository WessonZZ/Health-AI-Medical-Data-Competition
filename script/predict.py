

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
from tqdm import tqdm
import os



merged_model_path = "output/model_train-20250308-113610/merged_model"  # 替换为保存路径


llm = LLM(model=merged_model_path, trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(merged_model_path)
sampling_params = SamplingParams(temperature = 0.9, top_p = 0.7, max_tokens = 1024)


# FastLanguageModel.for_inference(model)
# 读取数据
with open('datasets/20250208181531_camp_data_step_1_without_answer.jsonl', 'r', encoding='utf-8') as file:
    data = [json.loads(line) for line in file.readlines()]

save_path = 'datasets/result_v7.jsonl'
data_id = []
if os.path.exists(save_path):
    with open(save_path, 'r', encoding='utf-8') as file:
        existing_data = [json.loads(line) for line in file.readlines()]
    data_id = [d.get('id', None) for d in existing_data]
else:
    existing_data = []
    
# 处理数据并生成结果
with open('datasets/result_v7.jsonl', 'w', encoding='utf-8') as output_file:
    if len(existing_data) > 0:
        for d in existing_data:
            output_file.write(json.dumps(d, ensure_ascii=False) + '\n')
    for i in tqdm(range(300)):
        records = data[i * 10: (i + 1) * 10]
        texts = []
        for re in records:
            id = re['id']
            if id not in data_id:
                feature_content = re['feature_content']
            
            # 构建消息列表
                prompt = [
                    {"role": "system", "content": "你是一个专业的医疗助手，掌握了广泛的医疗知识。现提供给你一些问诊症状，请根据你所掌握的医学知识尽可能确定患者所患的疾病以及相应的诊断依据。\n注意格式规范：如果有多个疾病则用阿拉伯数字编号，如\"<disease>1. 疾病；...</disease>\"，诊断依据以\"<reason>1. 依据1；2.依据2...</reason>\"格式表述。"},
                    {"role": "user", "content": f"请诊断：\n{feature_content}"}
                ]
                text = tokenizer.apply_chat_template(
                        prompt,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
            
                texts.append(text)
    
        if len(texts) > 0:
            outputs = llm.generate(texts, sampling_params)

            responses = []
            for output in outputs:
                generated_text = output.outputs[0].text
                responses.append(generated_text)

            for re1, re2 in zip(records,responses):
                id = re1['id']
                print(f"\nid: {id}\n症状：\n{re1['feature_content']}\n诊断：\n{re2}")
                # 写入结果
                result = {'id': id, 'diseases': re2, 'feature_content': re1['feature_content']}
                output_file.write(json.dumps(result, ensure_ascii=False) + '\n')