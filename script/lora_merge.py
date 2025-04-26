


from unsloth import FastLanguageModel
from peft import PeftModel
import torch

# 模型名称
base_model_path = "/root/Qwen2.5-7B-Instruct"
lora_path = "output/model_train-20250308-113610/checkpoint-316"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_path,
    max_seq_length=1024,
    dtype=torch.float16,
    load_in_4bit=False,
    gpu_memory_utilization=0.9
)

# 加载 LoRA 适配器
lora_model = PeftModel.from_pretrained(model, lora_path)


merged_model = lora_model.merge_and_unload()
merged_model_path = "output/model_train-20250308-113610/merged_model"  # 替换为保存路径
merged_model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)