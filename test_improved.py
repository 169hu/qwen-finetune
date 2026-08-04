import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "Qwen/Qwen2-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    trust_remote_code=True,
    local_files_only=True
)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    local_files_only=True,
)

# 加载改进后的 LoRA 模型
model = PeftModel.from_pretrained(model, "./lora_model_improved")
model.eval()

# 测试句子（全新的，不在训练数据中）
test_sentences = [
    "我爱学习",
    "今天是个好日子",
    "她写得一手好字",
    "明天我们要去爬山",
    "这个项目非常重要"
]

print("🧪 测试改进后的模型：\n")
for s in test_sentences:
    prompt = f"### 指令：翻译成英文\n### 输入：{s}\n### 输出："
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.1)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"中文：{s}")
    print(f"翻译：{result}\n")