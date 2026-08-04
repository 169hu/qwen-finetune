import torch
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # 强制离线模式，只从本地缓存加载
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ----- 加载基座模型（直接从本地缓存） -----
base_model_name = "Qwen/Qwen2-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    trust_remote_code=True,
    local_files_only=True  # 强制只从本地读取
)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    local_files_only=True,  # 强制只从本地读取
)

# ----- 加载你刚刚训练的 LoRA 插件 -----
model = PeftModel.from_pretrained(model, "./lora_model")
model.eval()

# ----- 测试翻译 -----
test_prompt = "### 指令：翻译成英文\n### 输入：我非常喜欢编程\n### 输出："
inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.1)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)