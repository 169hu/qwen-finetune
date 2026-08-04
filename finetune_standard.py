import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_XET_ENABLED'] = '0'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

# ----- 1. 加载模型（4-bit） -----
model_name = "Qwen/Qwen2-1.5B-Instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# ----- 2. LoRA -----
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# ----- 3. 数据准备 -----
data = [
    {"instruction": "翻译成英文", "input": "今天天气很好", "output": "The weather is nice today"},
    {"instruction": "翻译成英文", "input": "我喜欢编程", "output": "I love programming"},
    {"instruction": "翻译成英文", "input": "明天会更好", "output": "Tomorrow will be better"},
]
data = data * 50

def format_inst(x):
    return f"### 指令：{x['instruction']}\n### 输入：{x['input']}\n### 输出：{x['output']}"

formatted_texts = [format_inst(d) for d in data]
dataset = Dataset.from_dict({"text": formatted_texts})

def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# ----- 4. 训练 -----
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    max_steps=200,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    output_dir="./lora_output",
    report_to="none",
    save_steps=100,
    remove_unused_columns=False,   # 保留 labels
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

print("🚀 开始微调...")
trainer.train()
print("✅ 完成！")
model.save_pretrained("./lora_model")