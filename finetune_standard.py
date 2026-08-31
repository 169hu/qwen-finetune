"""
标准版微调：QLoRA(4-bit) + LoRA，使用内联小样本数据（示范用）。

可被 main.py 作为训练入口调用，也可独立运行:
    python finetune_standard.py --max-steps 200
"""
import os
import json
import argparse

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_XET_ENABLED"] = "0"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
    BitsAndBytesConfig, Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset


def format_inst(x):
    return f"### 指令：{x['instruction']}\n### 输入：{x['input']}\n### 输出：{x['output']}"


def train_standard(
    max_steps: int = 200,
    learning_rate: float = 2e-4,
    output_dir: str = "./lora_output",
    lora_save_dir: str = "./lora_model",
    history_path: str = "./training_history.json",
    model_name: str = "Qwen/Qwen2-1.5B-Instruct",
    logging_steps: int = 10,
):
    """标准版训练（内联 3 句 × 50 次扩充），保存 loss 历史"""
    # 1. 数据（内联小样本）
    data = [
        {"instruction": "翻译成英文", "input": "今天天气很好", "output": "The weather is nice today"},
        {"instruction": "翻译成英文", "input": "我喜欢编程", "output": "I love programming"},
        {"instruction": "翻译成英文", "input": "明天会更好", "output": "Tomorrow will be better"},
    ]
    data = data * 50
    formatted_texts = [format_inst(d) for d in data]
    dataset = Dataset.from_dict({"text": formatted_texts})

    # 2. 加载模型（4-bit）
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

    # 3. LoRA
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

    # 4. Tokenize
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"], truncation=True, max_length=512, padding="max_length"
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # 5. 训练
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_steps=max_steps,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=logging_steps,
        output_dir=output_dir,
        report_to="none",
        save_steps=100,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print(f"🚀 开始微调（{max_steps}步，学习率{learning_rate}）...")
    trainer.train()
    print("✅ 完成！")

    # 6. 保存 LoRA 权重
    model.save_pretrained(lora_save_dir)
    print(f"✅ LoRA 权重保存到 {lora_save_dir}")

    # 7. 保存 loss 历史
    history = [
        {"step": h["step"], "loss": h["loss"]}
        for h in trainer.state.log_history
        if "loss" in h
    ]
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(
            {"mode": "standard", "max_steps": max_steps, "history": history},
            f, ensure_ascii=False, indent=2,
        )
    print(f"📈 loss 历史({len(history)} 条)保存到 {history_path}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output-dir", default="./lora_output")
    parser.add_argument("--lora-save-dir", default="./lora_model")
    parser.add_argument("--history", default="./training_history.json")
    args = parser.parse_args()
    train_standard(
        max_steps=args.max_steps,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        lora_save_dir=args.lora_save_dir,
        history_path=args.history,
    )