"""Qwen 翻译模型 · 本地完整版（交互式推理）

加载 QLoRA 微调后的 LoRA 权重 + 基座模型，交互式中→英翻译。
注意：需要本地 GPU（或大内存 CPU），云端部署请使用 app.py（成果展示模式）。
"""
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# 页面配置
st.set_page_config(
    page_title="Qwen 翻译模型演示",
    page_icon="🌐",
    layout="centered"
)

st.title("🌐 Qwen2-1.5B 翻译模型演示")
st.caption("QLoRA 微调 · 中英翻译")


# 加载模型（使用缓存）
@st.cache_resource
def load_model():
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

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
    model = PeftModel.from_pretrained(model, "./lora_model_improved")
    model.eval()
    return model, tokenizer


# 加载模型（带加载状态）
with st.spinner("🔄 正在加载模型，请稍候..."):
    try:
        model, tokenizer = load_model()
        st.success("✅ 模型加载成功！")
    except Exception as e:
        st.error(f"❌ 模型加载失败：{e}")
        st.stop()

# 输入区
user_input = st.text_area(
    "请输入要翻译的中文：",
    placeholder="例如：今天天气很好",
    height=100
)

col1, col2 = st.columns([1, 3])
with col1:
    max_new = st.slider("生成长度", 16, 256, 64, step=16)
with col2:
    temperature = st.slider("温度", 0.1, 1.5, 0.7, step=0.1)

if st.button("翻译", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("请输入中文内容")
    else:
        messages = [
            {"role": "system", "content": "You are a professional translator."},
            {"role": "user", "content": f"将下面的中文翻译成英文：{user_input}"}
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
            )
        answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:],
                                 skip_special_tokens=True)
        st.success("**翻译结果：**")
        st.markdown(answer)