"""
BLEU 评估脚本：加载微调后的 LoRA 模型，在留出测试集上计算 BLEU-4。

用法:
    python evaluate.py                     # 评估 lora_model_improved（推荐）
    python evaluate.py --lora lora_model   # 评估标准训练结果
    python evaluate.py --base "Qwen/Qwen2-1.5B-Instruct"

说明:
    - 测试句均为训练数据之外的新句子，用于验证泛化能力。
    - BLEU-4 使用自实现（单参考 + 平滑），避免额外依赖。
"""
import os
import re
import math
import collections
import argparse

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ===== 留出测试集（中文 -> 英文，参考译文 golden）=====
# 前 8 条为短句（与训练分布接近），后 4 条为长句/需要泛化的句子，用于检验真实能力
TEST_SET = [
    ("我今天很开心", "I am very happy today"),
    ("他正在看一本书", "He is reading a book"),
    ("这个城市很美丽", "This city is very beautiful"),
    ("我们需要更多时间", "We need more time"),
    ("她喜欢唱歌和跳舞", "She likes singing and dancing"),
    ("春天是播种的季节", "Spring is the season for sowing"),
    ("知识改变命运", "Knowledge changes destiny"),
    ("坚持每天锻炼身体", "Keep exercising every day"),
    ("他历经千辛万苦，终于实现了自己的梦想", "He finally realized his dream after going through countless hardships"),
    ("尽管天气不好，他们还是按计划出发了", "Despite the bad weather, they set off as planned"),
    ("这本书深入浅出地讲解了机器学习的基本原理", "This book explains the basic principles of machine learning in an accessible way"),
    ("为了赶上截止日期，团队成员连续加班了整整一周", "To meet the deadline, the team worked overtime for a whole week"),
]


# ===== BLEU-4 实现（单参考 + 平滑）=====
def tokenize_en(text: str) -> list:
    return re.findall(r"\b[\w']+\b", text.lower())


def _ngrams(tokens: list, n: int) -> list:
    return [tuple(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1))]


def bleu_sentence(candidate: str, reference: str, max_n: int = 4) -> float:
    cand = tokenize_en(candidate)
    ref = tokenize_en(reference)
    cand_len, ref_len = len(cand), len(ref)

    # 长度惩罚：短译惩罚
    bp = 1.0
    if cand_len < ref_len and cand_len > 0:
        bp = math.exp(1 - ref_len / cand_len)
    elif cand_len == 0:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        cand_ng = collections.Counter(_ngrams(cand, n))
        ref_ng = collections.Counter(_ngrams(ref, n))
        clipped = sum((cand_ng & ref_ng).values())
        total = sum(cand_ng.values())
        # 平滑：分子分母各 +1（add-one），即使候选过短 total=0 也不会为 0
        precisions.append((clipped + 1) / (total + 1))

    log_sum = sum(math.log(p) for p in precisions) / max_n
    return bp * math.exp(log_sum)


# ===== 模型加载 =====
def load_model(lora_path: str, base_model: str):
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True, local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return model, tokenizer


def translate(model, tokenizer, text: str) -> str:
    prompt = f"### 指令：翻译成英文\n### 输入：{text}\n### 输出："
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=64, temperature=0.1, do_sample=False
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.split("### 输出：")[-1].strip()


def main():
    parser = argparse.ArgumentParser(description="BLEU 评估")
    parser.add_argument("--lora", default="./lora_model_improved", help="LoRA 权重目录")
    parser.add_argument("--base", default="Qwen/Qwen2-1.5B-Instruct", help="基座模型")
    args = parser.parse_args()

    print(f"🔄 加载模型: {args.base} + {args.lora}")
    model, tokenizer = load_model(args.lora, args.base)
    print("✅ 模型加载完成")

    print("\n🧪 翻译结果 + BLEU-4：\n")
    results = []
    for zh, ref in TEST_SET:
        pred = translate(model, tokenizer, zh)
        score = bleu_sentence(pred, ref)
        results.append((zh, ref, pred, score))
        print(f"中：{zh}")
        print(f"参考：{ref}")
        print(f"预测：{pred}")
        print(f"BLEU-4：{score:.4f}\n")

    avg = sum(r[3] for r in results) / len(results)
    print("=" * 46)
    print(f"平均 BLEU-4：{avg:.4f}（{len(results)} 条测试）")
    print("=" * 46)

    # 保存结果供 README/曲线使用
    import json
    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "lora": args.lora,
                "avg_bleu": round(avg, 4),
                "samples": [
                    {"zh": z, "ref": r, "pred": p, "bleu": round(s, 4)}
                    for z, r, p, s in results
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n📄 结果已保存到 eval_results.json")


if __name__ == "__main__":
    main()