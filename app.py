"""Qwen 翻译模型 · 云端演示页（成果展示模式）

说明：QLoRA 微调 Qwen2-1.5B 的训练过程在本地 GPU 完成，推理需要加载完整模型
权重（数 GB），Streamlit Cloud 免费实例无法承载。因此本页改为「成果展示」：
  - 训练损失曲线（来自 smoke_history.json）
  - 微调后模型 BLEU 指标（来自 eval_results.json）
  - 中英翻译样例对照表
如需在线交互式翻译，请在本地运行完整版 app（加载 lora_model_improved 权重）。
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent

st.set_page_config(page_title="Qwen 微调成果展示", page_icon=":material/translate:", layout="wide")

st.title("Qwen2-1.5B 中英翻译 · QLoRA 微调成果", icon=":material/translate:")
st.caption(
    "模型：Qwen2-1.5B-Instruct ｜ 方法：QLoRA（4bit + LoRA）｜ 任务：中→英翻译"
    " ｜ 训练 + 评测均在本地 GPU 完成，本页为成果报告展示。"
)

# ---------------- 数据加载 ----------------
@st.cache_data
def load_data():
    smoke = json.loads((HERE / "smoke_history.json").read_text(encoding="utf-8"))
    eval_ = json.loads((HERE / "eval_results.json").read_text(encoding="utf-8"))
    return smoke, eval_


def main():
    try:
        smoke, eval_ = load_data()
    except Exception as e:
        st.error(f"数据文件读取失败：{e}")
        return

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("微调步数", f"{smoke.get('max_steps', '—')} step")
    col_b.metric("最终 Loss", f"{smoke['history'][-1]['loss']:.2f}")
    col_c.metric("平均 BLEU", f"{eval_.get('avg_bleu', '—')}")

    # ---------------- 训练损失曲线 ----------------
    st.subheader("训练 Loss 曲线（smoke 30 步）", icon=":material/show_chart:")
    hist = pd.DataFrame(smoke["history"])
    st.line_chart(hist.set_index("step")["loss"])

    # ---------------- 翻译样例 ----------------
    st.subheader("中英翻译样例（微调后模型）", icon=":material/science:")
    sample_df = pd.DataFrame(
        [
            {"中文": s["zh"], "参考译文": s["ref"],
             "模型输出": s["pred"], "BLEU": f'{s["bleu"]:.2f}'}
            for s in eval_.get("samples", [])
        ]
    )
    # 高亮后 3 条（具备难度的长难句），其余正常展示
    st.dataframe(sample_df, width="stretch", hide_index=True)

    st.markdown("""
---
### 关于本页面
本页用于在云上展示**微调成果数据**（可被招聘方直接访问）。

完整能力（交互式翻译、加载 LoRA 权重推理）请运行仓库内的本地脚本：
```bash
pip install -r requirements.txt   # 本地训练依赖
streamlit run app_full.py          # 或搜索引擎入口，加载本地模型权重后交互
```
**训练代码**：`finetune_standard.py`（标准 LoRA）/ `finetune_improved.py`（改进版）
**评测代码**：`evaluate.py` / `test_model.py`（BLEU + 样例输出）
""")


if __name__ == "__main__":
    main()