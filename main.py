"""
Qwen2-1.5B 中英翻译微调项目 —— 统一入口。

用法:
    # 训练（默认 = 改进版）
    python main.py train
    python main.py train --mode improved [--max-steps 300 --lr 5e-5]
    python main.py train --mode standard [--max-steps 200 --lr 2e-4]

    # 快速冒烟训练（几十步，用于验证 + 生成 loss 曲线）
    python main.py train --smoke --mode improved --max-steps 30 --history .\smoke_history.json --lora-save-dir .\lora_smoke

    # 评估（BLEU-4，默认评估改进版）
    python main.py eval
    python main.py eval --lora .\lora_model

    # 绘制训练 loss 曲线
    python main.py plot [--history .\training_history.json] [--out loss_curve.png]

依赖: torch / transformers / peft / accelerate / bitsandbytes / datasets
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from finetune_improved import train_improved
from finetune_standard import train_standard


def cmd_train(args):
    """训练入口"""
    if args.smoke:
        # 冒烟模式：极小步数，快速验证 + 生成 demo loss 曲线
        print("🧪 冒烟模式：极小步数，主要验证流程 + 生成 loss 曲线")
        common = dict(
            max_steps=args.max_steps,
            learning_rate=args.lr,
            history_path=args.history,
            logging_steps=3,
        )
        if args.mode == "improved":
            train_improved(
                output_dir="./lora_output_smoke",
                lora_save_dir=args.lora_save_dir,
                **common,
            )
        else:
            train_standard(
                output_dir="./lora_output_smoke",
                lora_save_dir=args.lora_save_dir,
                **common,
            )
        return

    if args.mode == "improved":
        train_improved(max_steps=args.max_steps, learning_rate=args.lr,
                       lora_save_dir="./lora_model_improved", history_path=args.history)
    else:
        train_standard(max_steps=args.max_steps, learning_rate=args.lr,
                       lora_save_dir="./lora_model", history_path=args.history)


def cmd_eval(args):
    """评估入口：调用 evaluate.py 的评估逻辑"""
    import evaluate as ev

    print(f"🔄 加载模型: base + {args.lora}")
    model, tokenizer = ev.load_model(args.lora, args.base)
    print("✅ 模型加载完成")

    print("\n🧪 翻译结果 + BLEU-4：\n")
    results = []
    for zh, ref in ev.TEST_SET:
        pred = ev.translate(model, tokenizer, zh)
        score = ev.bleu_sentence(pred, ref)
        results.append((zh, ref, pred, score))
        print(f"中：{zh}\n参考：{ref}\n预测：{pred}\nBLEU-4：{score:.4f}\n")

    avg = sum(r[3] for r in results) / len(results)
    print("=" * 46)
    print(f"平均 BLEU-4：{avg:.4f}（{len(results)} 条测试）")
    print("=" * 46)

    import json
    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "lora": args.lora, "avg_bleu": round(avg, 4),
            "samples": [{"zh": z, "ref": r, "pred": p, "bleu": round(s, 4)}
                        for z, r, p, s in results]
        }, f, ensure_ascii=False, indent=2)
    print(f"\n📄 结果已保存到 eval_results.json")


def cmd_plot(args):
    """绘制 loss 曲线"""
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(args.history, "r", encoding="utf-8") as f:
        data = json.load(f)
    history = data["history"]
    steps = [h["step"] for h in history]
    losses = [h["loss"] for h in history]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, losses, marker="o", linestyle="-", color="#1664FF")
    plt.title(f"Training Loss Curve (mode={data.get('mode','?')}, max_steps={data.get('max_steps','?')})")
    plt.xlabel("Step")
    plt.ylabel("Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"📈 loss 曲线已保存到 {args.out}（{len(history)} 个采样点）")


def main():
    parser = argparse.ArgumentParser(description="Qwen2-1.5B 翻译微调统一入口")
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train", help="训练")
    p_train.add_argument("--mode", choices=["improved", "standard"], default="improved")
    p_train.add_argument("--max-steps", type=int, default=300)
    p_train.add_argument("--lr", type=float, default=5e-5)
    p_train.add_argument("--history", default="./training_history.json")
    p_train.add_argument("--smoke", action="store_true", help="冒烟快速验证")
    p_train.add_argument("--lora-save-dir", default="./lora_smoke")
    p_train.add_argument("--command", default="train", help=argparse.SUPPRESS)

    # eval
    p_eval = sub.add_parser("eval", help="评估 BLEU")
    p_eval.add_argument("--lora", default="./lora_model_improved")
    p_eval.add_argument("--base", default="Qwen/Qwen2-1.5B-Instruct")

    # plot
    p_plot = sub.add_parser("plot", help="绘制 loss 曲线")
    p_plot.add_argument("--history", default="./training_history.json")
    p_plot.add_argument("--out", default="loss_curve.png")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "plot":
        cmd_plot(args)


if __name__ == "__main__":
    main()