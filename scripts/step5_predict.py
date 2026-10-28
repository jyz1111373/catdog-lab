"""第五步：加载训练好的权重，对单张图片做推理测试。

用法：
    python scripts/step5_predict.py                                  # 用最佳模型预测示例图
    python scripts/step5_predict.py --image datasets/cat.7.jpg
    python scripts/step5_predict.py --ckpt outputs/checkpoints/resnet18.pth --image ./my.jpg
    python scripts/step5_predict.py --dir datasets --limit 10        # 批量抽查
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

from catdog.config import CKPT_DIR, CLASSES, DATA_DIR, IMAGE_SUFFIXES  # noqa: E402
from catdog.engine import load_experiment_records                      # noqa: E402
from catdog.inference import load_predictor                            # noqa: E402
from catdog.metrics import metrics_table_rows                          # noqa: E402


def pick_best_checkpoint() -> Path:
    """从已保存的实验记录里挑准确率最高的权重；没有记录时退回任意 .pth。"""
    records = load_experiment_records()
    rows = metrics_table_rows(records)
    for row in rows:
        if row["checkpoint"]:
            p = Path(row["checkpoint"])
            if p.exists():
                return p
    candidates = sorted(CKPT_DIR.glob("*.pth"))
    if not candidates:
        raise SystemExit(
            "outputs/checkpoints 下还没有权重文件，请先运行：\n"
            "  python scripts/train.py --arch resnet18 --epochs 5")
    return candidates[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="第五步：单张 / 批量图片推理")
    p.add_argument("--ckpt", type=str, default=None, help="权重路径，默认自动选最佳")
    p.add_argument("--image", type=str, default=None, help="单张图片路径")
    p.add_argument("--dir", type=str, default=None, help="批量预测整个目录")
    p.add_argument("--limit", type=int, default=10, help="批量预测的最大张数")
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def print_result(path: str, res: dict, truth: str | None = None) -> None:
    tag = ""
    if truth:
        tag = "  ✓" if truth == res["label"] else f"  ✗(真实:{truth})"
    print(f"图片: {Path(path).name}")
    print(f"  预测: {res['emoji']} {res['label_cn']} ({res['label']})"
          f"  置信度 {res['confidence']:.2%}"
          f"  耗时 {res['inference_ms']:.0f}ms{tag}")
    print("  各类概率: " + " | ".join(
        f"{d['label_cn']} {d['prob']:.2%}" for d in res["all"]))


def main() -> int:
    args = parse_args()
    ckpt = Path(args.ckpt) if args.ckpt else pick_best_checkpoint()
    predictor = load_predictor(ckpt, device=args.device)

    print("=" * 68)
    print(f"第五步：单图推理测试 | 权重 {ckpt.name} | 架构 {predictor.arch}")
    print("=" * 68)

    # ---- 批量模式 ----
    if args.dir:
        d = Path(args.dir)
        files = [p for p in sorted(d.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]
        files = files[: args.limit]
        correct = 0
        for f in files:
            truth = CLASSES[0] if f.name.lower().startswith("cat") else CLASSES[1]
            res = predictor.predict(f)
            print_result(str(f), res, truth)
            correct += int(res["label"] == truth)
        if files:
            print("-" * 68)
            print(f"抽查 {len(files)} 张，正确 {correct} 张，准确率 {correct / len(files):.2%}")
        return 0

    # ---- 单张模式 ----
    image = args.image or str(DATA_DIR / "cat.1.jpg")
    if not Path(image).exists():
        raise SystemExit(f"图片不存在: {image}")
    res = predictor.predict(image)
    print_result(image, res, truth=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
