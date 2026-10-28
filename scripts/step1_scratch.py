"""第一步：手写 AlexNet，随机初始化，从零训练。

目的：建立"基线"。记住这个数字，后面每加一个技巧都要跟它比。

用法：
    python scripts/step1_scratch.py
    python scripts/step1_scratch.py --epochs 10 --lr 1e-4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402  把 src/ 加入模块搜索路径

from catdog.config import TrainConfig                      # noqa: E402
from catdog.metrics import format_metrics                  # noqa: E402
from catdog.pipeline import file_logger_for, run_experiment  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="第一步：手写 AlexNet 从零训练")
    p.add_argument("--epochs", type=int, default=30, help="训练轮数（默认 30）")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4, help="从零训练建议较小学习率")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--no-save", action="store_true", help="不保存权重，只看结果")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log = file_logger_for("step1")

    log.info("=" * 68)
    log.info("第一步：手写 AlexNet + 从零训练（无预训练、无增强）")
    log.info("=" * 68)

    cfg = TrainConfig(
        arch="alexnet_scratch",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        pretrained=False,      # 关键：随机初始化权重
        freeze_backbone=False, # 从零训练时全部参数都要学
        augment=False,
        device=args.device,
    )

    record = run_experiment("step1_scratch", cfg,
                            note="手写 AlexNet 从零训练（基线）",
                            save_ckpt=not args.no_save,
                            log=log)

    log.info("-" * 68)
    log.info("第一步结果：%s", format_metrics(record.metrics))
    log.info("结论：小数据集（80 张训练图）从零训练学不出通用视觉特征，")
    log.info("      准确率通常停留在 0.5~0.8，且波动大。带着这个数字进入第二步。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
