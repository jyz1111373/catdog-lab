"""第二步：在手写模型上加载 ImageNet 预训练权重（迁移学习）。

两个做法对比（控制变量：同一份数据划分、同样的评价指标）：
    A. 冻结卷积层，只训练分类头 —— 把 AlexNet 当"特征提取器"（默认，CPU 友好）
    B. 解冻全部层，lr=1e-5 全量微调 —— 让浅层特征也向本任务偏移（数据少时易过拟合）

用法：
    python scripts/step2_pretrained.py            # 只跑 A
    python scripts/step2_pretrained.py --finetune # A + B 对比
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

from catdog.config import TrainConfig                        # noqa: E402
from catdog.metrics import format_metrics, metrics_table_rows  # noqa: E402
from catdog.pipeline import file_logger_for, run_experiment    # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="第二步：加载 ImageNet 预训练权重")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--finetune", action="store_true", help="额外跑一组全量微调做对比")
    p.add_argument("--finetune-lr", type=float, default=1e-5)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log = file_logger_for("step2")

    log.info("=" * 68)
    log.info("第二步：手写 AlexNet + ImageNet 预训练权重（迁移学习）")
    log.info("=" * 68)

    records = []

    # ---- A：冻结卷积层，只训练分类头 ----
    cfg_freeze = TrainConfig(
        arch="alexnet_manual", epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, pretrained=True, freeze_backbone=True, augment=False,
        device=args.device,
    )
    rec_a = run_experiment("step2_pretrained_frozen", cfg_freeze,
                           note="AlexNet + ImageNet 预训练，冻结卷积层",
                           log=log)
    records.append(rec_a)

    # ---- B（可选）：解冻全部层做微调 ----
    if args.finetune:
        cfg_ft = TrainConfig(
            arch="alexnet_manual", epochs=args.epochs, batch_size=args.batch_size,
            lr=args.finetune_lr, pretrained=True, freeze_backbone=False, augment=False,
            device=args.device,
        )
        rec_b = run_experiment("step2_pretrained_finetune", cfg_ft,
                               note=f"AlexNet + 预训练，全量微调 lr={args.finetune_lr:g}",
                               log=log)
        records.append(rec_b)

    log.info("-" * 68)
    log.info("第二步对比：")
    for row in metrics_table_rows([r.to_dict() for r in records]):
        log.info("  %-28s 准确率 %.4f  F1 %.4f  训练 %.1fs",
                 row["name"], row["accuracy"], row["f1"], row["train_seconds"])

    log.info("结论：小数据集上，'知识从哪来'比'训练多久'重要得多——")
    log.info("      灌入 ImageNet 权重后准确率通常从 0.6~0.8 直接跳到 0.9+。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
