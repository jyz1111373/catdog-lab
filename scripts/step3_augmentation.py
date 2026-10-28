"""第三步：数据增强的三组对照实验。

控制变量：同一模型（resnet18）、同一划分（seed=42）、同样轮数，只改"增强"与"预训练"：
    3-1  从零训练 + 无增强   —— 最差基线
    3-2  从零训练 + 有增强   —— 增强能不能救从零训练？
    3-3  预训练   + 有增强   —— 预训练与增强叠加，是否还能再涨？

注意：增强只作用于训练集，验证集始终用干净的 Resize+Normalize。

用法：
    python scripts/step3_augmentation.py --epochs 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

from catdog.config import TrainConfig                          # noqa: E402
from catdog.metrics import metrics_table_rows                  # noqa: E402
from catdog.pipeline import file_logger_for, run_experiment    # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="第三步：数据增强对照实验")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log = file_logger_for("step3")

    log.info("=" * 68)
    log.info("第三步：数据增强三组对照（同模型、同划分、同轮数）")
    log.info("=" * 68)

    base = dict(arch="resnet18", epochs=args.epochs,
                batch_size=args.batch_size, lr=args.lr, device=args.device)

    experiments = [
        ("step3a_scratch_noaug",
         TrainConfig(pretrained=False, freeze_backbone=False, augment=False, **base),
         "resnet18 从零训练 + 无增强"),
        ("step3b_scratch_aug",
         TrainConfig(pretrained=False, freeze_backbone=False, augment=True, **base),
         "resnet18 从零训练 + 数据增强"),
        ("step3c_pretrained_aug",
         TrainConfig(pretrained=True, freeze_backbone=True, augment=True, **base),
         "resnet18 预训练冻结 + 数据增强"),
    ]

    records = [run_experiment(name, cfg, note=note, log=log)
               for name, cfg, note in experiments]

    log.info("-" * 68)
    log.info("第三步横向对比：")
    for row in metrics_table_rows([r.to_dict() for r in records]):
        log.info("  %-24s 准确率 %.4f  精确率 %.4f  召回率 %.4f  F1 %.4f",
                 row["name"], row["accuracy"], row["precision"],
                 row["recall"], row["f1"])

    log.info("结论：增强通过制造'看起来不同'的样本逼模型学本质特征，")
    log.info("      在从零训练时提升明显；与预训练叠加后收益会变小（已经很强了）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
