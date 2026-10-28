"""通用训练入口：一次命令完成"选模型 → 训练 → 评估 → 保存"。

Web 后端的"开始训练"按钮调用的就是本文件背后的 `pipeline.run_experiment`，
所以命令行与网页两条路径完全等价。

用法示例：
    python scripts/train.py --arch resnet18 --epochs 5 --augment
    python scripts/train.py --arch mobilenet_v3_small --epochs 8 --lr 5e-4
    python scripts/train.py --arch alexnet_scratch --epochs 30 --lr 1e-4 --pretrained no
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

from catdog.config import TrainConfig                          # noqa: E402
from catdog.metrics import format_metrics                      # noqa: E402
from catdog.models import available_archs                      # noqa: E402
from catdog.pipeline import file_logger_for, run_experiment     # noqa: E402
from catdog.utils import format_seconds                        # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="通用训练入口",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--arch", type=str, default="resnet18",
                   choices=available_archs(), help="模型架构")
    p.add_argument("--name", type=str, default=None, help="实验名，默认 <arch>")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--pretrained", type=str, default="yes", choices=["yes", "no"])
    p.add_argument("--freeze", type=str, default="yes", choices=["yes", "no"],
                   help="预训练时是否冻结骨干网络")
    p.add_argument("--augment", action="store_true", help="训练集启用数据增强")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--no-save", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    name = args.name or args.arch
    log = file_logger_for(f"train_{name}")

    cfg = TrainConfig(
        arch=args.arch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pretrained=(args.pretrained == "yes"),
        freeze_backbone=(args.freeze == "yes"),
        augment=args.augment,
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        device=args.device,
    )

    log.info("=" * 68)
    log.info("通用训练：%s", cfg.summary())
    log.info("=" * 68)

    record = run_experiment(name, cfg, note=cfg.summary(),
                            save_ckpt=not args.no_save, log=log)

    log.info("-" * 68)
    log.info("实验「%s」完成，用时 %s", record.name, format_seconds(record.train_seconds))
    log.info("最终指标：%s", format_metrics(record.metrics))
    if record.checkpoint:
        log.info("权重已保存：%s", record.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
