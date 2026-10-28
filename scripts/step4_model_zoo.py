"""第四步：调用 torchvision.models 里的成熟模型横向对比。

除了准确率，本脚本额外记录**参数量 / 权重体积 / 单图推理延迟**，
因为小数据集上三个模型准确率往往都在 0.9+，此时选型看的是速度与体积。

用法：
    python scripts/step4_model_zoo.py
    python scripts/step4_model_zoo.py --model resnet18 --model mobilenet_v3_small
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

from catdog.config import CKPT_DIR, TrainConfig                 # noqa: E402
from catdog.engine import benchmark_latency                     # noqa: E402
from catdog.metrics import metrics_table_rows                   # noqa: E402
from catdog.pipeline import file_logger_for, run_experiment      # noqa: E402
from catdog.utils import model_file_size_mb, resolve_device      # noqa: E402

DEFAULT_MODELS = ["resnet18", "mobilenet_v3_small", "shufflenet_v2_x1_0"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="第四步：torchvision 模型横向对比")
    p.add_argument("--model", action="append", default=None,
                   help=f"指定模型，可重复；默认 {DEFAULT_MODELS}")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log = file_logger_for("step4")
    models = args.model or DEFAULT_MODELS

    log.info("=" * 68)
    log.info("第四步：torchvision 预训练模型横向对比（冻结骨干，只训练分类头）")
    log.info("=" * 68)

    records = []
    for name in models:
        cfg = TrainConfig(arch=name, epochs=args.epochs, batch_size=args.batch_size,
                          lr=args.lr, pretrained=True, freeze_backbone=True,
                          augment=False, device=args.device)
        rec = run_experiment(f"step4_{name}", cfg,
                             note=f"torchvision {name} 迁移学习", log=log)

        # 补充工程指标：体积与推理延迟
        ckpt = CKPT_DIR / f"step4_{name}.pth"
        device = resolve_device(args.device)
        try:
            from catdog.inference import load_predictor
            predictor = load_predictor(ckpt, device=str(device))
            latency = benchmark_latency(predictor.model,
                                        image_size=predictor.image_size,
                                        device=device)
        except Exception as exc:
            log.warning("%s 延迟测量失败: %s", name, exc)
            latency = -1.0

        rec.metrics["latency_ms"] = round(latency, 2)
        rec.metrics["ckpt_size_mb"] = model_file_size_mb(ckpt)

        from catdog.engine import save_experiment_record
        save_experiment_record(rec)
        records.append(rec)

    log.info("-" * 68)
    log.info("第四步横向对比（按准确率降序）：")
    log.info("  %-22s %8s %10s %10s %10s", "模型", "准确率", "参数量", "体积MB", "延迟ms")
    for row in metrics_table_rows([r.to_dict() for r in records]):
        rec = next(r for r in records if r.name == row["name"])
        log.info("  %-22s %8.4f %10s %10.2f %10.2f",
                 row["name"], row["accuracy"],
                 f"{rec.metrics.get('num_params_total', 0):,}",
                 rec.metrics.get("ckpt_size_mb", 0.0),
                 rec.metrics.get("latency_ms", -1.0))

    log.info("结论：小数据集上准确率差距很小，此时应优先选**体积小、延迟低**的模型，")
    log.info("      这也是第六步要把模型部署出去时的选型依据。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
