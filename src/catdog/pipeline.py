"""实验流水线：把"配置 -> 数据 -> 模型 -> 训练 -> 评估 -> 落盘"串成一次调用。

有了这一层，CLI 脚本和 Web 后端调用的是**同一个函数**，
从根本上避免"网页上训练出来的模型和命令行跑的不是一回事"。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict

import torch

from .config import CKPT_DIR, ExperimentRecord, TrainConfig
from .data import build_loaders
from .engine import Trainer, evaluate, save_experiment_record
from .models import build_model
from .utils import default_log_file, format_seconds, get_logger, resolve_device, seed_everything

logger = get_logger("catdog.pipeline")


def run_experiment(name: str,
                   cfg: TrainConfig,
                   note: str = "",
                   save_ckpt: bool = True,
                   on_epoch: Callable[[Dict[str, Any]], None] | None = None,
                   log=None) -> ExperimentRecord:
    """跑完一次完整实验。

    Args:
        name: 实验名，同时作为权重文件名与指标文件名。
        cfg: 训练配置。
        note: 一句话实验说明（会显示到前端对比表里）。
        save_ckpt: 是否保存最佳权重到 outputs/checkpoints/<name>.pth。
        on_epoch: 每轮回调，用于把进度推给前端。

    Returns:
        ExperimentRecord，包含完整指标、历史曲线与权重路径。
    """
    log = log or logger
    seed_everything(cfg.seed)

    # 1) 数据
    bundle = build_loaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        augment=cfg.augment,
        image_size=cfg.image_size,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    log.info("[%s] 数据就绪：%s", name, bundle.describe())

    # 2) 模型
    model = build_model(cfg.arch, pretrained=cfg.pretrained,
                        freeze_backbone=cfg.freeze_backbone)

    # 3) 训练
    ckpt_path = (CKPT_DIR / f"{name}.pth") if save_ckpt else None
    trainer = Trainer(model, bundle.train_loader, bundle.val_loader, cfg,
                      checkpoint_path=ckpt_path, on_epoch=on_epoch, log=log)
    result = trainer.fit()

    # 4) 用"最佳权重"再评估一次，作为该实验的最终成绩
    if ckpt_path and ckpt_path.exists():
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["state_dict"])
    device = resolve_device(cfg.device)
    final_ev = evaluate(model, bundle.val_loader, device)
    final_metrics = final_ev.to_metrics()
    final_metrics["val_loss"] = round(final_ev.loss, 5)

    # 5) 落盘
    record = ExperimentRecord(
        name=name,
        arch=cfg.arch,
        config=cfg.to_dict(),
        metrics=final_metrics,
        history=result.history,
        checkpoint=str(ckpt_path) if ckpt_path else None,
        train_seconds=round(result.train_seconds, 2),
        note=note or cfg.summary(),
    )
    record.metrics["best_accuracy"] = round(result.best_accuracy, 4)
    record.metrics["num_params_total"] = result.num_params_total
    record.metrics["num_params_trainable"] = result.num_params_trainable
    save_experiment_record(record)

    log.info("[%s] 实验完成，验证集指标已保存（%s）", name, format_seconds(result.train_seconds))
    return record


def run_many(experiments: Dict[str, TrainConfig],
             notes: Dict[str, str] | None = None,
             on_epoch: Callable[[str, Dict[str, Any]], None] | None = None) -> Dict[str, ExperimentRecord]:
    """按顺序跑多组实验，用于横向对比（第三步、第四步）。"""
    notes = notes or {}
    results: Dict[str, ExperimentRecord] = {}
    started = time.perf_counter()
    for name, cfg in experiments.items():
        callback = None
        if on_epoch:
            callback = (lambda info, n=name: on_epoch(n, info))
        results[name] = run_experiment(name, cfg, note=notes.get(name, ""),
                                       on_epoch=callback)
    logger.info("共 %d 组实验，总耗时 %s", len(results),
                format_seconds(time.perf_counter() - started))
    return results


def file_logger_for(name: str):
    """为每个脚本生成带文件的 logger，便于事后回看。"""
    return get_logger(f"catdog.{name}", log_file=default_log_file(f"{name}.log"))
