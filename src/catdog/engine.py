"""训练引擎：评估、训练循环、最佳模型保存、实验记录落盘。

训练与评价指标计算被彻底解耦：
    - `evaluate()` 只负责"跑一遍数据集，产出预测与概率"；
    - `compute_metrics()` 只负责"由预测算指标"；
    - `Trainer.fit()` 只负责"按 epoch 组织流程并回调"。
因此无论是脚本、Web 后端还是单元测试，都可以单独复用其中任何一环。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import CKPT_DIR, CLASSES, IMAGE_SIZE, METRIC_DIR, ExperimentRecord, TrainConfig
from .metrics import compute_metrics, format_metrics
from .utils import (
    count_all_parameters,
    count_parameters,
    get_logger,
    resolve_device,
    seed_everything,
)

logger = get_logger("catdog.engine")


# --------------------------------------------------------------------------- #
# 评估
# --------------------------------------------------------------------------- #
@dataclass
class EvalOutput:
    """一次前向评估的原始产物。"""

    y_true: List[int]
    y_pred: List[int]
    y_prob: List[List[float]]
    loss: float

    def to_metrics(self) -> Dict[str, Any]:
        return compute_metrics(self.y_true, self.y_pred, self.y_prob)


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             device: torch.device,
             loss_fn: nn.Module | None = None) -> EvalOutput:
    """在给定 DataLoader 上跑一遍模型，收集真实标签、预测标签与概率。"""
    model.eval()
    loss_fn = loss_fn or nn.CrossEntropyLoss()

    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[List[float]] = []
    total_loss, total_n = 0.0, 0

    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        logits = model(X)
        loss = loss_fn(logits, Y)
        probs = torch.softmax(logits, dim=1)

        total_loss += loss.item() * X.size(0)
        total_n += X.size(0)
        y_true.extend(Y.cpu().tolist())
        y_pred.extend(probs.argmax(dim=1).cpu().tolist())
        y_prob.extend(probs.cpu().tolist())

    return EvalOutput(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        loss=total_loss / max(total_n, 1),
    )


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #
@dataclass
class TrainResult:
    """训练结果：既给人看（日志），也给程序用（JSON / 前端）。"""

    arch: str
    history: List[Dict[str, Any]] = field(default_factory=list)
    best_accuracy: float = 0.0
    best_epoch: int = 0
    final_metrics: Dict[str, Any] = field(default_factory=dict)
    best_metrics: Dict[str, Any] = field(default_factory=dict)
    checkpoint: str | None = None
    train_seconds: float = 0.0
    num_params_total: int = 0
    num_params_trainable: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arch": self.arch,
            "history": self.history,
            "best_accuracy": self.best_accuracy,
            "best_epoch": self.best_epoch,
            "final_metrics": self.final_metrics,
            "best_metrics": self.best_metrics,
            "checkpoint": self.checkpoint,
            "train_seconds": self.train_seconds,
            "num_params_total": self.num_params_total,
            "num_params_trainable": self.num_params_trainable,
        }


class Trainer:
    """训练器：封装 epoch 循环、验证、最佳权重保存与进度回调。

    Args:
        model: 待训练模型（requires_grad 应已由 models.build_model 设置好）。
        train_loader / val_loader: 数据。
        cfg: TrainConfig。
        checkpoint_path: 最佳权重保存路径，None 表示不保存。
        on_epoch: 每轮结束的回调，签名 `(info: dict) -> None`，用于 Web 实时进度。
        logger: 日志记录器。
    """

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 cfg: TrainConfig,
                 checkpoint_path: str | Path | None = None,
                 on_epoch: Callable[[Dict[str, Any]], None] | None = None,
                 log=None) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.on_epoch = on_epoch
        self.log = log or logger

        self.device = resolve_device(cfg.device)
        self.model.to(self.device)

        self.loss_fn = nn.CrossEntropyLoss()
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("模型中没有可训练参数，请检查 freeze 设置")
        self.optimizer = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # ---------------- 单轮训练 ----------------
    def _train_one_epoch(self) -> float:
        self.model.train()
        running_loss, total = 0.0, 0
        for X, Y in self.train_loader:
            X, Y = X.to(self.device), Y.to(self.device)
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.model(X), Y)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * X.size(0)
            total += X.size(0)
        return running_loss / max(total, 1)

    # ---------------- 主循环 ----------------
    def fit(self) -> TrainResult:
        seed_everything(self.cfg.seed)
        started = time.perf_counter()

        result = TrainResult(
            arch=self.cfg.arch,
            num_params_total=count_all_parameters(self.model),
            num_params_trainable=count_parameters(self.model),
        )
        best_acc = -1.0

        self.log.info("开始训练：%s", self.cfg.summary())
        self.log.info("设备 %s | 总参数 %s | 可训练参数 %s",
                      self.device,
                      f"{result.num_params_total:,}",
                      f"{result.num_params_trainable:,}")

        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.perf_counter()
            train_loss = self._train_one_epoch()
            ev = evaluate(self.model, self.val_loader, self.device, self.loss_fn)
            metrics = ev.to_metrics()
            epoch_seconds = time.perf_counter() - t0

            info = {
                "epoch": epoch,
                "total_epochs": self.cfg.epochs,
                "train_loss": round(train_loss, 5),
                "val_loss": round(ev.loss, 5),
                "accuracy": round(metrics["accuracy"], 5),
                "f1_macro": round(metrics["f1_macro"], 5),
                "seconds": round(epoch_seconds, 2),
            }
            result.history.append(info)
            self.log.info("Epoch [%d/%d] loss %.4f | val_loss %.4f | %s | %.2fs",
                          epoch, self.cfg.epochs, train_loss, ev.loss,
                          format_metrics(metrics), epoch_seconds)

            # 早停思想：只保留验证集表现最好的那一轮权重
            if metrics["accuracy"] > best_acc:
                best_acc = metrics["accuracy"]
                result.best_epoch = epoch
                result.best_metrics = metrics
                if self.checkpoint_path:
                    save_checkpoint(self.model, self.checkpoint_path,
                                    arch=self.cfg.arch,
                                    config=self.cfg.to_dict(),
                                    metrics=metrics)
                    result.checkpoint = str(self.checkpoint_path)

            if self.on_epoch:
                try:
                    self.on_epoch(info)
                except Exception as exc:  # 回调失败不应中断训练
                    self.log.warning("进度回调异常: %s", exc)

        # 训练结束：用最后一轮再评估一次（不覆盖 best 权重）
        final_ev = evaluate(self.model, self.val_loader, self.device, self.loss_fn)
        result.final_metrics = final_ev.to_metrics()
        result.best_accuracy = max(best_acc, 0.0)
        result.train_seconds = time.perf_counter() - started

        self.log.info("训练完成 | 最佳准确率 %.4f (第 %d 轮) | 总耗时 %.2fs",
                      result.best_accuracy, result.best_epoch, result.train_seconds)
        return result


@torch.no_grad()
def benchmark_latency(model: nn.Module,
                      image_size: int = IMAGE_SIZE,
                      device: torch.device | None = None,
                      n: int = 30,
                      warmup: int = 3) -> float:
    """测量单张图片的平均推理延迟（毫秒）。

    用于第四步"模型选型"：小数据集上准确率拉不开差距时，
    推理速度和模型体积才是真正的选择依据。
    """
    device = device or resolve_device("auto")
    model.eval().to(device)
    x = torch.randn(1, 3, image_size, image_size, device=device)

    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / n * 1000


# --------------------------------------------------------------------------- #
# 权重保存 / 加载
# --------------------------------------------------------------------------- #
def save_checkpoint(model: nn.Module,
                    path: str | Path,
                    arch: str,
                    config: Dict[str, Any] | None = None,
                    metrics: Dict[str, Any] | None = None) -> Path:
    """保存权重，并附带结构元信息（arch / classes / 指标）。

    为什么不只存 state_dict：推理端若不知道 arch，就无法重建网络结构。
    带上元信息后，"一个 .pth 文件 = 一个可直接部署的模型"。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "arch": arch,
        "num_classes": len(CLASSES),
        "classes": list(CLASSES),
        "state_dict": model.state_dict(),
        "config": config or {},
        "metrics": metrics or {},
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "format": "catdog-ckpt-v1",
    }
    torch.save(payload, path)
    # 额外写一份轻量元信息（不含张量），供服务端快速列出模型清单，
    # 避免为了看 arch/指标而把上百 MB 的权重读进内存。
    meta_path = path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({
        "arch": payload["arch"],
        "classes": payload["classes"],
        "config": payload["config"],
        "metrics": payload["metrics"],
        "created_at": payload["created_at"],
        "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        "file": path.name,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def is_plain_state_dict(obj: Any) -> bool:
    """判断 .pth 里是不是"裸的 state_dict"（兼容指导书里那种保存方式）。"""
    return isinstance(obj, dict) and obj and all(
        isinstance(v, torch.Tensor) for v in obj.values())


# --------------------------------------------------------------------------- #
# 实验记录落盘
# --------------------------------------------------------------------------- #
def save_experiment_record(record: ExperimentRecord,
                           directory: str | Path = METRIC_DIR) -> Path:
    """把一次实验写进 outputs/metrics/<name>.json，供前端"实验对比"页读取。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{record.name}.json"
    path.write_text(json.dumps(record.to_dict(), ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path


def load_experiment_records(directory: str | Path = METRIC_DIR) -> List[Dict[str, Any]]:
    """读取所有实验记录，按修改时间升序（即实验进行顺序）。"""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    files = sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime)
    records = []
    for f in files:
        try:
            records.append(json.loads(f.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            logger.warning("实验记录解析失败，已跳过: %s", f)
    return records


def default_checkpoint_path(arch: str) -> Path:
    return CKPT_DIR / f"{arch}.pth"
