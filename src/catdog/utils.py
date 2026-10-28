"""通用工具：随机种子、设备选择、计时、参数统计、日志记录。"""

from __future__ import annotations

import logging
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from .config import LOG_DIR

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


# --------------------------------------------------------------------------- #
# 可复现性
# --------------------------------------------------------------------------- #
def seed_everything(seed: int = 42) -> None:
    """固定所有随机源，保证划分与训练过程可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 关掉 cuDNN 的非确定性算法（CPU 上无影响，写上便于换 GPU 时仍可复现）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# 设备
# --------------------------------------------------------------------------- #
def resolve_device(preference: str = "auto") -> torch.device:
    """把 'auto' / 'cpu' / 'cuda' / 'cuda:0' 解析成 torch.device。"""
    preference = (preference or "auto").lower()
    if preference.startswith("cuda") and torch.cuda.is_available():
        return torch.device(preference)
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def device_info(device: torch.device) -> dict:
    """返回设备描述信息，供 /api/health 展示。"""
    info = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(device)
    return info


# --------------------------------------------------------------------------- #
# 计时
# --------------------------------------------------------------------------- #
@contextmanager
def timer(name: str = "", logger: logging.Logger | None = None) -> Iterator[list]:
    """用法：with timer("训练") as t: ... ; print(t[0]) 得到耗时秒数。"""
    start = time.perf_counter()
    holder = [0.0]
    try:
        yield holder
    finally:
        holder[0] = time.perf_counter() - start
        if logger:
            logger.info("%s 耗时 %.2f 秒", name, holder[0])


def format_seconds(seconds: float) -> str:
    """把秒数格式化成 '1m23.4s' 这种易读形式。"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    return f"{int(minutes)}m{sec:04.1f}s"


# --------------------------------------------------------------------------- #
# 模型统计
# --------------------------------------------------------------------------- #
def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """统计参数量。默认只算需要求梯度的（即真正参与训练的参数）。"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def count_all_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_file_size_mb(path: str | Path) -> float:
    """权重文件体积（MB）。"""
    p = Path(path)
    if not p.exists():
        return 0.0
    return round(p.stat().st_size / (1024 * 1024), 2)


# --------------------------------------------------------------------------- #
# 日志
# --------------------------------------------------------------------------- #
def get_logger(name: str = "catdog",
               level: int = logging.INFO,
               log_file: str | Path | None = None) -> logging.Logger:
    """创建统一格式的 logger；重复调用不会重复添加 handler。"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def default_log_file(filename: str = "run.log") -> Path:
    return LOG_DIR / filename
