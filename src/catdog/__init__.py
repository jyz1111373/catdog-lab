"""catdog —— 猫狗二分类实验工程包。

分层结构：
    config     全局配置与超参（单一事实来源）
    data       数据集、划分、变换、DataLoader
    models     手写网络与 torchvision 模型工厂
    metrics    评价指标计算
    engine     训练 / 评估循环、权重保存、实验记录
    inference  面向应用的 Predictor（单图推理）
    utils      种子、设备、计时、日志等通用工具
"""

from .config import (
    CLASSES,
    DATA_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMAGE_SIZE,
    NUM_CLASSES,
    PROJECT_ROOT,
    TrainConfig,
)
from .data import CatDogDataset, build_loaders, build_val_transform, dataset_stats
from .engine import Trainer, evaluate, save_checkpoint, save_experiment_record
from .inference import Predictor
from .metrics import compute_metrics, format_metrics
from .models import available_archs, build_model

__version__ = "1.0.0"

__all__ = [
    "CLASSES", "DATA_DIR", "IMAGENET_MEAN", "IMAGENET_STD", "IMAGE_SIZE",
    "NUM_CLASSES", "PROJECT_ROOT", "TrainConfig",
    "CatDogDataset", "build_loaders", "build_val_transform", "dataset_stats",
    "Trainer", "evaluate", "save_checkpoint", "save_experiment_record",
    "Predictor", "compute_metrics", "format_metrics",
    "available_archs", "build_model", "__version__",
]
