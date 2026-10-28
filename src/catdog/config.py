"""全局配置模块。

设计原则：所有"魔法数字"（路径、超参、类别名、归一化系数）集中在本文件，
其它模块一律从此处引用，避免同一常量在多处硬编码导致实验不可复现。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict

# --------------------------------------------------------------------------- #
# 路径
# --------------------------------------------------------------------------- #
# src/catdog/config.py -> 上溯两级得到项目根目录
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "datasets"
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
CKPT_DIR: Path = OUTPUT_DIR / "checkpoints"
METRIC_DIR: Path = OUTPUT_DIR / "metrics"
LOG_DIR: Path = OUTPUT_DIR / "logs"
WEB_DIR: Path = PROJECT_ROOT / "web"

for _d in (CKPT_DIR, METRIC_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# 任务定义
# --------------------------------------------------------------------------- #
CLASSES: tuple[str, ...] = ("cat", "dog")          # 索引 0/1，与文件名前缀一致
CLASS_LABEL_CN: Dict[str, str] = {"cat": "猫", "dog": "狗"}
CLASS_EMOJI: Dict[str, str] = {"cat": "🐱", "dog": "🐶"}
NUM_CLASSES: int = len(CLASSES)

# --------------------------------------------------------------------------- #
# 图像预处理常量（沿用 ImageNet 统计值，与预训练模型保持一致）
# --------------------------------------------------------------------------- #
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

IMAGE_SIZE: int = 224          # 模型输入尺寸
RESIZE_SIZE: int = 256         # 增强前先放大到的尺寸

# --------------------------------------------------------------------------- #
# 训练默认超参
# --------------------------------------------------------------------------- #
DEFAULT_SEED: int = 42         # 固定随机种子：保证"每次只改一个变量"的公平对比
VAL_RATIO: float = 0.2         # 80% 训练 / 20% 验证
DEFAULT_BATCH_SIZE: int = 16
DEFAULT_EPOCHS: int = 5
DEFAULT_LR: float = 1e-3

# 支持的图片后缀
IMAGE_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class TrainConfig:
    """一次训练实验的完整配置。

    所有实验脚本与 Web 后端都通过构造本对象来启动训练，
    保证 CLI 与 HTTP 两条入口走的是同一套代码路径。
    """

    arch: str = "resnet18"
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    lr: float = DEFAULT_LR
    weight_decay: float = 1e-4

    pretrained: bool = True       # 是否加载 ImageNet 预训练权重
    freeze_backbone: bool = True  # 是否冻结特征提取层（只训练分类头）
    augment: bool = False         # 训练集是否做数据增强

    data_dir: Path = DATA_DIR
    image_size: int = IMAGE_SIZE
    val_ratio: float = VAL_RATIO
    num_workers: int = 0          # Windows 下 0 最稳；Linux/Mac 可设 2~4
    seed: int = DEFAULT_SEED
    device: str = "auto"          # auto / cpu / cuda
    pin_memory: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """序列化为 JSON 友好的字典（Path 会转成字符串）。"""
        data = asdict(self)
        data["data_dir"] = str(self.data_dir)
        return data

    def summary(self) -> str:
        """一行人类可读的实验描述，用于日志与前端展示。"""
        mode = "从零训练" if not self.pretrained else (
            "迁移学习(冻结卷积)" if self.freeze_backbone else "迁移学习(全量微调)")
        aug = "有增强" if self.augment else "无增强"
        return (f"{self.arch} | {mode} | {aug} | "
                f"epochs={self.epochs} bs={self.batch_size} lr={self.lr:g}")


@dataclass
class ExperimentRecord:
    """一次实验的结果记录，会被写入 outputs/metrics/*.json 供前端读取。"""

    name: str                       # 实验名，如 step2_pretrained
    arch: str
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    history: list = field(default_factory=list)
    checkpoint: str | None = None
    train_seconds: float = 0.0
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "arch": self.arch,
            "config": self.config,
            "metrics": self.metrics,
            "history": self.history,
            "checkpoint": self.checkpoint,
            "train_seconds": self.train_seconds,
            "note": self.note,
        }
