"""数据层：样本发现、标签解析、划分、变换与 DataLoader 构建。

相比指导书里的 `common.py`，这里做了三处工程化改进：
1. 用**分层抽样**代替纯随机划分，保证训练/验证集里猫狗比例一致（小数据集上很重要）；
2. 数据集对象只保存"路径 + 标签"元信息，图片按需读取，避免一次性载入内存；
3. 训练/验证的 transform 通过视图类注入，同一份样本索引不会重复扫描磁盘。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import (
    CLASSES,
    DATA_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMAGE_SUFFIXES,
    IMAGE_SIZE,
    RESIZE_SIZE,
    VAL_RATIO,
)


# --------------------------------------------------------------------------- #
# 数据集
# --------------------------------------------------------------------------- #
class CatDogDataset(Dataset):
    """猫狗二分类数据集。

    标签直接从文件名解析：`cat.1.jpg` -> 0，`dog.1.jpg` -> 1。

    Args:
        root: 图片所在目录（扁平结构，不要求 train/val 子目录）。
        transform: 图像变换，None 则返回原始 RGB 图（供可视化用）。
    """

    def __init__(self, root: str | Path = DATA_DIR, transform=None) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"数据集目录不存在: {self.root}")

        self.samples: List[Tuple[Path, int]] = [
            (p, label_from_name(p.name))
            for p in sorted(self.root.iterdir())
            if p.suffix.lower() in IMAGE_SUFFIXES
        ]
        if not self.samples:
            raise RuntimeError(f"目录 {self.root} 下没有找到图片文件")

        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        # 打开后必须 convert('RGB')：数据集中可能混入灰度图或 RGBA 图
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    # ---------- 便捷属性 ----------
    @property
    def paths(self) -> List[Path]:
        return [p for p, _ in self.samples]

    @property
    def labels(self) -> List[int]:
        return [y for _, y in self.samples]

    def class_counts(self) -> Dict[str, int]:
        counts = {name: 0 for name in CLASSES}
        for _, y in self.samples:
            counts[CLASSES[y]] += 1
        return counts


def label_from_name(filename: str) -> int:
    """从文件名解析标签：含 'cat' -> 0，否则 -> 1。"""
    name = Path(filename).name.lower()
    for idx, cls in enumerate(CLASSES):
        if name.startswith(cls) or f"{cls}." in name:
            return idx
    # 兜底：文件名不规范时按"非猫即狗"处理并交给使用者检查
    return 1


class _TransformView(Dataset):
    """给「样本索引子集」套上指定 transform 的只读视图。"""

    def __init__(self, base: CatDogDataset, indices: Sequence[int], transform=None):
        self.base = base
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        path, label = self.base.samples[self.indices[i]]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    @property
    def paths(self) -> List[Path]:
        return [self.base.samples[i][0] for i in self.indices]


# --------------------------------------------------------------------------- #
# 图像变换
# --------------------------------------------------------------------------- #
def build_val_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """验证/推理用的确定性变换：只缩放 + 归一化，不做任何随机操作。"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_train_transform(image_size: int = IMAGE_SIZE,
                          augment: bool = False) -> transforms.Compose:
    """训练用变换。augment=True 时叠加随机翻转/裁剪/色彩抖动/旋转。

    注意：增强只作用于训练集，验证集必须用干净的 build_val_transform，
    否则评估指标会因为随机性而不可信。
    """
    if not augment:
        return build_val_transform(image_size)

    return transforms.Compose([
        transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),  # 模拟拍摄距离变化
        transforms.RandomHorizontalFlip(p=0.5),                      # 猫狗朝左朝右仍是猫狗
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(degrees=10),                       # 相机轻微倾斜
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# --------------------------------------------------------------------------- #
# 划分
# --------------------------------------------------------------------------- #
def stratified_split(labels: Sequence[int], val_ratio: float = VAL_RATIO,
                     seed: int = 42) -> Tuple[List[int], List[int]]:
    """按类别比例划分训练/验证索引，保证两集合中各类别占比一致。"""
    by_class: Dict[int, List[int]] = {}
    for idx, y in enumerate(labels):
        by_class.setdefault(int(y), []).append(idx)

    rng = random.Random(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for y in sorted(by_class):
        items = by_class[y][:]
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_ratio)))
        n_val = min(n_val, len(items) - 1)  # 至少留一个样本给训练集
        val_idx.extend(items[:n_val])
        train_idx.extend(items[n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


@dataclass
class DataBundle:
    """一次划分的产物，方便把划分信息一并传给上层。"""

    train_loader: DataLoader
    val_loader: DataLoader
    train_indices: List[int]
    val_indices: List[int]
    train_paths: List[Path]
    val_paths: List[Path]

    def describe(self) -> str:
        return f"训练集 {len(self.train_indices)} 张 / 验证集 {len(self.val_indices)} 张"


def build_loaders(data_dir: str | Path = DATA_DIR,
                  batch_size: int = 16,
                  augment: bool = False,
                  image_size: int = IMAGE_SIZE,
                  val_ratio: float = VAL_RATIO,
                  seed: int = 42,
                  num_workers: int = 0,
                  pin_memory: bool = False) -> DataBundle:
    """构建训练/验证 DataLoader 的唯一入口。

    所有实验共用同一 seed，从而共用同一份划分——这是"控制变量"对比的前提。
    """
    base = CatDogDataset(data_dir)
    train_idx, val_idx = stratified_split(base.labels, val_ratio=val_ratio, seed=seed)

    train_view = _TransformView(base, train_idx, build_train_transform(image_size, augment))
    val_view = _TransformView(base, val_idx, build_val_transform(image_size))

    train_loader = DataLoader(train_view, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_view, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        train_indices=train_idx,
        val_indices=val_idx,
        train_paths=train_view.paths,
        val_paths=val_view.paths,
    )


# --------------------------------------------------------------------------- #
# 数据集统计（供前端"数据集"页展示）
# --------------------------------------------------------------------------- #
def dataset_stats(data_dir: str | Path = DATA_DIR, limit: int | None = None) -> dict:
    """统计数据集中各类别数量，并返回样本元信息列表。"""
    ds = CatDogDataset(data_dir)
    counts = ds.class_counts()
    items = []
    for path, label in ds.samples:
        items.append({
            "name": path.name,
            "label": CLASSES[label],
            "url": f"/api/dataset/image/{path.name}",
        })
    if limit:
        items = items[:limit]
    return {
        "total": len(ds),
        "counts": counts,
        "classes": list(CLASSES),
        "items": items,
    }
