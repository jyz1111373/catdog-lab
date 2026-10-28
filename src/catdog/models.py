"""模型层：手写网络 + torchvision 预训练模型工厂。

统一入口是 `build_model(arch, ...)`，无论手写还是官方模型，
都返回「同样接口」的 nn.Module，因此训练引擎完全不需要关心模型种类。
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn
from torchvision import models

from .config import NUM_CLASSES


# =========================================================================== #
# 第一步：手写 AlexNet（从零训练）
# =========================================================================== #
class AlexNetScratch(nn.Module):
    """精简版 AlexNet，输入 224×224，输出 num_classes 类。

    尺寸推导（输入 224）：
        conv11/s4/p1 -> 54 ; pool3/s2 -> 26
        conv5/p2     -> 26 ; pool3/s2 -> 12
        conv3/p1 ×3  -> 12 ; pool3/s2 -> 5
        flatten      -> 256*5*5 = 6400
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6400, 4096), nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier 初始化，避免从零训练时梯度爆炸/消失。"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# =========================================================================== #
# 第二步：与 torchvision 官方 AlexNet 结构对齐的手写版（用于装载预训练权重）
# =========================================================================== #
class AlexNetOfficialLayout(nn.Module):
    """结构、子模块命名、层索引都与 `torchvision.models.alexnet` 完全一致。

    只有完全一致才能用 `strict=True` 加载 ImageNet 权重；
    装载完以后再把 classifier[6] 换成 2 分类的输出头。
    """

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),   # 索引 6 <- 要替换的分类头
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_alexnet_pretrained(num_classes: int = NUM_CLASSES,
                             freeze_backbone: bool = True) -> nn.Module:
    """加载 ImageNet 预训练 AlexNet，并把 1000 类分类头换成 2 类。

    步骤：
        1. 先建 1000 类的完整结构；
        2. strict=True 装载官方权重（此时形状完全对齐）；
        3. 把 classifier[6] 换成 nn.Linear(4096, 2)（新头自动随机初始化）。
    """
    official = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    net = AlexNetOfficialLayout(num_classes=1000)
    net.load_state_dict(official.state_dict())
    net.classifier[6] = nn.Linear(4096, num_classes)
    if freeze_backbone:
        set_requires_grad(net, trainable_names=("classifier",))
    return net


# =========================================================================== #
# 第四步：torchvision 模型工厂
# =========================================================================== #
# 每个模型：构造官方预训练模型 + 给出分类头替换方式与"骨干参数名前缀"
_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "resnet18": {
        "weights": models.ResNet18_Weights.DEFAULT,
        "ctor": models.resnet18,
        "head": "fc",
        "params": "11.7M",
        "desc": "残差网络经典款，结构规整，入门首选",
    },
    "mobilenet_v3_small": {
        "weights": models.MobileNet_V3_Small_Weights.DEFAULT,
        "ctor": models.mobilenet_v3_small,
        "head": "classifier.3",
        "params": "2.5M",
        "desc": "深度可分离卷积，为移动端设计，体积最小",
    },
    "shufflenet_v2_x1_0": {
        "weights": models.ShuffleNet_V2_X1_0_Weights.DEFAULT,
        "ctor": models.shufflenet_v2_x1_0,
        "head": "fc",
        "params": "2.3M",
        "desc": "通道混洗，极致轻量，CPU 推理快",
    },
    "efficientnet_b0": {
        "weights": models.EfficientNet_B0_Weights.DEFAULT,
        "ctor": models.efficientnet_b0,
        "head": "classifier.1",
        "params": "5.3M",
        "desc": "复合缩放，精度/开销比优秀（CPU 上偏慢）",
    },
    "alexnet": {
        "weights": models.AlexNet_Weights.DEFAULT,
        "ctor": models.alexnet,
        "head": "classifier.6",
        "params": "61.1M",
        "desc": "经典网络，参数量大但结构直观，便于教学",
    },
}


def available_archs() -> List[str]:
    """返回所有可用模型名（含两个手写入口）。"""
    return ["alexnet_scratch", "alexnet_manual", *_MODEL_REGISTRY.keys()]


def arch_info() -> List[Dict[str, Any]]:
    """模型清单 + 说明，供前端下拉框展示。"""
    items = [
        {"name": "alexnet_scratch", "params": "58.3M", "desc": "手写 AlexNet，随机初始化，从零训练（第一步）"},
        {"name": "alexnet_manual", "params": "61.1M", "desc": "手写 AlexNet + ImageNet 预训练权重（第二步）"},
    ]
    for name, meta in _MODEL_REGISTRY.items():
        items.append({"name": name, "params": meta["params"], "desc": meta["desc"]})
    return items


def _set_module_by_path(net: nn.Module, path: str, module: nn.Module) -> None:
    """按 'classifier.3' 这样的点分路径替换子模块。"""
    parts = path.split(".")
    obj = net
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    last = parts[-1]
    if last.isdigit():
        obj[int(last)] = module
    else:
        setattr(obj, last, module)


def _get_module_by_path(net: nn.Module, path: str) -> nn.Module:
    obj = net
    for p in path.split("."):
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    return obj


def set_requires_grad(net: nn.Module, trainable_names: tuple[str, ...]) -> None:
    """冻结网络，只放开名字以 trainable_names 中任一前缀开头的参数。

    例如 trainable_names=("classifier",) 表示"只训练分类头"。
    """
    for name, param in net.named_parameters():
        param.requires_grad = name.startswith(trainable_names)


def build_model(arch: str,
                num_classes: int = NUM_CLASSES,
                pretrained: bool = True,
                freeze_backbone: bool = True) -> nn.Module:
    """统一的模型构建入口。

    Args:
        arch: 模型名，见 `available_archs()`。
        num_classes: 输出类别数，猫狗二分类为 2。
        pretrained: 是否加载 ImageNet 预训练权重（`alexnet_scratch` 恒为 False）。
        freeze_backbone: 预训练时是否冻结骨干，只训练分类头。

    Returns:
        配置好 requires_grad 的 nn.Module。
    """
    arch = arch.lower()

    # ---- 手写入口 ----
    if arch == "alexnet_scratch":
        return AlexNetScratch(num_classes=num_classes)

    if arch == "alexnet_manual":
        if not pretrained:
            net = AlexNetOfficialLayout(num_classes=num_classes)
            return net
        return build_alexnet_pretrained(num_classes, freeze_backbone)

    # ---- torchvision 官方模型 ----
    if arch not in _MODEL_REGISTRY:
        raise ValueError(f"未知模型 '{arch}'，可选：{available_archs()}")

    meta = _MODEL_REGISTRY[arch]
    weights = meta["weights"] if pretrained else None
    net = meta["ctor"](weights=weights)

    head_path = meta["head"]
    old_head = _get_module_by_path(net, head_path)
    in_features = getattr(old_head, "in_features", None)
    if in_features is None:
        raise RuntimeError(f"模型 {arch} 的分类头 {head_path} 不是 Linear，无法自动替换")
    _set_module_by_path(net, head_path, nn.Linear(in_features, num_classes))

    if pretrained and freeze_backbone:
        # 先全冻，再放开新分类头
        for p in net.parameters():
            p.requires_grad = False
        for p in _get_module_by_path(net, head_path).parameters():
            p.requires_grad = True

    return net
