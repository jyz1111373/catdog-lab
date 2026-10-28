"""推理层：把"加载权重 + 预处理 + 前向 + 后处理"封装成一个 Predictor。

训练脚本与 Web 后端共用同一个 Predictor，因此页面上的结果
与命令行 `python scripts/step5_predict.py` 的结果必然一致。
"""

from __future__ import annotations

import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
from PIL import Image
from torchvision import models  # noqa: F401  (确保 torchvision 已导入)

from .config import CLASS_EMOJI, CLASS_LABEL_CN, CLASSES, IMAGE_SIZE
from .data import build_val_transform
from .models import build_model
from .utils import resolve_device

ImageSource = Union[str, Path, bytes, Image.Image]


class Predictor:
    """猫狗分类器。

    Args:
        checkpoint: .pth 权重路径（由 engine.save_checkpoint 产出）。
        device: 'auto' / 'cpu' / 'cuda'。
    """

    def __init__(self, checkpoint: str | Path, device: str = "auto") -> None:
        self.checkpoint_path = Path(checkpoint)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"权重文件不存在: {self.checkpoint_path}")

        payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("format") != "catdog-ckpt-v1":
            raise ValueError(
                f"{self.checkpoint_path} 不是本项目产出的权重文件（缺少元信息）。"
                "请用 scripts/train.py 或各 step 脚本重新训练保存。")

        self.meta: Dict[str, Any] = payload
        self.arch: str = payload["arch"]
        self.classes: List[str] = payload.get("classes", list(CLASSES))
        self.image_size: int = int(payload.get("config", {}).get("image_size", IMAGE_SIZE))

        self.device = resolve_device(device)
        self.model = build_model(self.arch, num_classes=len(self.classes),
                                 pretrained=False, freeze_backbone=False)
        self.model.load_state_dict(payload["state_dict"])
        self.model.to(self.device).eval()

        self.transform = build_val_transform(self.image_size)
        # 预热一次：避免首次请求把编译/初始化耗时算进"推理延迟"
        self.predict(Image.new("RGB", (self.image_size, self.image_size)))

    # ------------------------------------------------------------------ #
    def _to_pil(self, source: ImageSource) -> Image.Image:
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        if isinstance(source, (bytes, bytearray)):
            return Image.open(BytesIO(bytes(source))).convert("RGB")
        return Image.open(str(source)).convert("RGB")

    @torch.no_grad()
    def predict_probs(self, source: ImageSource) -> List[float]:
        """返回各类别概率列表，如 [0.03, 0.97]。"""
        x = self.transform(self._to_pil(source)).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(x), dim=1)[0]
        return [float(p) for p in probs]

    def predict(self, source: ImageSource) -> Dict[str, Any]:
        """完整预测结果，字段设计成前端可直接渲染。

        Returns:
            {
              label, label_cn, emoji, confidence,
              probabilities: {cat: .., dog: ..},
              all: [{name,label_cn,prob}...],   # 按概率降序
              inference_ms, arch, checkpoint
            }
        """
        started = time.perf_counter()
        probs = self.predict_probs(source)
        elapsed_ms = (time.perf_counter() - started) * 1000

        best = max(range(len(probs)), key=lambda i: probs[i])
        label = self.classes[best]

        all_items = [
            {
                "name": self.classes[i],
                "label_cn": CLASS_LABEL_CN.get(self.classes[i], self.classes[i]),
                "emoji": CLASS_EMOJI.get(self.classes[i], ""),
                "prob": round(probs[i], 6),
            }
            for i in range(len(probs))
        ]
        all_items.sort(key=lambda d: -d["prob"])

        return {
            "label": label,
            "label_cn": CLASS_LABEL_CN.get(label, label),
            "emoji": CLASS_EMOJI.get(label, ""),
            "confidence": round(probs[best], 6),
            "probabilities": {self.classes[i]: round(probs[i], 6) for i in range(len(probs))},
            "all": all_items,
            "inference_ms": round(elapsed_ms, 2),
            "arch": self.arch,
            "checkpoint": self.checkpoint_path.name,
        }

    def describe(self) -> Dict[str, Any]:
        """模型元信息，供 /api/models 与页面顶部展示。"""
        return {
            "arch": self.arch,
            "checkpoint": self.checkpoint_path.name,
            "path": str(self.checkpoint_path),
            "classes": self.classes,
            "image_size": self.image_size,
            "metrics": self.meta.get("metrics", {}),
            "config": self.meta.get("config", {}),
            "created_at": self.meta.get("created_at"),
        }


def load_predictor(checkpoint: str | Path, device: str = "auto") -> Predictor:
    return Predictor(checkpoint, device=device)
