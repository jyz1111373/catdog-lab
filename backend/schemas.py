"""后端接口的请求 / 响应数据模型（Pydantic）。

把"接口长什么样"集中定义，FastAPI 会自动完成校验并生成 /api/docs 文档。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    """发起一次训练的请求体。字段与 TrainConfig 一一对应。"""

    arch: str = Field(default="resnet18", description="模型架构名")
    epochs: int = Field(default=5, ge=1, le=200)
    batch_size: int = Field(default=16, ge=1, le=128)
    lr: float = Field(default=1e-3, gt=0, le=1)
    weight_decay: float = Field(default=1e-4, ge=0)
    pretrained: bool = Field(default=True, description="是否加载 ImageNet 预训练权重")
    freeze: bool = Field(default=True, description="预训练时是否冻结骨干网络")
    augment: bool = Field(default=False, description="训练集是否做数据增强")
    image_size: int = Field(default=224, ge=32, le=512)
    val_ratio: float = Field(default=0.2, gt=0, lt=0.9)
    seed: int = Field(default=42)
    name: Optional[str] = Field(default=None, description="实验名，默认用 arch")
    note: str = Field(default="", description="实验备注")

    model_config = {
        "json_schema_extra": {
            "example": {
                "arch": "resnet18", "epochs": 5, "batch_size": 16,
                "lr": 0.001, "pretrained": True, "freeze": True, "augment": True,
            }
        }
    }


class TrainStatus(BaseModel):
    """训练任务状态。前端据此渲染进度条与曲线。"""

    running: bool = False
    status: str = "idle"            # idle / running / done / error / cancelled
    name: Optional[str] = None
    config: Dict[str, Any] = {}
    progress: Dict[str, Any] = {}   # 当前 epoch 的信息
    history: List[Dict[str, Any]] = []
    message: str = ""
    result: Optional[Dict[str, Any]] = None
    elapsed: float = 0.0


class PredictResponse(BaseModel):
    """预测结果。字段由 Predictor.predict 产出，此处只做文档说明。"""

    label: str
    label_cn: str
    emoji: str
    confidence: float
    probabilities: Dict[str, float]
    all: List[Dict[str, Any]]
    inference_ms: float
    arch: str
    checkpoint: str
    truth: Optional[str] = None     # 若前端告知了真实标签（样本库点击时会带）

    model_config = {"from_attributes": True}


class ModelItem(BaseModel):
    """模型清单里的一项。"""

    file: str
    arch: str
    size_mb: float = 0.0
    created_at: Optional[str] = None
    accuracy: float = 0.0
    metrics: Dict[str, Any] = {}
    config: Dict[str, Any] = {}
    active: bool = False
