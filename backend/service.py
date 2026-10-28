"""后端业务层：模型仓库 + 训练任务管理。

把"状态"从 HTTP 路由里剥离出来，路由只负责收发 JSON，
所有与模型/训练相关的副作用都发生在这里，便于单独测试。
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys

# 允许 `python backend/service.py` 直接运行，也允许被 server.py 导入
ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from catdog.config import CKPT_DIR, DATA_DIR, TrainConfig          # noqa: E402
from catdog.engine import load_experiment_records                  # noqa: E402
from catdog.inference import Predictor                             # noqa: E402
from catdog.models import arch_info, available_archs               # noqa: E402
from catdog.pipeline import run_experiment                         # noqa: E402
from catdog.utils import device_info, get_logger, resolve_device   # noqa: E402

logger = get_logger("catdog.service")


# --------------------------------------------------------------------------- #
# 模型仓库
# --------------------------------------------------------------------------- #
class ModelRegistry:
    """管理 outputs/checkpoints 下的所有权重，并维护"当前使用的那一个"。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._predictor: Optional[Predictor] = None
        self._active_file: Optional[str] = None

    # ---------- 清单 ----------
    def list_models(self) -> List[Dict[str, Any]]:
        """扫描权重目录，返回带指标的模型清单（按准确率降序）。"""
        items: List[Dict[str, Any]] = []
        for meta_path in sorted(CKPT_DIR.glob("*.meta.json"),
                                key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            ckpt = CKPT_DIR / meta.get("file", meta_path.name.replace(".meta.json", ""))
            if not ckpt.exists():
                continue
            metrics = meta.get("metrics", {})
            items.append({
                "file": ckpt.name,
                "arch": meta.get("arch", "unknown"),
                "size_mb": meta.get("size_mb", 0.0),
                "created_at": meta.get("created_at"),
                "accuracy": round(float(metrics.get("accuracy", 0.0)), 4),
                "metrics": metrics,
                "config": meta.get("config", {}),
                "active": ckpt.name == self._active_file,
            })
        items.sort(key=lambda d: -d["accuracy"])
        return items

    # ---------- 加载 ----------
    def load(self, file_name: str) -> Predictor:
        """加载指定权重并设为当前模型。"""
        path = CKPT_DIR / Path(file_name).name
        if not path.exists():
            raise FileNotFoundError(f"权重不存在: {path}")
        with self._lock:
            self._predictor = Predictor(path)
            self._active_file = path.name
        logger.info("已加载模型: %s", path.name)
        return self._predictor

    def auto_load_best(self) -> Optional[Predictor]:
        """启动时自动挑选准确率最高的权重；没有则回退到任意一个。"""
        items = self.list_models()
        if not items:
            # 没有元信息时，尝试直接扫描 .pth
            files = sorted(CKPT_DIR.glob("*.pth"))
            if not files:
                logger.warning("未找到任何权重，请先训练或运行 scripts/run_all.py")
                return None
            return self.load(files[0].name)
        return self.load(items[0]["file"])

    # ---------- 访问 ----------
    @property
    def predictor(self) -> Optional[Predictor]:
        return self._predictor

    @property
    def active_file(self) -> Optional[str]:
        return self._active_file

    def describe(self) -> Optional[Dict[str, Any]]:
        if self._predictor is None:
            return None
        info = self._predictor.describe()
        info["file"] = self._active_file
        return info

    def reload_active(self) -> None:
        """训练结束后重新加载当前权重，拿到最新的 state_dict。"""
        if self._active_file:
            self.load(self._active_file)


# --------------------------------------------------------------------------- #
# 训练任务
# --------------------------------------------------------------------------- #
class TrainingManager:
    """后台训练任务：单任务串行，线程安全，支持进度查询。"""

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._state: Dict[str, Any] = {
            "status": "idle", "name": None, "config": {},
            "progress": {}, "history": [], "message": "",
            "result": None, "started_at": 0.0, "elapsed": 0.0,
        }

    # ---------- 查询 ----------
    def status(self) -> Dict[str, Any]:
        with self._lock:
            st = dict(self._state)
        st["running"] = st["status"] == "running"
        st["elapsed"] = round(
            time.perf_counter() - st["started_at"], 2) if st["started_at"] else 0.0
        return st

    # ---------- 启动 ----------
    def start(self, req: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if self._state["status"] == "running":
                raise RuntimeError("已有训练任务在运行，请等待完成")
            name = req.get("name") or req.get("arch") or "resnet18"
            self._state = {
                "status": "running", "name": name, "config": req,
                "progress": {}, "history": [], "message": "正在准备数据…",
                "result": None, "started_at": time.perf_counter(), "elapsed": 0.0,
            }
        self._thread = threading.Thread(target=self._run, args=(req,), daemon=True)
        self._thread.start()
        return self.status()

    # ---------- 执行 ----------
    def _run(self, req: Dict[str, Any]) -> None:
        name = req.get("name") or req.get("arch") or "resnet18"
        try:
            cfg = TrainConfig(
                arch=req.get("arch", "resnet18"),
                epochs=int(req.get("epochs", 5)),
                batch_size=int(req.get("batch_size", 16)),
                lr=float(req.get("lr", 1e-3)),
                weight_decay=float(req.get("weight_decay", 1e-4)),
                pretrained=bool(req.get("pretrained", True)),
                freeze_backbone=bool(req.get("freeze", True)),
                augment=bool(req.get("augment", False)),
                image_size=int(req.get("image_size", 224)),
                val_ratio=float(req.get("val_ratio", 0.2)),
                seed=int(req.get("seed", 42)),
                data_dir=DATA_DIR,
            )

            def on_epoch(info: Dict[str, Any]) -> None:
                with self._lock:
                    self._state["progress"] = info
                    self._state["history"].append(info)
                    self._state["message"] = (
                        f"第 {info['epoch']}/{info['total_epochs']} 轮 · "
                        f"准确率 {info['accuracy']:.2%}")

            record = run_experiment(name, cfg, note=req.get("note", cfg.summary()),
                                    on_epoch=on_epoch)

            # 训练完成：把新模型设为当前模型
            try:
                self.registry.load(f"{name}.pth")
            except Exception as exc:
                logger.warning("自动切换模型失败: %s", exc)

            with self._lock:
                self._state.update({
                    "status": "done",
                    "result": record.to_dict(),
                    "message": f"完成 · 最佳准确率 {record.metrics.get('accuracy', 0):.2%}",
                })
            logger.info("训练任务完成: %s", name)
        except Exception as exc:  # 任何异常都要落到状态里，否则前端会一直转圈
            logger.error("训练任务失败: %s\n%s", exc, traceback.format_exc())
            with self._lock:
                self._state.update({"status": "error", "message": f"训练失败: {exc}"})


# --------------------------------------------------------------------------- #
# 全局单例
# --------------------------------------------------------------------------- #
registry = ModelRegistry()
manager = TrainingManager(registry)


def bootstrap() -> Dict[str, Any]:
    """服务启动时调用：载入最佳模型，返回健康信息。"""
    registry.auto_load_best()
    dev = resolve_device("auto")
    info = device_info(dev)
    info["models"] = len(registry.list_models())
    info["active_model"] = registry.active_file
    info["archs"] = available_archs()
    info["arch_info"] = arch_info()
    return info


def experiment_summary() -> List[Dict[str, Any]]:
    """读取全部实验记录，附带权重体积，供前端对比表使用。"""
    records = load_experiment_records()
    for rec in records:
        ckpt = rec.get("checkpoint")
        if ckpt:
            meta = Path(ckpt).with_suffix(".meta.json")
            if meta.exists():
                try:
                    rec["size_mb"] = json.loads(
                        meta.read_text(encoding="utf-8")).get("size_mb", 0.0)
                except json.JSONDecodeError:
                    pass
    return records
