"""猫狗分类 Web 服务（FastAPI）。

接口一览：
    GET  /api/health              服务与模型状态
    GET  /api/models              权重清单（含准确率/体积）
    POST /api/models/active       切换当前使用的模型
    POST /api/predict             上传图片识别（multipart）
    POST /api/predict/base64      base64 图片识别（摄像头拍照用）
    GET  /api/experiments         全部实验记录（对比表 / 曲线）
    GET  /api/dataset/stats       数据集统计与样本列表
    GET  /api/dataset/image/{n}   读取数据集图片
    POST /api/train               启动训练任务
    GET  /api/train/status        查询训练进度
    GET  /api/archs               可用模型架构清单

启动：
    python backend/server.py
    # 或 uvicorn backend.server:app --reload --port 8000
"""

from __future__ import annotations

import base64
import io
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT), str(ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import FastAPI, File, HTTPException, UploadFile            # noqa: E402
from fastapi.middleware.cors import CORSMiddleware                      # noqa: E402
from fastapi.responses import FileResponse, JSONResponse                # noqa: E402
from fastapi.staticfiles import StaticFiles                             # noqa: E402
from pydantic import BaseModel                                          # noqa: E402
from PIL import Image                                                   # noqa: E402

from catdog.config import DATA_DIR, IMAGE_SUFFIXES, WEB_DIR             # noqa: E402
from catdog.data import dataset_stats                                   # noqa: E402
from catdog.models import arch_info, available_archs                    # noqa: E402
from catdog.utils import get_logger, resolve_device                     # noqa: E402

import service                                                          # noqa: E402
from schemas import ModelItem, TrainRequest, TrainStatus                # noqa: E402

logger = get_logger("catdog.server")

MAX_UPLOAD_MB = 12


# --------------------------------------------------------------------------- #
# 生命周期
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载模型，避免第一个请求才去读权重导致超时。"""
    logger.info("服务启动，正在加载模型…")
    info = service.bootstrap()
    logger.info("设备: %s | 模型: %s | 可用架构: %d 个",
                info["device"], info.get("active_model"), len(info["archs"]))
    app.state.info = info
    yield
    logger.info("服务关闭")


app = FastAPI(
    title="猫狗分类器 API",
    description="从模型训练到部署上线的完整示例：PyTorch + FastAPI + 原生前端",
    version="1.0.0",
    lifespan=lifespan,
)

# 开发阶段放开跨域，方便用其它端口的调试页面或小程序访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


def _predictor():
    p = service.registry.predictor
    if p is None:
        raise HTTPException(
            status_code=503,
            detail="尚未加载模型。请先运行 python scripts/train.py --arch resnet18 --epochs 5")
    return p


def _read_image(data: bytes) -> Image.Image:
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"图片过大，请控制在 {MAX_UPLOAD_MB}MB 以内")
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"无法解析图片: {exc}")


# --------------------------------------------------------------------------- #
# 健康检查
# --------------------------------------------------------------------------- #
@app.get("/api/health", summary="服务与模型状态")
async def health() -> Dict[str, Any]:
    info = dict(getattr(app.state, "info", {}))
    info.update({
        "status": "ok" if service.registry.predictor else "no_model",
        "active_model": service.registry.active_file,
        "device": str(resolve_device("auto")),
    })
    return info


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
@app.get("/api/models", response_model=list[ModelItem], summary="权重清单")
async def list_models():
    return service.registry.list_models()


class ActiveRequest(BaseModel):
    file: str


@app.post("/api/models/active", summary="切换当前模型")
async def set_active(req: ActiveRequest):
    try:
        predictor = service.registry.load(req.file)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"加载失败: {exc}")
    return {"ok": True, "active": req.file, "describe": predictor.describe()}


@app.get("/api/archs", summary="可用模型架构")
async def archs():
    return {
        "archs": available_archs(),
        "arch_info": arch_info(),
    }


# --------------------------------------------------------------------------- #
# 推理
# --------------------------------------------------------------------------- #
@app.post("/api/predict", summary="上传图片识别")
async def predict(file: UploadFile = File(...),
                  truth: Optional[str] = None):
    if not file.content_type or not file.content_type.startswith("image/"):
        if not Path(file.filename or "").suffix.lower().endswith(IMAGE_SUFFIXES):
            raise HTTPException(status_code=400, detail="请上传图片文件")
    data = await file.read()
    image = _read_image(data)
    result = _predictor().predict(image)
    result["truth"] = truth
    result["filename"] = file.filename
    return result


class Base64Request(BaseModel):
    image: str          # 允许带 data:image/...;base64, 前缀
    truth: Optional[str] = None


@app.post("/api/predict/base64", summary="base64 图片识别（摄像头）")
async def predict_base64(req: Base64Request):
    raw = req.image
    if "," in raw[:64]:                      # 去掉 data URL 前缀
        raw = raw.split(",", 1)[1]
    try:
        data = base64.b64decode(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"base64 解码失败: {exc}")
    image = _read_image(data)
    result = _predictor().predict(image)
    result["truth"] = req.truth
    return result


# --------------------------------------------------------------------------- #
# 数据集
# --------------------------------------------------------------------------- #
@app.get("/api/dataset/stats", summary="数据集统计")
async def stats():
    return dataset_stats(DATA_DIR)


@app.get("/api/dataset/image/{name}", summary="读取数据集图片")
async def dataset_image(name: str):
    # 防目录穿越：只取文件名，且必须在数据集目录内
    safe = Path(name).name
    path = (DATA_DIR / safe).resolve()
    if DATA_DIR.resolve() not in path.parents:
        raise HTTPException(status_code=403, detail="非法路径")
    if not path.exists():
        raise HTTPException(status_code=404, detail="图片不存在")
    return FileResponse(path)


# --------------------------------------------------------------------------- #
# 实验记录
# --------------------------------------------------------------------------- #
@app.get("/api/experiments", summary="全部实验记录")
async def experiments():
    return {"records": service.experiment_summary()}


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #
@app.post("/api/train", response_model=TrainStatus, summary="启动训练任务")
async def train(req: TrainRequest):
    if req.arch not in available_archs():
        raise HTTPException(status_code=400, detail=f"未知架构: {req.arch}")
    try:
        return service.manager.start(req.model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.get("/api/train/status", response_model=TrainStatus, summary="训练进度")
async def train_status():
    return service.manager.status()


# --------------------------------------------------------------------------- #
# 静态前端（必须放在最后，避免吞掉 /api 路由）
# --------------------------------------------------------------------------- #
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    path = WEB_DIR / "assets" / "favicon.svg"
    return FileResponse(path) if path.exists() else JSONResponse({}, status_code=404)


if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="启动猫狗分类 Web 服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--lan", action="store_true",
                        help="监听 0.0.0.0，便于手机在同一局域网访问（拍照测试）")
    args = parser.parse_args()

    host = "0.0.0.0" if args.lan else args.host
    print(f"\n  猫狗分类器已启动： http://127.0.0.1:{args.port}")
    if args.lan:
        print(f"  手机同局域网访问： http://<本机IP>:{args.port}")
    print("  接口文档： http://127.0.0.1:%d/api/docs\n" % args.port)
    uvicorn.run(app, host=host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
