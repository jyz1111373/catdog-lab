"""评价指标计算。

把"指标怎么算"从训练循环里剥离出来：engine 只负责产出 y_true / y_pred / y_prob，
具体指标一律由本模块计算，保证训练脚本、单图测试、Web 后端用的是同一套口径。
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .config import CLASSES


def compute_metrics(y_true: Sequence[int],
                    y_pred: Sequence[int],
                    y_prob: Sequence[Sequence[float]] | None = None,
                    classes: Sequence[str] = CLASSES) -> Dict[str, Any]:
    """计算二分类（可推广到多分类）的完整指标。

    Args:
        y_true: 真实标签索引。
        y_pred: 预测标签索引。
        y_prob: 每个样本属于各类别的概率，形如 [[p0, p1], ...]，用于算 AUC。
        classes: 类别名列表。

    Returns:
        包含 accuracy / precision / recall / f1（macro 与逐类）/ 混淆矩阵 / AUC 的字典，
        全部为 Python 原生类型，可直接 JSON 序列化。
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n_classes = len(classes)

    # zero_division=0：某个类完全没被预测到时返回 0 而不是报错
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)),
        average=None, zero_division=0)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    result: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(macro_p),
        "recall_macro": float(macro_r),
        "f1_macro": float(macro_f1),
        "precision": {classes[i]: float(precision[i]) for i in range(n_classes)},
        "recall": {classes[i]: float(recall[i]) for i in range(n_classes)},
        "f1": {classes[i]: float(f1[i]) for i in range(n_classes)},
        "support": {classes[i]: int(support[i]) for i in range(n_classes)},
        "confusion_matrix": cm.tolist(),
        "classes": list(classes),
        "num_samples": int(len(y_true)),
    }

    # AUC 需要概率；二分类用正类概率，多分类用 one-vs-rest
    if y_prob is not None:
        try:
            prob = np.asarray(y_prob, dtype=float)
            if n_classes == 2:
                result["auc"] = float(roc_auc_score(y_true, prob[:, 1]))
            else:
                result["auc"] = float(
                    roc_auc_score(y_true, prob, multi_class="ovr", average="macro"))
        except ValueError:
            # 验证集只包含一个类别时无法计算 AUC，忽略即可
            result["auc"] = None
    else:
        result["auc"] = None

    return result


def format_metrics(metrics: Dict[str, Any]) -> str:
    """把指标字典格式化成一行日志文本。"""
    base = (f"准确率 {metrics['accuracy']:.4f} | "
            f"精确率 {metrics['precision_macro']:.4f} | "
            f"召回率 {metrics['recall_macro']:.4f} | "
            f"F1 {metrics['f1_macro']:.4f}")
    if metrics.get("auc") is not None:
        base += f" | AUC {metrics['auc']:.4f}"
    return base


def metrics_table_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把多份实验记录压成前端表格需要的行（按准确率降序）。"""
    rows = []
    for rec in records:
        m = rec.get("metrics", {})
        rows.append({
            "name": rec.get("name", "-"),
            "arch": rec.get("arch", "-"),
            "note": rec.get("note", ""),
            "accuracy": round(m.get("accuracy", 0.0), 4),
            "precision": round(m.get("precision_macro", 0.0), 4),
            "recall": round(m.get("recall_macro", 0.0), 4),
            "f1": round(m.get("f1_macro", 0.0), 4),
            "train_seconds": round(rec.get("train_seconds", 0.0), 2),
            "checkpoint": rec.get("checkpoint"),
        })
    return sorted(rows, key=lambda r: -r["accuracy"])
