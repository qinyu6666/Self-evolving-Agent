import json
from typing import List, Dict
import numpy as np


def compute_map(pred_boxes: List[List[float]],
                true_boxes: List[List[float]],
                iou_thresh: float = 0.5) -> float:
    """
    极简 mAP@0.5（单类）
    pred_boxes / true_boxes: [[x1,y1,x2,y2], ...]
    """
    if not true_boxes:
        return 1.0 if not pred_boxes else 0.0
    if not pred_boxes:
        return 0.0

    pred = np.array(pred_boxes)
    true = np.array(true_boxes)
    tp = np.zeros(len(pred))
    used = np.zeros(len(true), dtype=bool)

    for i, p in enumerate(pred):
        ious = _iou(p, true)
        best = np.argmax(ious)
        if ious[best] >= iou_thresh and not used[best]:
            tp[i] = 1
            used[best] = True
    return float(np.sum(tp)) / max(len(pred), 1)


def topk_recall(pred_labels: List[str], gt_labels: List[str], k: int = 5) -> float:
    """Top-k 召回，简单字符串匹配"""
    if not gt_labels:
        return 1.0
    return len(set(pred_labels[:k]) & set(gt_labels)) / len(gt_labels)


def _iou(boxA: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """向量版 IoU"""
    xA = np.maximum(boxA[0], boxes[:, 0])
    yA = np.maximum(boxA[1], boxes[:, 1])
    xB = np.minimum(boxA[2], boxes[:, 2])
    yB = np.minimum(boxA[3], boxes[:, 3])
    inter = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxes[:, 2] - boxes[:, 1]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (areaA + areaB - inter + 1e-6)
