#!/usr/bin/env python3
import argparse
import json
import cv2
from pathlib import Path
from ultralytics import YOLO
from metrics import compute_map, topk_recall


def evaluate(weights: str, gt_path: str, topk: int, img_root: str = ""):
    model = YOLO(weights)
    with open(gt_path) as f:
        gts = [json.loads(l) for l in f if l.strip()]

    all_map, all_recall = [], []
    results = []

    for item in gts:
        img_path = Path(img_root) / item["image"] if img_root else Path(item["image"])
        img = cv2.imread(str(img_path))
        assert img is not None, f"无法读取 {img_path}"

        preds = model(img, verbose=False)  # ultralytics 返回 Results 列表
        boxes = preds[0].boxes.xyxy.cpu().numpy()  # [[x1,y1,x2,y2], ...]
        labels = preds[0].names
        pred_cls = [labels[int(c)] for c in preds[0].boxes.cls.cpu().tolist()]

        # 真值框（这里用整图当框，仅演示）
        h, w = img.shape[:2]
        gt_boxes = [[0, 0, w, h]] * len(item["labels"])
        mAP = compute_map(boxes.tolist(), gt_boxes)
        recall = topk_recall(pred_cls, item["labels"], k=topk)

        all_map.append(mAP)
        all_recall.append(recall)
        results.append({"image": str(img_path), "mAP": mAP, "Top-k Recall": recall})

    report = {
        "mAP@0.5": float(np.mean(all_map)),
        f"Top-{topk} Recall": float(np.mean(all_recall)),
        "details": results,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    Path("results").mkdir(exist_ok=True)
    with open("results/eval_report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="YOLO11.pt")
    parser.add_argument("--gt", required=True, help="ground_truth.jsonl")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--img-root", default="", help="若真值路径是相对路径，可指定根目录")
    args = parser.parse_args()
    import numpy as np
    evaluate(args.weights, args.gt, args.topk, args.img_root)
