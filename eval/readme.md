## Evaluation Script Usage Guide

## 1. Prepare Ground Truth

Write the ground truth file as data/ground_truth.json, with one record per line, for example:
```
{"image": "path/to/img.jpg", "labels": ["cat", "remote"]}
```

## 2. Run Evaluation

```
cd eval

python eval.py \
  --weights ../yolo_concept_sdk/yolo_concept/data/weights/yolov8n.pt \
  --gt data/ground_truth.json \
  --topk 5
```

## 3. Output

The console will display:
```
mAP@0.5: 0.823
Top-5 Recall: 0.912
```

and will generate a detailed report at:
```
results/eval_report.json
```

The console prints average performance metrics, and a full evaluation report is saved under eval/results/.
Example command:
```
cd eval
python eval.py \
  --weights ../yolo_concept_sdk/yolo_concept/data/weights/yolov8n.pt \
  --gt data/ground_truth.json \
  --topk 5
```


