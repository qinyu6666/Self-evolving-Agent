\# 评估脚本使用说明



1\. 准备真值  

&nbsp;  把真值写成 `data/ground\_truth.json`，每行一条：  

&nbsp;  {"image": "path/to/img.jpg", "labels": \["cat", "remote"]}



2\. 运行评估  

&nbsp;  cd eval

&nbsp;  python eval.py \\

&nbsp;      --weights ../yolo\_concept\_sdk/yolo\_concept/data/weights/yolov8n.pt \\

&nbsp;      --gt data/ground\_truth.json \\

&nbsp;      --topk 5



3\. 输出  

&nbsp;  控制台打印：  

&nbsp;  mAP@0.5: 0.823  

&nbsp;  Top-5 Recall: 0.912  

&nbsp;  并生成 `results/eval\_report.json`




控制台会打印平均指标，并在 eval/results/ 下生成详细报告
cd eval
python eval.py --weights ../yolo_concept_sdk/yolo_concept/data/weights/yolov8n.pt \
               --gt data/ground_truth.json \
               --topk 5
