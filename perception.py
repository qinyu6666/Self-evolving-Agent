import cv2, torch
from PIL import Image
from ultralytics import YOLO

class Perception:
    def __init__(self, cap_id=0, yolo_size="yolo11x.pt"):
        self.cap = cv2.VideoCapture(cap_id)
        self.yolo = YOLO(yolo_size)

    def capture_keyframe(self):
        ret, frame = self.cap.read()
        if not ret: return None
        return frame

    def detect_and_crop(self, frame, conf=0.35):
        results = self.yolo(frame, verbose=False)
        if len(results[0].boxes) == 0:
            return None, None
        # 取置信度最高框
        best = max(results[0].boxes, key=lambda x: x.conf)
        x1, y1, x2, y2 = map(int, best.xyxy[0])
        crop = frame[y1:y2, x1:x2]
        crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        label = self.yolo.names[int(best.cls)]
        return crop, label