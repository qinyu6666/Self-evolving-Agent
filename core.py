import cv2
import threading
import time
from .config import (
    CAMERA_ID,
    YOLO_WEIGHTS,
    FAISS_INDEX_PATH,
    META_JSON_PATH,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    LORA_CKPT,
)
from .perception import Perception
from .memory import Memory
from .kg import KnowledgeGraph
from .learner import Learner


class ConceptLearner:
    """
    统一入口：采集→检测→向量存储→知识图谱→LoRA 微调
    用法：
        bot = ConceptLearner()
        bot.start()          # 非阻塞，后台线程持续采集
        ...
        bot.stop()
    """

    def __init__(
        self,
        camera_id: int = CAMERA_ID,
        yolo_weights: str = YOLO_WEIGHTS,
        faiss_path: str = FAISS_INDEX_PATH,
        meta_path: str = META_JSON_PATH,
        neo4j_uri: str = NEO4J_URI,
        neo4j_user: str = NEO4J_USER,
        neo4j_pwd: str = NEO4J_PASSWORD,
        lora_ckpt: str = LORA_CKPT,
    ):
        # 实例化四个子模块
        self.perception = Perception(weights=yolo_weights)
        self.memory = Memory(faiss_path=faiss_path, meta_path=meta_path)
        self.kg = KnowledgeGraph(uri=neo4j_uri, user=neo4j_user, password=neo4j_pwd)
        self.learner = Learner(lora_ckpt=lora_ckpt)

        # 摄像头
        self.cap = cv2.VideoCapture(camera_id)
        self.running = False
        self.thread = None

    # -------------------- 生命周期 --------------------
    def start(self):
        """后台线程开始采集与学习"""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print("[ConceptLearner] 后台采集线程已启动")

    def stop(self):
        """优雅停止"""
        self.running = False
        if self.thread is not None:
            self.thread.join()
        self.cap.release()
        print("[ConceptLearner] 已停止")

    # -------------------- 主循环 --------------------
    def _capture_loop(self):
        """逐帧处理：检测→记忆→知识图谱→微调"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            # 1. 检测 & 裁剪
            crops, labels = self.perception.detect_and_crop(frame)
            if not crops:
                continue

            # 2. 逐个概念入库
            for img, label in zip(crops, labels):
                key = self.memory.add_image_text(img, label)   # 返回唯一 key
                self.kg.add_triplet(label, "has_example", key)  # 示例三元组
                self.learner.update(img, label)                # 增量 LoRA

    # -------------------- 单次学习接口（API 用） --------------------
    def learn_once(self, bgr_frame):
        """外部单次喂图，无需开摄像头"""
            crops, labels = self.perception.detect_and_crop(bgr_frame)
            for img, label in zip(crops, labels):
                key = self.memory.add_image_text(img, label)
                self.kg.add_triplet(label, "has_example", key)
                self.learner.update(img, label)
            return {"concepts": labels}

    # -------------------- 统计 --------------------
    def get_stats(self):
        return {
            "faiss_count": self.memory.index.ntotal,
            "neo4j_nodes": self.kg.node_count(),
            "lora_steps":  self.learner.global_step,
        }


# 简易 CLI（可选）
if __name__ == "__main__":
    bot = ConceptLearner()
    try:
        bot.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        bot.stop()
