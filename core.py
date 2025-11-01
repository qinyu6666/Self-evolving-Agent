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

    
    # ---------- 内部工具 ----------
    def _load_or_create_index(self):
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.idmap_path) as f:
                self.id2path = json.load(f)
        else:
            dim = self.clip.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatIP(dim)   # 内积 = cosine（已归一化）
            self.id2path = {}

    def _save_index(self):
        faiss.write_index(self.index, str(self.index_path))
        with open(self.idmap_path, "w") as f:
            json.dump(self.id2path, f, indent=2, ensure_ascii=False)

    # ---------- 核心入口 ----------
    def perceive(self, img_path: str | Path):
        """
        单张图片：检测 → 裁图 → 编码 → 双库写入
        """
        img_path = Path(img_path).resolve()
        img_cv = cv2.imread(str(img_path))
        assert img_cv is not None, f"cannot read {img_path}"

        # 1. YOLO 检测
        results = self.yolo(img_cv, verbose=False)   # list[Results]
        if not results or len(results[0].boxes) == 0:
            print(f"[perceive] no object found in {img_path.name}")
            return

        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)  # [N,4]
        crops = [img_cv[y1:y2, x1:x2] for x1, y1, x2, y2 in boxes]

        # 2. CLIP 编码（归一化）
        embs = self.clip.encode(crops, convert_to_numpy=True, normalize_embeddings=True)

        # 3. 写入 Faiss
        next_id = self.index.ntotal
        self.index.add(embs)

        # 4. 写入 Neo4j（节点 + 关系）
        with self.driver.session() as sess:
            for i, (emb, crop) in enumerate(zip(embs, crops)):
                node_id = str(next_id + i)
                # 向量转 list 以便 JSON 序列化
                sess.execute_write(self._create_node, node_id, img_path, emb.tolist())

        # 5. 更新 id2path 并落盘
        for i in range(len(crops)):
            self.id2path[str(next_id + i)] = str(img_path)
        self._save_index()
        print(f"[perceive] {len(crops)} objects embedded & stored.")

    # ---------- Neo4j 事务 ----------
    @staticmethod
    def _create_node(tx, node_id: str, img_path: Path, embedding: list):
        tx.run(
            """
            MERGE (i:Image {path: $img_path})
            CREATE (o:Object {id: $node_id, embedding: $emb})
            CREATE (i)-[:CONTAINS]->(o)
            """,
            img_path=str(img_path),
            node_id=node_id,
            emb=embedding,
        )

    # ---------- 优雅关闭 ----------
    def close(self):
        self.driver.close()

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

    def learn_from_image(agent: ConceptLearner,
                     img_path: str,
                     concepts: list[str] | None = None):
        agent.perceive(img_path)          # 自动检测 + 入库
        if concepts:
            agent.annotate(img_path, concepts)

   
    

# 简易 CLI（可选）
if __name__ == "__main__":
    bot = ConceptLearner()
    try:
        bot.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        bot.stop()
