import os, torch, threading, time, cv2
from perception import Perception
from memory import Memory
from kg_builder import KGBuilder
from learner import Learner
from async_worker import start_worker, submit_learning_task, shutdown

class AutoLearnAgent:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.perception = Perception()
        self.memory = Memory()
        self.kg = KGBuilder()
        self.learner = Learner()
        self.running = True
        self.frame = None
        self.lock = threading.Lock()

    # ---------- 1. 摄像头线程 ----------
    def _cam_thread(self):
        print("[Camera] 30 FPS 采集线程启动")
        while self.running:
            ret, frame = self.perception.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            cv2.waitKey(33)  # ~30 FPS

    # ---------- 2. 主逻辑线程 ----------
    def _main_thread(self):
        print("[Agent] 主逻辑线程启动")
        while self.running:
            with self.lock:
                frame = self.frame
            if frame is None:
                continue
            crop, label = self.perception.detect_and_crop(frame)
            if crop is None:
                continue
            # 判重（内存操作，<10 ms）
            name, score = self.memory.is_new_concept(crop)
            if name is not None:
                continue
            # 自动描述（网络 IO，也扔后台）
            desc, category = self.memory.search_web_and_describe(label)
            # 把“重活”塞进队列，主线程立即返回
            submit_learning_task(crop, label, desc, category)
            print(f"[Queue] 已提交学习任务 {label}")

    # ---------------- 主循环 ----------------
    def run(self):
        start_worker()                                #  先起后台线程池
        t1 = threading.Thread(target=self._cam_thread, daemon=True)
        t2 = threading.Thread(target=self._main_thread, daemon=True)
        t1.start(); t2.start()
        try:
            t1.join(); t2.join()
        except KeyboardInterrupt:
            self.running = False
            shutdown()
            print("[Agent] 摄像头启动，把镜头对准任意物体即可自动学习……")
        # 单线程阻塞版    
        # while self.running:
        #     frame = self.perception.capture_keyframe()
        #     if frame is None:
        #         time.sleep(0.5); continue
        #     crop, label = self.perception.detect_and_crop(frame)
        #     if crop is None:
        #         time.sleep(0.5); continue
        #     name, score = self.memory.is_new_concept(crop)
        #     if name is not None:          # 已存在
        #         continue
        #     # ------ 新知识 ------
        #     print(f"[New] 发现新概念，正在检索知识……")
        #     desc, category = self.memory.search_web_and_describe(label)
        #     if desc is None:
        #         continue
        #     # 三元组 → Neo4j
        #     triples = self.kg.extract_triples(desc)
        #     self.kg.write_to_neo4j(name=label, category=category, triples=triples)
        #     # 图文向量 → Faiss
        #     self.memory.add_image_text(crop, label, desc)
        #     # 微调 LoRA
        #     self.learner.update(label, desc)
        #     print(f"[Learned] 已完成 {label} 的学习并更新模型！")
    # 优雅退出
    def stop(self):
        self.running = False

if __name__ == "__main__":
    agent = AutoLearnAgent()
    agent.run()

    # os.makedirs("data/lora", exist_ok=True)
    # agent = AutoLearnAgent()
    # try:
    #     agent.run()
    # except KeyboardInterrupt:
    #     agent.stop()