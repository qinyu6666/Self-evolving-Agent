import os
import cv2
import time
import queue
import uuid
import threading
import numpy as np
import mysql.connector
import torch
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForCausalLM
import speech_recognition as sr
import pyttsx3
from PIL import Image
import logging
import gc
import subprocess

# ========================
# 配置和日志设置
# ========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(threadName)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('SmartAgent')


class Config:
    # 摄像头配置
    CAMERA_INDEX = 0
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 15

    # 关键帧配置
    KEYFRAME_DIR = "keyframes"
    DIFF_THRESHOLD = 25  # 帧间差异阈值
    MIN_SAMPLE_INTERVAL = 0.2  # 最小采样间隔(秒)
    MAX_SAMPLE_INTERVAL = 5.0  # 最大采样间隔(秒)
    INIT_SAMPLE_INTERVAL = 1.0  # 初始采样间隔(秒)

    # 模型配置
    YOLO_MODEL = "yolov8n.pt"  # 目标检测模型
    VLM_MODEL_PATH = "E:\\2025\\Qwen2.5-VL-3B-Instruct"  # 视觉语言模型

    # 数据库配置
    DB_HOST = "localhost"
    DB_USER = "root"
    DB_PASSWORD = "123456"
    DB_NAME = "smart_agent_db"

    # 语音配置
    SPEECH_LANGUAGE = "zh-CN"  # 中文
    LISTEN_TIMEOUT = 1.0  # 语音监听超时(秒)
    LISTEN_PHRASE_LIMIT = 5  # 语音最大时长(秒)

    # 系统参数
    CAMERA_IDLE_TIMEOUT = 60  # 摄像头空闲超时(秒)
    MAX_RECENT_FRAMES = 10  # 保留的最近帧数量
    FRAME_QUEUE_SIZE = 5  # 帧队列大小
    PROCESS_QUEUE_SIZE = 3  # 处理队列大小
    MAX_GPU_MEMORY = 0.8  # 最大GPU内存使用率
    MEMORY_CHECK_INTERVAL = 15  # 内存检查间隔(秒)


# 确保关键帧目录存在
os.makedirs(Config.KEYFRAME_DIR, exist_ok=True)


# ========================
# 数据库管理 (带连接池)
# ========================
class DatabaseManager:
    def __init__(self):
        self.conn_pool = mysql.connector.pooling.MySQLConnectionPool(
            pool_name="smart_agent_pool",
            pool_size=3,
            host=Config.DB_HOST,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            database=Config.DB_NAME
        )
        self.create_tables()
        logger.info("数据库连接池已创建")

    def get_connection(self):
        return self.conn_pool.get_connection()

    def create_tables(self):
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # 创建概念图存储表
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS concepts (
                id VARCHAR(36) PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                path VARCHAR(255) NOT NULL,
                description TEXT,
                category VARCHAR(100),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # 创建交互历史表
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                question TEXT,
                answer TEXT,
                image_path VARCHAR(255),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            conn.commit()
            cursor.close()
            conn.close()
            logger.info("数据库表已创建")
        except mysql.connector.Error as err:
            logger.error(f"数据库表创建错误: {err}")

    def save_concept(self, concept_id, name, path, description, category):
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            query = """
            INSERT INTO concepts (id, name, path, description, category)
            VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(query, (concept_id, name, path, description, category))
            conn.commit()
            cursor.close()
            conn.close()
            return True
        except mysql.connector.Error as err:
            logger.error(f"保存概念图错误: {err}")
            return False

    def save_interaction(self, question, answer, image_path):
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            query = """
            INSERT INTO interactions (question, answer, image_path)
            VALUES (%s, %s, %s)
            """
            cursor.execute(query, (question, answer, image_path))
            conn.commit()
            cursor.close()
            conn.close()
            return True
        except mysql.connector.Error as err:
            logger.error(f"保存交互记录错误: {err}")
            return False


# ========================
# 视觉处理模块 (带资源锁)
# ========================
class VisionProcessor:
    def __init__(self):
        # 模型加载锁
        self.model_init_lock = threading.Lock()

        # 初始化目标检测模型
        self.detector = None
        self.detector_lock = threading.Lock()

        # 初始化视觉语言模型
        self.vlm_processor = None
        self.vlm_model = None
        self.vlm_lock = threading.Lock()

        # 设备选择
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用计算设备: {self.device}")

        # 延迟初始化模型
        self.models_initialized = False

    def initialize_models(self):
        """延迟初始化模型，避免启动时占用过多资源"""
        with self.model_init_lock:
            if self.models_initialized:
                return

            logger.info("开始初始化视觉模型...")

            # 初始化YOLOv8目标检测模型
            try:
                logger.info(f"加载目标检测模型: {Config.YOLO_MODEL}")
                self.detector = YOLO(Config.YOLO_MODEL)
            except Exception as e:
                logger.error(f"目标检测模型加载失败: {e}")
                self.detector = None

            # 初始化视觉语言模型
            try:
                logger.info(f"加载视觉语言模型: {Config.VLM_MODEL_PATH}")

                # 使用量化加载以减少显存占用
                self.vlm_processor = AutoProcessor.from_pretrained(Config.VLM_MODEL_PATH)

                self.vlm_model = AutoModelForCausalLM.from_pretrained(
                    Config.VLM_MODEL_PATH,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto",
                    load_in_4bit=True,  # 4位量化
                    bnb_4bit_compute_dtype=torch.float16
                )

                # 预热模型
                if self.vlm_model:
                    self.vlm_model.eval()
                    with torch.no_grad():
                        dummy_image = Image.new('RGB', (100, 100))
                        inputs = self.vlm_processor(
                            text="预热模型",
                            images=dummy_image,
                            return_tensors="pt"
                        ).to(self.device)
                        self.vlm_model.generate(**inputs, max_new_tokens=1)
            except Exception as e:
                logger.error(f"视觉语言模型加载失败: {e}")
                self.vlm_processor = None
                self.vlm_model = None

            self.models_initialized = True
            logger.info("视觉模型初始化完成")

    def detect_objects(self, frame):
        """使用YOLOv8检测图像中的对象 (带锁保护)"""
        if not self.detector:
            self.initialize_models()
            if not self.detector:
                return []

        with self.detector_lock:
            try:
                results = self.detector(frame)
                detections = []

                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()

                    for i, (box, cls_id, conf) in enumerate(zip(boxes, classes, confidences)):
                        x1, y1, x2, y2 = map(int, box)
                        class_name = self.detector.names[int(cls_id)]

                        detections.append({
                            "id": i,
                            "class": class_name,
                            "confidence": float(conf),
                            "bbox": (x1, y1, x2, y2),
                            "image": frame[y1:y2, x1:x2]
                        })

                return detections
            except Exception as e:
                logger.error(f"目标检测错误: {e}")
                return []

    def describe_image(self, frame):
        """使用VLM生成图像描述 (带锁保护)"""
        if not self.vlm_model:
            self.initialize_models()
            if not self.vlm_model:
                return "视觉模型未初始化"

        with self.vlm_lock:
            try:
                # 将OpenCV图像转换为PIL格式
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # 处理图像并生成描述
                inputs = self.vlm_processor(
                    text="详细描述这张图片的内容，包括场景、物体和任何可见的文字。",
                    images=image,
                    return_tensors="pt"
                ).to(self.device)

                generated_ids = self.vlm_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True
                )

                description = self.vlm_processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]

                return description
            except Exception as e:
                logger.error(f"图像描述生成错误: {e}")
                return "无法生成图像描述"

    def answer_question(self, frame, question):
        """使用VLM回答关于图像的问题 (带锁保护)"""
        if not self.vlm_model:
            self.initialize_models()
            if not self.vlm_model:
                return "视觉模型未初始化"

        with self.vlm_lock:
            try:
                # 将OpenCV图像转换为PIL格式
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # 处理图像和问题
                inputs = self.vlm_processor(
                    text=question,
                    images=image,
                    return_tensors="pt"
                ).to(self.device)

                generated_ids = self.vlm_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True
                )

                answer = self.vlm_processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]

                return answer
            except Exception as e:
                logger.error(f"问题回答错误: {e}")
                return "无法回答这个问题"


# ========================
# 语音处理模块 (带资源锁)
# ========================
class SpeechProcessor:
    def __init__(self):
        # 语音识别锁
        self.recognition_lock = threading.Lock()

        # 语音合成锁
        self.synthesis_lock = threading.Lock()

        # 初始化语音识别
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # 初始化语音合成
        self.engine = None

        # 设置语音识别参数
        with self.recognition_lock:
            try:
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source)
                logger.info("语音识别器已初始化")
            except Exception as e:
                logger.error(f"语音识别器初始化失败: {e}")

        # 初始化语音合成
        with self.synthesis_lock:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 150)  # 语速
                self.engine.setProperty('volume', 0.9)  # 音量
                logger.info("语音合成器已初始化")
            except Exception as e:
                logger.error(f"语音合成器初始化失败: {e}")
                self.engine = None

    def listen(self):
        """监听麦克风输入 (带锁保护)"""
        if not self.recognizer:
            return None

        with self.recognition_lock:
            try:
                with self.microphone as source:
                    logger.info("正在聆听...")
                    audio = self.recognizer.listen(
                        source,
                        timeout=Config.LISTEN_TIMEOUT,
                        phrase_time_limit=Config.LISTEN_PHRASE_LIMIT
                    )

                text = self.recognizer.recognize_google(audio, language=Config.SPEECH_LANGUAGE)
                logger.info(f"识别到语音: {text}")
                return text
            except sr.WaitTimeoutError:
                return None  # 没有检测到语音
            except sr.UnknownValueError:
                logger.warning("无法识别语音")
                return None
            except Exception as e:
                logger.error(f"语音识别错误: {e}")
                return None

    def speak(self, text):
        """使用TTS朗读文本 (带锁保护)"""
        if not self.engine:
            return

        with self.synthesis_lock:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
                logger.info(f"语音输出: {text}")
            except Exception as e:
                logger.error(f"语音输出错误: {e}")


# ========================
# 关键帧管理器
# ========================
class KeyframeManager:
    def __init__(self):
        self.last_frame = None
        self.last_time = time.time()
        self.sample_interval = Config.INIT_SAMPLE_INTERVAL
        self.lock = threading.Lock()

    def should_capture(self, frame):
        """判断是否应该捕获关键帧 (带锁保护)"""
        with self.lock:
            current_time = time.time()

            # 检查是否达到采样间隔
            if current_time - self.last_time < self.sample_interval:
                return False

            # 如果是第一帧，直接捕获
            if self.last_frame is None:
                self.last_frame = frame.copy()
                self.last_time = current_time
                return True

            # 计算帧间差异
            diff = self.calculate_frame_diff(self.last_frame, frame)

            # 根据差异调整采样率
            if diff > Config.DIFF_THRESHOLD:
                # 变化大，增加采样率（减少间隔）
                self.sample_interval = max(
                    Config.MIN_SAMPLE_INTERVAL,
                    self.sample_interval * 0.7
                )
            else:
                # 变化小，减少采样率（增加间隔）
                self.sample_interval = min(
                    Config.MAX_SAMPLE_INTERVAL,
                    self.sample_interval * 1.3
                )

            logger.info(f"帧差异: {diff:.2f}, 采样间隔: {self.sample_interval:.2f}s")

            # 如果差异足够大，捕获为关键帧
            if diff > Config.DIFF_THRESHOLD:
                self.last_frame = frame.copy()
                self.last_time = current_time
                return True

            return False

    def calculate_frame_diff(self, frame1, frame2):
        """计算两帧之间的差异"""
        # 转换为灰度图
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # 计算绝对差异
        diff = cv2.absdiff(gray1, gray2)

        # 计算差异平均值
        return np.mean(diff)


# ========================
# 资源监控器
# ========================
class ResourceMonitor:
    def __init__(self, stop_event, vision_processor):
        self.stop_event = stop_event
        self.vision_processor = vision_processor
        self.lock = threading.Lock()
        self.last_check_time = time.time()

    def monitor(self):
        """监控系统资源并采取预防措施"""
        while not self.stop_event.is_set():
            try:
                # 每15秒检查一次
                if time.time() - self.last_check_time > Config.MEMORY_CHECK_INTERVAL:
                    self.check_gpu_memory()
                    self.check_system_memory()
                    self.last_check_time = time.time()

                time.sleep(1)
            except Exception as e:
                logger.error(f"资源监控错误: {e}")
                time.sleep(5)

    def check_gpu_memory(self):
        """检查GPU内存使用情况"""
        if not torch.cuda.is_available():
            return

        try:
            alloc_mem = torch.cuda.memory_allocated() / 1024 ** 3  # GB
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3

            logger.info(f"GPU内存使用: {alloc_mem:.2f}GB / {total_mem:.2f}GB")

            # 如果内存使用超过阈值，尝试清理
            if alloc_mem > total_mem * Config.MAX_GPU_MEMORY:
                logger.warning(f"GPU内存使用超过 {Config.MAX_GPU_MEMORY * 100:.0f}%，清理缓存")
                with self.lock:
                    # 尝试获取模型锁，避免在模型运行时清理
                    if self.vision_processor.model_init_lock.acquire(blocking=False):
                        try:
                            torch.cuda.empty_cache()
                            gc.collect()
                        finally:
                            self.vision_processor.model_init_lock.release()
                    else:
                        logger.warning("模型正在使用，跳过清理GPU缓存")
        except Exception as e:
            logger.error(f"GPU内存检查错误: {e}")

    def check_system_memory(self):
        """检查系统内存使用情况"""
        try:
            # 获取系统内存使用情况（Windows系统）
            result = subprocess.run(['wmic', 'OS', 'get', 'FreePhysicalMemory,TotalVisibleMemorySize'],
                                    capture_output=True, text=True)
            output = result.stdout.strip().split('\n')[-1].split()

            if len(output) >= 2:
                free_mem = int(output[0]) / (1024 * 1024)  # 转换为GB
                total_mem = int(output[1]) / (1024 * 1024)  # 转换为GB
                used_mem = total_mem - free_mem
                usage_percent = (used_mem / total_mem) * 100

                logger.info(f"系统内存使用: {usage_percent:.1f}%")

                # 如果内存使用超过90%，清理缓存
                if usage_percent > 90:
                    logger.warning("系统内存使用超过90%，清理缓存")
                    with self.lock:
                        gc.collect()
        except Exception as e:
            logger.error(f"系统内存检查错误: {e}")


# ========================
# 智能体主系统
# ========================
class SmartAgent:
    def __init__(self):
        # 系统状态
        self.camera_active = True
        self.last_activity_time = time.time()
        self.recent_frames = []  # 存储最近帧和时间戳
        self.recent_frames_lock = threading.Lock()  # 最近帧列表锁
        self.camera_lock = threading.Lock()  # 摄像头资源锁
        self.cap = None  # 摄像头对象

        # 初始化组件
        self.db_manager = DatabaseManager()
        self.vision_processor = VisionProcessor()
        self.speech_processor = SpeechProcessor()
        self.keyframe_manager = KeyframeManager()

        # 创建队列
        self.frame_queue = queue.Queue(maxsize=Config.FRAME_QUEUE_SIZE)
        self.process_queue = queue.Queue(maxsize=Config.PROCESS_QUEUE_SIZE)

        # 创建线程停止标志
        self.stop_event = threading.Event()

        # 资源监控器
        self.resource_monitor = ResourceMonitor(self.stop_event, self.vision_processor)

        # 启动工作线程
        self.capture_thread = threading.Thread(target=self.capture_frames, name="CaptureThread")
        self.process_thread = threading.Thread(target=self.process_frames, name="ProcessThread")
        self.speech_thread = threading.Thread(target=self.monitor_speech, name="SpeechThread")
        self.monitor_thread = threading.Thread(target=self.resource_monitor.monitor, name="MonitorThread")
        self.processing_thread = threading.Thread(target=self.run_processing, name="ProcessingThread")

        self.capture_thread.start()
        self.process_thread.start()
        self.speech_thread.start()
        self.monitor_thread.start()
        self.processing_thread.start()

        logger.info("智能体系统已启动")

    def capture_frames(self):
        """捕获摄像头帧的线程函数"""
        while not self.stop_event.is_set():
            try:
                # 检查摄像头是否需要关闭
                current_time = time.time()
                if self.camera_active and current_time - self.last_activity_time > Config.CAMERA_IDLE_TIMEOUT:
                    logger.info("摄像头空闲超时，关闭摄像头")
                    self.camera_active = False
                    with self.camera_lock:
                        if self.cap:
                            self.cap.release()
                            self.cap = None

                # 检查摄像头是否需要开启
                if not self.camera_active and current_time - self.last_activity_time <= Config.CAMERA_IDLE_TIMEOUT:
                    logger.info("检测到活动，开启摄像头")
                    self.camera_active = True

                # 如果摄像头需要开启但未初始化
                if self.camera_active and self.cap is None:
                    with self.camera_lock:
                        self.cap = cv2.VideoCapture(Config.CAMERA_INDEX)
                        if not self.cap.isOpened():
                            logger.error("无法打开摄像头")
                            time.sleep(1)
                            continue
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
                        self.cap.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)
                        logger.info("摄像头已开启")

                # 如果摄像头未激活，等待
                if not self.camera_active:
                    time.sleep(1)
                    continue

                # 读取帧
                with self.camera_lock:
                    if self.cap is None or not self.cap.isOpened():
                        time.sleep(0.1)
                        continue

                    ret, frame = self.cap.read()
                    if not ret:
                        logger.warning("摄像头读取失败，尝试重新初始化")
                        self.cap.release()
                        self.cap = cv2.VideoCapture(Config.CAMERA_INDEX)
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
                        self.cap.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)
                        time.sleep(0.1)
                        continue

                # 更新最近帧列表（带锁保护）
                timestamp = time.time()
                with self.recent_frames_lock:
                    self.recent_frames.append((timestamp, frame.copy()))
                    # 只保留最近的几帧
                    if len(self.recent_frames) > Config.MAX_RECENT_FRAMES:
                        self.recent_frames.pop(0)

                # 检查是否捕获关键帧
                if self.keyframe_manager.should_capture(frame):
                    try:
                        # 将帧放入队列
                        self.frame_queue.put((timestamp, frame.copy()), timeout=0.5)
                    except queue.Full:
                        logger.warning("帧队列已满，丢弃帧")

                # 显示实时画面
                cv2.imshow("Smart Agent", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()

            except Exception as e:
                logger.error(f"捕获帧错误: {e}")
                time.sleep(0.5)

        # 清理摄像头资源
        with self.camera_lock:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        cv2.destroyAllWindows()

    def process_frames(self):
        """处理关键帧的线程函数"""
        while not self.stop_event.is_set():
            try:
                # 从队列获取帧
                timestamp, frame = self.frame_queue.get(timeout=1.0)

                # 创建处理任务
                task = {
                    "type": "auto_describe",
                    "timestamp": timestamp,
                    "frame": frame,
                    "frame_path": self.save_frame(frame)
                }

                # 将任务放入处理队列
                self.process_queue.put(task)

                # 处理目标检测
                threading.Thread(target=self.process_detection, args=(frame,)).start()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"处理帧错误: {e}")

    def process_detection(self, frame):
        """处理目标检测和概念提取"""
        try:
            detections = self.vision_processor.detect_objects(frame)

            for obj in detections:
                # 生成唯一ID
                concept_id = str(uuid.uuid4())

                # 保存对象图像
                obj_path = os.path.join(Config.KEYFRAME_DIR, f"{concept_id}.jpg")
                cv2.imwrite(obj_path, obj["image"])

                # 生成描述（这里简化处理，实际中可以使用VLM生成更丰富的描述）
                description = f"检测到 {obj['class']}，置信度 {obj['confidence']:.2f}"

                # 保存到数据库
                self.db_manager.save_concept(
                    concept_id=concept_id,
                    name=obj["class"],
                    path=obj_path,
                    description=description,
                    category=obj["class"]
                )

                logger.info(f"保存概念: {obj['class']} ({concept_id})")
        except Exception as e:
            logger.error(f"目标检测处理错误: {e}")

    def monitor_speech(self):
        """监听语音输入的线程函数"""
        while not self.stop_event.is_set():
            try:
                # 检测语音输入
                speech_text = self.speech_processor.listen()

                if speech_text:
                    # 更新最后活动时间
                    self.last_activity_time = time.time()

                    # 获取最近的帧（带锁保护）
                    with self.recent_frames_lock:
                        if self.recent_frames:
                            # 获取语音输入时间最近的帧
                            current_time = time.time()
                            closest_frame = min(
                                self.recent_frames,
                                key=lambda x: abs(x[0] - current_time)
                            )
                            timestamp, frame = closest_frame
                        else:
                            frame = None

                    if frame is not None:
                        # 创建处理任务
                        task = {
                            "type": "question",
                            "timestamp": timestamp,
                            "frame": frame,
                            "question": speech_text,
                            "frame_path": self.save_frame(frame)
                        }

                        # 将任务放入处理队列
                        self.process_queue.put(task)
            except Exception as e:
                logger.error(f"语音监听错误: {e}")
                time.sleep(0.5)

    def run_processing(self):
        """处理队列中的任务"""
        while not self.stop_event.is_set():
            try:
                # 从队列获取任务
                task = self.process_queue.get(timeout=1.0)

                if task["type"] == "auto_describe":
                    # 自动描述图像
                    description = self.vision_processor.describe_image(task["frame"])
                    logger.info(f"图像描述: {description}")

                    # 生成问题
                    question = f"我看到了: {description}。你有什么问题吗？"

                    # 语音提问
                    self.speech_processor.speak(question)

                    # 保存交互记录
                    self.db_manager.save_interaction(
                        question=question,
                        answer="",
                        image_path=task["frame_path"]
                    )

                elif task["type"] == "question":
                    # 回答用户问题
                    answer = self.vision_processor.answer_question(
                        task["frame"],
                        task["question"]
                    )

                    # 语音回答
                    self.speech_processor.speak(answer)

                    # 保存交互记录
                    self.db_manager.save_interaction(
                        question=task["question"],
                        answer=answer,
                        image_path=task["frame_path"]
                    )

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"任务处理错误: {e}")

    def save_frame(self, frame):
        """保存帧到文件"""
        frame_id = str(uuid.uuid4())
        frame_path = os.path.join(Config.KEYFRAME_DIR, f"{frame_id}.jpg")
        cv2.imwrite(frame_path, frame)
        return frame_path

    def stop(self):
        """停止系统"""
        logger.info("正在停止智能体系统...")
        self.stop_event.set()

        # 等待线程结束
        self.capture_thread.join(timeout=2.0)
        self.process_thread.join(timeout=2.0)
        self.speech_thread.join(timeout=2.0)
        self.monitor_thread.join(timeout=2.0)
        self.processing_thread.join(timeout=2.0)

        logger.info("智能体系统已停止")


# ========================
# 主程序
# ========================
if __name__ == "__main__":
    logger.info("启动智能体系统...")
    agent = SmartAgent()

    try:
        # 主循环
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        agent.stop()




