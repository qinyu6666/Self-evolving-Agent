"""
阻塞修复 + 数据库写入修复版（完整可运行）
"""
import os
import sys
import cv2
import time
import uuid
import queue
import threading
import traceback
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pymysql
import sounddevice as sd
import torch
import faiss
from dotenv import load_dotenv
from ultralytics import YOLO
from PIL import Image
from transformers import (
    pipeline,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 关闭冗余日志
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

load_dotenv()

# -------------------- 配置 --------------------
CFG = dict(
    camera_index=int(os.getenv("CAMERA_INDEX", 0)),
    min_sound=int(os.getenv("MIN_SOUND_THRESHOLD", 40)),
    inactivity_timeout=int(os.getenv("INACTIVITY_TIMEOUT", 30)),
    db=dict(
        host=os.getenv("MYSQL_HOST"),
        port=int(os.getenv("MYSQL_PORT", 3306)),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DATABASE"),
        charset="utf8mb4",
        autocommit=True,
    ),
    yolo_path=os.getenv("YOLO_MODEL", "yolov8n.pt"),
    vlm_path=os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"),
    asr_path=os.getenv("ASR_MODEL", "openai/whisper-tiny"),
)

# 全局变量
frame_queue = queue.Queue(maxsize=5)
user_q = queue.Queue()
tts_q = queue.Queue()
camera_active = False
last_activity = time.time()

# -------------------- 工具 --------------------
def get_conn():
    """获取 pymysql 连接，带保活"""
    try:
        conn = pymysql.connect(**CFG["db"])
        conn.ping(reconnect=True)
        return conn
    except pymysql.Error as err:
        print(f"数据库连接错误: {err}")
        return None

def save_concept(concept: dict):
    conn = get_conn()
    if conn is None:
        logging.error("数据库连接失败")
        return
    try:
        with conn.cursor() as cur:
            # _, img_encoded = cv2.imencode('.jpg', cv2.imread(concept["image_path"]))
            # blob = img_encoded.tobytes()
            if isinstance(concept["blob"],np.ndarray):
                concept["blob"] = concept["blob"].tobytes()

            sql = """
                INSERT INTO concept_images
                (id, name, image_blob, description, category, frame_time)
                VALUES (%s,%s,%s,%s,%s,%s)
            """
            cur.execute(sql, (
                concept["id"],
                concept["name"][:256],
                # blob,
                concept['blob'],
                concept["description"][:1024],
                concept["category"][:100],
                concept["frame_time"],
            ))
            conn.commit()
            print(f"帧已保存到数据库: {datetime.now()}")
    except Exception as e:
        conn.rollback()
        logging.error("写入数据库失败: %s", e)
        traceback.print_exc()
    finally:
        conn.close()

def save_qa(question, answer, image_id=None):
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            sql = "INSERT INTO qa_history (question, answer, image_id) VALUES (%s,%s,%s)"
            cur.execute(sql, (question, answer, image_id))
            conn.commit()
    except Exception as e:
        logging.error("写入 qa_history 失败: %s", e)

# -------------------- 模型初始化 --------------------
def init_models():
    logging.info("加载模型...")
    yolo = YOLO(CFG["yolo_path"])
    # vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     CFG["vlm_path"], torch_dtype="auto", device_map="auto"
    # )
    vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "E:\\2025\\Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
    )
    # vlm_proc = AutoProcessor.from_pretrained(CFG["vlm_path"])
    vlm_proc = AutoProcessor.from_pretrained("E:\\2025\\Qwen2.5-VL-3B-Instruct")

    asr = pipeline(
        "automatic-speech-recognition",
        model=CFG["asr_path"],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    faiss_idx = faiss.IndexFlatL2(512)
    logging.info("模型加载完成")
    return dict(yolo=yolo, vlm_model=vlm_model, vlm_proc=vlm_proc, asr=asr, faiss=faiss_idx)

# -------------------- TTS --------------------
def tts_worker():
    import pyttsx4
    engine = pyttsx4.init()
    engine.setProperty("rate", 180)
    engine.say("语音系统已启动")
    engine.runAndWait()
    while True:
        text = tts_q.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()

def speak(text):
    tts_q.put(text)

# -------------------- 语音监听 --------------------
def detect_sound(timeout=0.2, fs=16000):
    try:
        data = sd.rec(int(timeout * fs), samplerate=fs, channels=1, dtype="float32")
        sd.wait()
        rms = np.sqrt(np.mean(data ** 2))
        db = 20 * np.log10(rms) if rms > 0 else -60
        return db > CFG["min_sound"]
    except Exception as e:
        logging.error("detect_sound: %s", e)
        return False

def voice_listener(models):
    logging.info("语音监听启动")
    while True:
        if detect_sound():
            logging.info("检测到声音，录音 3 秒...")
            fs = 16000
            audio = sd.rec(int(3 * fs), samplerate=fs, channels=1, dtype="float32")
            sd.wait()
            try:
                text = models["asr"](audio.squeeze())["text"].strip()
                if text:
                    logging.info("识别: %s", text)
                    user_q.put(text)
                    globals()["last_activity"] = time.time()
            except Exception as e:
                logging.error("ASR: %s", e)
        time.sleep(0.5)

# -------------------- 视觉任务 --------------------
prev_frame, last_sample, interval = None, 0, 1.0

def is_keyframe(curr):
    global prev_frame, last_sample, interval
    if prev_frame is None:
        prev_frame = curr
        last_sample = time.time()
        return True
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_prev, gray_curr)
    _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    ratio = np.sum(th) / (255 * th.size)

    if ratio > 0.3:
        interval = 0.2
    elif ratio > 0.1:
        interval = 0.5
    else:
        interval = 1.0

    if time.time() - last_sample > interval:
        prev_frame = curr
        last_sample = time.time()
        return True
    return False

# -------------------- 任务函数 --------------------
# def task_detect(frm, models):
#     try:
#         results = models["yolo"](frm, verbose=False)
#         if not results or len(results[0].boxes) == 0:
#             return
#
#         for box, cls_idx, conf in zip(
#             results[0].boxes.xyxy.cpu().numpy(),
#             results[0].boxes.cls.cpu().numpy(),
#             results[0].boxes.conf.cpu().numpy(),
#         ):
#             if conf < 0.5:
#                 continue
#             x1, y1, x2, y2 = map(int, box)
#             if x2 <= x1 or y2 <= y1:
#                 continue
#
#             crop = frm[y1:y2, x1:x2]
#             crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).convert("RGB")
#             crop_pil = crop_pil.resize((448, 448))
#
#             # 描述
#             messages = [
#                 {
#                     "role": "user",
#                     "content": "<|vision_start|><|image_pad|><|vision_end|>Describe this object in detail."
#                 }
#             ]
#             text = models["vlm_proc"].apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#             inputs = models["vlm_proc"](text=[text], images=[crop_pil], return_tensors="pt").to(models["vlm_model"].device)
#             outs = models["vlm_model"].generate(
#                 **inputs,
#                 max_new_tokens=64,
#                 do_sample=False,
#                 pad_token_id=models["vlm_proc"].tokenizer.eos_token_id
#             )
#             desc = models["vlm_proc"].batch_decode(
#                 outs[:, inputs["input_ids"].shape[1]:],
#                 skip_special_tokens=True
#             )[0].strip()
#
#             # 名称
#             messages2 = [
#                 {
#                     "role": "user",
#                     "content": "<|vision_start|><|image_pad|><|vision_end|>What is this object? Answer with a single noun."
#                 }
#             ]
#             text2 = models["vlm_proc"].apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
#             inputs2 = models["vlm_proc"](text=[text2], images=[crop_pil], return_tensors="pt").to(models["vlm_model"].device)
#             outs2 = models["vlm_model"].generate(
#                 **inputs2,
#                 max_new_tokens=8,
#                 do_sample=False,
#                 pad_token_id=models["vlm_proc"].tokenizer.eos_token_id
#             )
#             name = models["vlm_proc"].batch_decode(
#                 outs2[:, inputs2["input_ids"].shape[1]:],
#                 skip_special_tokens=True
#             )[0].strip()
#
#             cid = str(uuid.uuid4())
#             Path("concepts").mkdir(exist_ok=True)
#             path = f"concepts/{cid}.jpg"
#             cv2.imwrite(path, crop)
#             # add  new
#             # os.fsync(open(path, 'rb').fileno())  # 强制刷盘
#             # threading.Thread(target=save_concept, args=({...},)).start()
#             save_concept({
#                 "id": cid,
#                 "name": name,
#                 "image_path": path,
#                 "description": desc,
#                 "category": results[0].names[int(cls_idx)],
#                 "frame_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             })
#
#             vec = np.random.rand(512).astype("float32")
#             models["faiss"].add(np.array([vec]))
#     except Exception as e:
#         logging.error("detect: %s", e)
#         traceback.print_exc()

def task_detect(frm, models):
    try:
        results = models["yolo"](frm, verbose=False)
        if not results or len(results[0].boxes) == 0:
            return

        for box, cls_idx, conf in zip(
            results[0].boxes.xyxy.cpu().numpy(),
            results[0].boxes.cls.cpu().numpy(),
            results[0].boxes.conf.cpu().numpy(),
        ):
            if conf < 0.5:
                continue
            x1, y1, x2, y2 = map(int, box)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frm[y1:y2, x1:x2]

            if crop.size == 0:
                logging.warning("裁剪空，跳过")
                continue

            # 直接编码成 jpg 二进制，不落盘
            ok, buf = cv2.imencode('.jpg', crop)
            if not ok:
                logging.error("imencode failed")
                continue
            blob = buf.tobytes()


            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).convert("RGB")
            crop_pil = crop_pil.resize((448, 448))

            # 描述
            messages = [
                {
                    "role": "user",
                    "content": "<|vision_start|><|image_pad|><|vision_end|>Describe this object in detail."
                }
            ]
            text = models["vlm_proc"].apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = models["vlm_proc"](text=[text], images=[crop_pil], return_tensors="pt").to(models["vlm_model"].device)
            outs = models["vlm_model"].generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=models["vlm_proc"].tokenizer.eos_token_id
            )
            desc = models["vlm_proc"].batch_decode(
                outs[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )[0].strip()

            # 名称
            messages2 = [
                {
                    "role": "user",
                    "content": "<|vision_start|><|image_pad|><|vision_end|>What is this object? Answer with a single noun."
                }
            ]
            text2 = models["vlm_proc"].apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
            inputs2 = models["vlm_proc"](text=[text2], images=[crop_pil], return_tensors="pt").to(models["vlm_model"].device)
            outs2 = models["vlm_model"].generate(
                **inputs2,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=models["vlm_proc"].tokenizer.eos_token_id
            )
            name = models["vlm_proc"].batch_decode(
                outs2[:, inputs2["input_ids"].shape[1]:],
                skip_special_tokens=True
            )[0].strip()

            cid = str(uuid.uuid4())
            Path("concepts").mkdir(exist_ok=True)
            path = f"concepts/{cid}.jpg"
            cv2.imwrite(path, crop)
            # add  new
            # os.fsync(open(path, 'rb').fileno())  # 强制刷盘
            # threading.Thread(target=save_concept, args=({...},)).start()
            concept = {
                "id": cid,
                "name": name,
                "blob": blob,
                "description": desc,
                "category": results[0].names[int(cls_idx)],
                "frame_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            logging.info("启动保存线程。。。")

            # 启动线程保存到数据库
            # threading.Thread(target=save_concept, args=(concept,), daemon=True).start()
            t = threading.Thread(target=save_concept, args=(concept,))
            # t.daemon = True
            t.start()
            t.join(timeout=5)


            vec = np.random.rand(512).astype("float32")
            models["faiss"].add(np.array([vec]))
    except Exception as e:
        logging.error("detect: %s", e)
        traceback.print_exc()


def task_describe(frm, models):
    try:
        pil = Image.fromarray(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)).convert("RGB")
        pil = pil.resize((448, 448))

        messages = [
            {
                "role": "user",
                "content": "<|vision_start|><|image_pad|><|vision_end|>Describe this scene in detail."
            }
        ]
        text = models["vlm_proc"].apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = models["vlm_proc"](text=[text], images=[pil], return_tensors="pt").to(models["vlm_model"].device)
        outs = models["vlm_model"].generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=models["vlm_proc"].tokenizer.eos_token_id
        )
        desc = models["vlm_proc"].batch_decode(
            outs[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )[0].strip()

        q = f"I see {desc}. Is there anything I can help you with?"
        speak(q)
        save_qa(q, "[AUTO-PROMPT]")
    except Exception as e:
        logging.error("describe: %s", e)
        traceback.print_exc()

def task_qa(frm, question, models):
    try:
        pil = Image.fromarray(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)).convert("RGB")
        pil = pil.resize((448, 448))

        messages = [
            {
                "role": "user",
                "content": f"<|vision_start|><|image_pad|><|vision_end|>{question}"
            }
        ]
        text = models["vlm_proc"].apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = models["vlm_proc"](text=[text], images=[pil], return_tensors="pt").to(models["vlm_model"].device)
        outs = models["vlm_model"].generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=models["vlm_proc"].tokenizer.eos_token_id
        )
        answer = models["vlm_proc"].batch_decode(
            outs[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )[0].strip()
        speak(answer)
        save_qa(question, answer)
    except Exception as e:
        logging.error("qa: %s", e)
        traceback.print_exc()

# -------------------- 主循环 --------------------
def camera_loop(models):
    global camera_active, last_activity
    cap = cv2.VideoCapture(CFG["camera_index"], cv2.CAP_DSHOW)
    if not cap.isOpened():
        logging.error("摄像头打开失败")
        return
    logging.info("摄像头就绪")

    # 添加帧计数器
    frame_count = 0
    try:
        while True:
            now = time.time()
            if not camera_active and user_q.empty():
                user_q.put("start")

            if camera_active and now - last_activity > CFG["inactivity_timeout"]:
                camera_active = False
                logging.info("无活动超时，摄像头关闭")

            if not user_q.empty():
                _ = user_q.get()
                camera_active = True
                last_activity = now

            if camera_active:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue

                frame_count += 1
                if frame_count % 10 == 0:
                    cv2.imshow("Camera Feed", frame)
                if cv2.waitKey(1) == 27:
                    break
                if is_keyframe(frame) and not frame_queue.full():
                    frame_queue.put(frame.copy())
                    last_activity = now

            while not frame_queue.empty():
                frm = frame_queue.get()

                #交替执行检测和描述任务
                if frame_count % 2 == 0:
                    threading.Thread(target=task_detect, args=(frm, models), daemon=False).start()
                else:
                    threading.Thread(target=task_describe, args=(frm, models), daemon=False).start()

                if not user_q.empty():
                    q = user_q.get()
                    threading.Thread(target=task_qa, args=(frm, q, models), daemon=False).start()

            time.sleep(0.05)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tts_q.put(None)

# -------------------- main --------------------
def main():
    try:
        conn = get_conn()
        conn.close()
    except Exception:
        logging.error("数据库连接失败")
        traceback.print_exc()
        return

    models = init_models()
    threading.Thread(target=tts_worker, daemon=False).start()
    threading.Thread(target=voice_listener, args=(models,), daemon=False).start()
    camera_loop(models)

if __name__ == "__main__":
    main()
