from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from yolo_concept import ConceptLearner
import uvicorn, io, cv2, numpy as np
from PIL import Image

app = FastAPI(title="YOLO-Concept API", version="0.1.0")
bot = ConceptLearner()          # 全局单例

@app.on_event("startup")
async def startup():
    bot.start()                 # 启动后台采集线程

@app.post("/learn")
async def learn(file: UploadFile = File(...)):
    """单张图立即学习"""
    img = Image.open(io.BytesIO(await file.read()))
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    bot.learn_once(frame)       # 在 core.py 里加一个小封装
    return {"status": "submitted"}

@app.get("/stats")
async def stats():
    return bot.get_stats()      # 返回已学概念数、Neo4j 节点数等
