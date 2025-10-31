import threading, queue, concurrent.futures as fut
from memory import Memory
from kg_builder import KGBuilder
from learner import Learner

# 全局任务队列
_TASK_Q = queue.Queue(maxsize=200)

# 线程池：IO 密集任务 8 线程，计算任务 2 线程
_io_pool   = fut.ThreadPoolExecutor(max_workers=8)
_calc_pool = fut.ThreadPoolExecutor(max_workers=2)

def _bg_loop():
    """后台守护线程：不断从队列拿任务并提交到线程池"""
    while True:
        job = _TASK_Q.get()
        if job is None:          # 优雅退出信号
            break
        _io_pool.submit(_do_learn, *job)

def _do_learn(crop_pil, label, desc, category):
    """真正的重活：网络搜索、三元组、入库、微调"""
    mem   = Memory()          # 单例，内部有锁
    kg    = KGBuilder()
    learn = Learner()

    # 1. 图文对向量化（轻量，放 IO 池）
    uid = mem.add_image_text(crop_pil, label, desc)

    # 2. 三元组 → Neo4j（IO 池）
    triples = kg.extract_triples(desc)
    kg.write_to_neo4j(name=label, category=category, triples=triples)

    # 3. LoRA 微调（计算重，放计算池）
    _calc_pool.submit(learn.update, label, desc)

def start_worker():
    t = threading.Thread(target=_bg_loop, daemon=True)
    t.start()
    return t

def submit_learning_task(crop_pil, label, desc, category):
    _TASK_Q.put((crop_pil, label, desc, category))

def shutdown():
    _TASK_Q.put(None)