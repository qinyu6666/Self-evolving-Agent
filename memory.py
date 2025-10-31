import os, json, requests, wikipedia, torch
import clip, faiss, numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
from datetime import datetime

class Memory:
    def __init__(self,
                 faiss_vis_path ="data/vis.index",   # 图向量
                 faiss_txt_path ="data/txt.index",   # 文向量
                 meta_path      ="data/meta.json"):  # 图文对元数据
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 1. 视觉塔
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        # 2. 文本塔
        self.sent_model = SentenceTransformer('all-MiniLM-L6-v2')
        # 3. 双索引
        self.vis_index = faiss.IndexFlatIP(512)   # CLIP 视觉
        self.txt_index = faiss.IndexFlatIP(384)   # MiniLM 文本
        self.meta      = {}                       # id -> {"name":..,"desc":..,"vis_idx":..,"txt_idx":..}
        self.load(faiss_vis_path, faiss_txt_path, meta_path)

    # —— 加载或新建 ——
    def load(self, vp, tp, mp):
        if os.path.exists(vp):
            self.vis_index = faiss.read_index(vp)
            self.txt_index = faiss.read_index(tp)
            self.meta      = json.load(open(mp))
        else:
            self.vis_index = faiss.IndexFlatIP(512)
            self.txt_index = faiss.IndexFlatIP(384)
            self.meta      = {}

    # ---- 判重 ----
    def is_new_concept(self, image_pil, tau=0.75):
        image = self.clip_preprocess(image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            vec = self.clip_model.encode_image(image).cpu().numpy().astype("float32")
            vec /= np.linalg.norm(vec)
        if self.index.ntotal == 0:
            return None, 0.0
        D, I = self.index.search(vec, 1)
        score = D[0][0]
        if score >= tau:
            name = self.meta[str(I[0][0])]["name"]
            return name, score
        return None, score

    # ---- 网络搜索 + 描述 ----
    def search_web_and_describe(self, label: str) -> (str, str):
        """
        输入：YOLO 给出的单词，如 "cactus"
        输出：一段百科摘要 + 分类
        """
        try:
            # 1. 优先 Wikipedia（英文）
            page = wikipedia.page(label, auto_suggest=True)
            desc   = page.summary[:300]          # 截断 300 字
            category = page.categories[0] if page.categories else "object"
        except Exception as e:
            # 2. 维基找不到就 fallback 到 Google 爬虫（简单示例）
            desc = self._google_snippet(label)
            category = "object"
        return desc, category

    # —— 极简 Google 顶部 snippet 爬虫 ——
    def _google_snippet(self, keyword):
        headers = {"User-Agent": "Mozilla/5.0"}
        url = f"https://www.google.com/search?q={keyword}"
        resp = requests.get(url, headers=headers, timeout=5)
        match = re.search(r'<span class="aCOpRe">(.*?)</span>', resp.text)
        snippet = match.group(1) if match else f"{keyword} is a common object."
        return snippet

    # ---- 图文入库 ----
    def add_image_text(self, image_pil, name, desc):
        # 1. 生成全局唯一 id
        uid = self._make_id(image_pil, name)

        # 2. 图向量 512d
        image = self.clip_preprocess(image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            vis_vec = self.clip_model.encode_image(image).cpu().numpy().astype('float32')
            vis_vec /= np.linalg.norm(vis_vec)          # 单位向量
        self.vis_index.add(vis_vec)

        # 3. 文向量 384d
        txt_vec = self.sent_model.encode(desc).astype('float32')
        txt_vec /= np.linalg.norm(txt_vec)
        self.txt_index.add(txt_vec)

        # 4. 记录映射
        vis_idx = self.vis_index.ntotal - 1
        txt_idx = self.txt_index.ntotal - 1
        self.meta[uid] = {
            "name"    : name,
            "desc"    : desc,
            "vis_idx" : vis_idx,
            "txt_idx" : txt_idx,
            "time"    : datetime.now().isoformat()
        }
        self.save(vp="data/vis.index", tp="data/txt.index", mp="data/meta.json")
        print(f"[Memory] 图文对已入库  id={uid}  name={name}")

    def save(self, vp, tp, mp):
        faiss.write_index(self.vis_index, vp)
        faiss.write_index(self.txt_index, tp)
        json.dump(self.meta, open(mp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        
    # —— 辅助：生成唯一 id ——
    def _make_id(self, image_pil, label):
        import hashlib, time
        t = str(time.time_ns())
        px = image_pil.resize((1, 1)).getpixel((0, 0))[0]
        return hashlib.md5(f"{t}_{label}_{px}".encode()).hexdigest()[:16]
