#!/usr/bin/env bash
set -e

# 0. 不论从哪执行，都把 BASE 固定为仓库根目录
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE"

# 1. 可配置变量
PYTHON=python3
VENV_DIR="$BASE/venv"
SDK_DIR="$BASE/yolo_concept_sdk"
NEO4J_URI="bolt://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASS="123456"
API_PORT=8000

# 2. 系统依赖检查
command -v $PYTHON >/dev/null 2>&1 || { echo "请先安装 Python3"; exit 1; }

# 3. 创建并激活虚拟环境
if [ ! -d "$VENV_DIR" ]; then
    $PYTHON -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# 4. 升级 pip & 安装 SDK（可编辑模式）
pip install -U pip setuptools wheel
pip install -e "$SDK_DIR"

# 5. 默认数据目录
mkdir -p "$SDK_DIR/yolo_concept/data"/{faiss,weights,lora}

# 6. 下载 YOLO 权重（若不存在）
WEIGHT="$SDK_DIR/yolo_concept/data/weights/yolov8n.pt"
if [ ! -f "$WEIGHT" ]; then
    echo "下载 YOLOv8n 权重..."
    wget -q -O "$WEIGHT" https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
fi

# 7. 启动 Neo4j（Docker 版）
if command -v docker >/dev/null 2>&1; then
    RUNNING=$(docker ps -q --filter name=neo4j-yolo)
    if [ -z "$RUNNING" ]; then
        echo "启动 Neo4j Docker 容器..."
        docker run -d --rm \
          --name neo4j-yolo \
          -p 7687:7687 -p 7474:7474 \
          -e NEO4J_AUTH="$NEO4J_USER/$NEO4J_PASS" \
          neo4j:5-community
        sleep 10
    fi
else
    echo "未安装 docker，请自行保证 Neo4j 已运行"
fi

# 8. 启动 FastAPI 服务
echo "启动 FastAPI 服务，端口 $API_PORT ..."
uvicorn yolo_concept.api:app --host 0.0.0.0 --port "$API_PORT" --reload
