from setuptools import setup, find_packages
setup(
    name="yolo-concept",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "ultralytics>=8.0",
        "faiss-cpu",
        "neo4j",
        "fastapi", "uvicorn",         # API 依赖也写这里，方便一步装好
        "torch", "transformers",
        "lmdb", "pillow", "numpy"
    ],
    entry_points={
        "console_scripts": [
            "yolo-concept=yolo_concept.cli:main"   # 可选 CLI
        ]
    },
)
