CREATE DATABASE IF NOT EXISTS vision_db;

USE vision_db;

-- 概念图表
CREATE TABLE IF NOT EXISTS concept_images (
    id CHAR(36) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
--    image_path TEXT NOT NULL,
    image_blob LONGBLOB NOT NULL,
    description TEXT,
    category VARCHAR(100) NOT NULL,
    frame_time DATETIME NOT NULL
);

-- 问答历史表
CREATE TABLE IF NOT EXISTS qa_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    image_id CHAR(36),
    timestamp DATETIME NOT NULL,
    FOREIGN KEY (image_id) REFERENCES concept_images(id)
);

-- 创建向量存储表 (在内存中处理，这里只存储元数据)
CREATE TABLE IF NOT EXISTS vectors (
    id CHAR(36) PRIMARY KEY,
    vector_data BLOB,
    FOREIGN KEY (id) REFERENCES concept_images(id)
);
