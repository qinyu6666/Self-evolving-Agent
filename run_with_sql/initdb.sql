CREATE DATABASE IF NOT EXISTS vision_voice_db
  DEFAULT CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE vision_voice_db;

CREATE TABLE IF NOT EXISTS concept_images (
    id         VARCHAR(36) PRIMARY KEY,
    name       VARCHAR(255) NOT NULL,
    image_blob LONGBLOB     NOT NULL,
    description TEXT,
    category   VARCHAR(100) NOT NULL,
    frame_time DATETIME     NOT NULL
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS qa_history (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    question   TEXT         NOT NULL,
    answer     TEXT         NOT NULL,
    image_id   VARCHAR(36),
    timestamp  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (image_id) REFERENCES concept_images(id)
    ON DELETE SET NULL
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS vectors (
    id         VARCHAR(36) PRIMARY KEY,
    vector_data BLOB,
    FOREIGN KEY (id) REFERENCES concept_images(id)
    ON DELETE CASCADE
) ENGINE=InnoDB;
