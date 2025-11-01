import cv2
import pymysql
import time
from datetime import datetime
import os
import matplotlib.pyplot as plt

# 数据库配置
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "123456",
    "database": "frame_db",
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor
}


# 创建数据库连接
def create_db_connection():
    try:
        conn = pymysql.connect(**DB_CONFIG)
        return conn
    except pymysql.Error as err:
        print(f"数据库连接错误: {err}")
        return None


# 初始化数据库（创建数据库和表）
def init_database():
    conn = None
    try:
        # 临时连接（不指定数据库）
        conn = pymysql.connect(
            host=DB_CONFIG["host"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"]
        )
        cursor = conn.cursor()

        # 创建数据库
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
        # 切换到新数据库
        cursor.execute(f"USE {DB_CONFIG['database']}")

        # 创建表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS captured_frames (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp DATETIME NOT NULL,
            frame_data LONGBLOB NOT NULL
        )
        """)
        conn.commit()
        print("数据库初始化成功")

    except pymysql.Error as err:
        print(f"数据库初始化失败: {err}")
    finally:
        if conn:
            conn.close()


# 保存帧到数据库
def save_frame_to_db(frame):
    conn = None
    try:
        conn = create_db_connection()
        if conn is None:
            return False

        with conn.cursor() as cursor:
            # 将图像转换为二进制数据
            _, img_encoded = cv2.imencode('.jpg', frame)
            frame_binary = img_encoded.tobytes()

            # 插入记录
            timestamp = datetime.now()
            cursor.execute(
                "INSERT INTO captured_frames (timestamp, frame_data) VALUES (%s, %s)",
                (timestamp, frame_binary)
            )
            conn.commit()
            print(f"帧已保存到数据库: {timestamp}")
            return True
    except pymysql.Error as err:
        print(f"数据库操作错误: {err}")
        return False
    finally:
        if conn:
            conn.close()


# 主捕获函数
def capture_frames():
    # 初始化数据库
    init_database()

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    print("开始捕获视频流（按ESC键退出）...")
    last_capture_time = time.time()

    cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)

    try:
        while True:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print("无法获取帧")
                break

            # 显示实时画面
            # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # plt.show()
            cv2.imshow('Camera Feed', frame)

            # 每5秒捕获一帧
            current_time = time.time()
            if current_time - last_capture_time >= 5:
                save_frame_to_db(frame)
                last_capture_time = current_time

            #按ESC键退出
            if cv2.waitKey(1) == 27:
                break
    finally:
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        print("程序已终止")


# 主函数
if __name__ == "__main__":
    # 创建必要目录
    os.makedirs("logs", exist_ok=True)

    # 启动捕获
    capture_frames()
