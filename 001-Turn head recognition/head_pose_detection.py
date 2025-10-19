#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
转头识别系统 - Head Pose Detection System
实时检测头部姿态，包括上下、左右转动和旋转角度
"""

import cv2
import mediapipe as mp
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont


class HeadPoseDetector:
    def __init__(self):
        """初始化 MediaPipe Face Mesh 和相机参数"""
        # 初始化 MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 3D 人脸模型关键点（标准人脸模型坐标）
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # 鼻尖
            (0.0, -330.0, -65.0),        # 下巴
            (-225.0, 170.0, -135.0),     # 左眼左角
            (225.0, 170.0, -135.0),      # 右眼右角
            (-150.0, -150.0, -125.0),    # 左嘴角
            (150.0, -150.0, -125.0)      # 右嘴角
        ], dtype=np.float64)

        # MediaPipe Face Mesh 关键点索引
        # 对应上述 3D 模型点
        self.face_mesh_indices = [1, 152, 33, 263, 61, 291]

        # 加载中文字体
        try:
            # macOS 系统字体路径
            self.font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 28)
            self.font_small = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 20)
        except:
            try:
                # 备用字体
                self.font = ImageFont.truetype("/System/Library/Fonts/STHeiti Light.ttc", 28)
                self.font_small = ImageFont.truetype("/System/Library/Fonts/STHeiti Light.ttc", 20)
            except:
                # 如果都失败，使用默认字体
                self.font = ImageFont.load_default()
                self.font_small = ImageFont.load_default()
                print("警告: 无法加载中文字体，使用默认字体")

    def get_head_pose(self, image):
        """
        检测头部姿态

        Args:
            image: 输入图像 (BGR格式)

        Returns:
            tuple: (success, pitch, yaw, roll, image_points)
            - success: 是否成功检测
            - pitch: 俯仰角 (上下点头)
            - yaw: 偏航角 (左右转头)
            - roll: 翻滚角 (左右倾斜)
            - image_points: 2D 关键点坐标
        """
        # 转换为 RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return False, 0, 0, 0, None

        # 获取图像尺寸
        img_h, img_w = image.shape[:2]

        # 获取人脸关键点
        face_landmarks = results.multi_face_landmarks[0]

        # 提取 2D 关键点
        image_points = []
        for idx in self.face_mesh_indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * img_w)
            y = int(landmark.y * img_h)
            image_points.append([x, y])

        image_points = np.array(image_points, dtype=np.float64)

        # 相机内参矩阵（简化估计）
        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        # 假设无畸变
        dist_coeffs = np.zeros((4, 1))

        # 求解 PnP 问题
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return False, 0, 0, 0, image_points

        # 转换旋转向量为旋转矩阵
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # 计算欧拉角
        pitch, yaw, roll = self.rotation_matrix_to_euler_angles(rotation_matrix)

        return True, pitch, yaw, roll, image_points

    def rotation_matrix_to_euler_angles(self, R):
        """
        将旋转矩阵转换为欧拉角

        Args:
            R: 3x3 旋转矩阵

        Returns:
            tuple: (pitch, yaw, roll) 以度为单位
        """
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        # 转换为角度
        pitch = math.degrees(x)
        yaw = math.degrees(y)
        roll = math.degrees(z)

        return pitch, yaw, roll

    def put_chinese_text(self, image, text, position, font, color=(255, 255, 255)):
        """
        使用 PIL 在图像上绘制中文文本

        Args:
            image: OpenCV 图像 (BGR)
            text: 要绘制的文本
            position: (x, y) 位置
            font: PIL 字体对象
            color: BGR 颜色元组
        """
        # 将 BGR 转换为 RGB
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # PIL 使用 RGB，需要转换颜色
        color_rgb = (color[2], color[1], color[0])

        # 绘制文本
        draw.text(position, text, font=font, fill=color_rgb)

        # 转回 OpenCV 格式
        image_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # 复制回原图像
        image[:] = image_np

    def draw_info(self, image, pitch, yaw, roll, image_points):
        """
        在图像上绘制头部姿态信息

        Args:
            image: 输入图像
            pitch: 俯仰角
            yaw: 偏航角
            roll: 翻滚角
            image_points: 2D 关键点坐标
        """
        h, w = image.shape[:2]

        # 绘制关键点
        if image_points is not None:
            for point in image_points:
                cv2.circle(image, tuple(point.astype(int)), 3, (0, 255, 0), -1)

        # 创建信息面板背景
        panel_height = 180
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        # 显示角度信息 - 使用 PIL 绘制中文
        # Pitch (上下)
        pitch_text = f"Pitch (上下): {pitch:.1f}°"
        pitch_color = self.get_angle_color(pitch)
        self.put_chinese_text(image, pitch_text, (20, 20), self.font, pitch_color)

        # Yaw (左右)
        yaw_text = f"Yaw (左右): {yaw:.1f}°"
        yaw_color = self.get_angle_color(yaw)
        self.put_chinese_text(image, yaw_text, (20, 60), self.font, yaw_color)

        # Roll (旋转)
        roll_text = f"Roll (旋转): {roll:.1f}°"
        roll_color = self.get_angle_color(roll)
        self.put_chinese_text(image, roll_text, (20, 100), self.font, roll_color)

        # 显示方向
        direction = self.get_direction(pitch, yaw, roll)
        self.put_chinese_text(image, f"方向: {direction}", (20, 140), self.font, (255, 255, 255))

        # 绘制方向箭头指示器
        self.draw_direction_indicator(image, pitch, yaw, roll)

        # 绘制角度条
        self.draw_angle_bars(image, pitch, yaw, roll)

    def get_angle_color(self, angle):
        """根据角度大小返回颜色 (BGR)"""
        abs_angle = abs(angle)
        if abs_angle < 10:
            return (0, 255, 0)  # 绿色 - 正常
        elif abs_angle < 30:
            return (0, 255, 255)  # 黄色 - 轻微转动
        else:
            return (0, 0, 255)  # 红色 - 大幅转动

    def get_direction(self, pitch, yaw, roll):
        """根据角度判断头部方向"""
        directions = []

        # 上下（反转）
        if pitch > 15:
            directions.append("向上")
        elif pitch < -15:
            directions.append("向下")

        # 左右（反转，因为镜像模式）
        if yaw > 15:
            directions.append("向左")
        elif yaw < -15:
            directions.append("向右")

        # 旋转（反转，因为镜像模式）
        if roll > 15:
            directions.append("左倾")
        elif roll < -15:
            directions.append("右倾")

        return " ".join(directions) if directions else "正面"

    def draw_direction_indicator(self, image, pitch, yaw, roll):
        """绘制方向指示器（右上角）"""
        h, w = image.shape[:2]
        center_x, center_y = w - 100, 100
        radius = 60

        # 绘制圆形背景
        cv2.circle(image, (center_x, center_y), radius, (50, 50, 50), -1)
        cv2.circle(image, (center_x, center_y), radius, (255, 255, 255), 2)

        # 绘制中心点
        cv2.circle(image, (center_x, center_y), 5, (255, 255, 255), -1)

        # 计算箭头终点（基于 yaw 和 pitch）
        arrow_length = 40
        end_x = int(center_x + arrow_length * np.sin(np.radians(yaw)))
        end_y = int(center_y + arrow_length * np.sin(np.radians(pitch)))

        # 绘制箭头
        cv2.arrowedLine(image, (center_x, center_y), (end_x, end_y),
                       (0, 255, 255), 3, tipLength=0.3)

    def draw_angle_bars(self, image, pitch, yaw, roll):
        """绘制角度条形图（右侧）"""
        h, w = image.shape[:2]
        bar_x = w - 200
        bar_y_start = 220
        bar_width = 150
        bar_height = 20
        gap = 10

        angles = [
            ("Pitch", pitch, (255, 100, 100)),
            ("Yaw", yaw, (100, 255, 100)),
            ("Roll", roll, (100, 100, 255))
        ]

        for i, (label, angle, color) in enumerate(angles):
            y_pos = bar_y_start + i * (bar_height + gap)

            # 绘制标签
            cv2.putText(image, label, (bar_x, y_pos + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 绘制背景条
            cv2.rectangle(image, (bar_x + 60, y_pos),
                         (bar_x + 60 + bar_width, y_pos + bar_height),
                         (50, 50, 50), -1)

            # 绘制角度条
            # 将角度映射到 -90 到 90 度范围
            normalized_angle = max(-90, min(90, angle))
            bar_length = int((normalized_angle / 90.0) * (bar_width / 2))

            if bar_length > 0:
                cv2.rectangle(image,
                             (bar_x + 60 + bar_width // 2, y_pos),
                             (bar_x + 60 + bar_width // 2 + bar_length, y_pos + bar_height),
                             color, -1)
            else:
                cv2.rectangle(image,
                             (bar_x + 60 + bar_width // 2 + bar_length, y_pos),
                             (bar_x + 60 + bar_width // 2, y_pos + bar_height),
                             color, -1)

            # 绘制中心线
            cv2.line(image,
                    (bar_x + 60 + bar_width // 2, y_pos),
                    (bar_x + 60 + bar_width // 2, y_pos + bar_height),
                    (255, 255, 255), 1)

    def close(self):
        """释放资源"""
        self.face_mesh.close()


def put_text_chinese(image, text, position, font_path, font_size, color=(255, 255, 255)):
    """
    在图像上绘制中文文本的全局辅助函数

    Args:
        image: OpenCV 图像 (BGR)
        text: 要绘制的文本
        position: (x, y) 位置
        font_path: 字体文件路径
        font_size: 字体大小
        color: BGR 颜色元组
    """
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()

    # 将 BGR 转换为 RGB
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # PIL 使用 RGB，需要转换颜色
    color_rgb = (color[2], color[1], color[0])

    # 绘制文本
    draw.text(position, text, font=font, fill=color_rgb)

    # 转回 OpenCV 格式
    image_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # 复制回原图像
    image[:] = image_np


def main():
    """主函数"""
    print("=" * 50)
    print("转头识别系统 - Head Pose Detection System")
    print("=" * 50)
    print("功能说明:")
    print("- 实时检测头部姿态")
    print("- Pitch: 上下点头角度")
    print("- Yaw: 左右转头角度")
    print("- Roll: 左右倾斜角度")
    print("\n操作说明:")
    print("- 按 'q' 键退出程序")
    print("=" * 50)

    # 初始化检测器
    detector = HeadPoseDetector()

    # 打开摄像头
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("错误: 无法打开摄像头!")
        return

    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n摄像头已启动，正在检测...")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("错误: 无法读取摄像头画面!")
            break

        # 水平翻转（镜像效果）
        frame = cv2.flip(frame, 1)

        # 检测头部姿态
        success, pitch, yaw, roll, image_points = detector.get_head_pose(frame)

        if success:
            # 绘制信息
            detector.draw_info(frame, pitch, yaw, roll, image_points)
        else:
            # 未检测到人脸
            put_text_chinese(frame, "未检测到人脸", (50, 50),
                           "/System/Library/Fonts/PingFang.ttc", 40, (0, 0, 255))

        # 显示退出提示
        put_text_chinese(frame, "按 'q' 退出", (frame.shape[1] - 180, frame.shape[0] - 40),
                       "/System/Library/Fonts/PingFang.ttc", 24, (255, 255, 255))

        # 显示画面
        cv2.imshow("转头识别系统 - Head Pose Detection", frame)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    print("\n正在关闭程序...")
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("程序已退出")


if __name__ == "__main__":
    main()
