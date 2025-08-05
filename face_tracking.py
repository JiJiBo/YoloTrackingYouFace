# -*- coding: utf-8 -*-
"""
face_tracking.py

基于 YOLOv8 + ByteTrack 人脸检测与多目标追踪，以及静态图库人脸识别示例
依赖：
    pip install ultralytics opencv-python face_recognition pillow

功能：
    1. 加载多张已知人脸图片（文件名即人名），提取人脸特征编码
    2. 对视频/摄像头流进行检测 + 跟踪
    3. 对每个跟踪到的人脸进行识别，匹配已知人脸列表，并根据Track ID缓存姓名
    4. 在画面中标注已识别姓名（支持中文），对于不在已知列表的人脸，用“Unknown”并高亮框出

用法示例：
    python face_tracking.py \
        --known_dir known_faces/ \
        --source 0 \
        --model yolov8n-face.pt \
        --font_path /path/to/simhei.ttf
"""
import argparse
import os
import cv2
import numpy as np
import face_recognition
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 + ByteTrack 人脸检测与识别 (支持中文显示)")
    parser.add_argument("--known_dir", type=str, required=True,
                        help="已知人脸图片目录，文件名为姓名，支持jpg/png等格式")
    parser.add_argument("--source", type=str, required=True,
                        help="输入源：摄像头索引(0,1...)、视频/流地址或图像文件路径")
    parser.add_argument("--model", type=str, required=True,
                        help="YOLOv8 人脸检测模型权重路径")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml",
                        help="追踪器配置文件路径")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="检测置信度阈值")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="NMS IoU 阈值")
    parser.add_argument("--font_path", type=str, default="font/simhei.ttf",
                        help="中文字体文件路径，例如 simhei.ttf")
    return parser.parse_args()

def load_known_faces(known_dir):
    known_encodings, known_names = [], []
    for fname in os.listdir(known_dir):
        path = os.path.join(known_dir, fname)
        name, ext = os.path.splitext(fname)
        if ext.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
        image = face_recognition.load_image_file(path)
        encs = face_recognition.face_encodings(image)
        if encs:
            known_encodings.append(encs[0])
            known_names.append(name)
            print(f"Loaded encoding for {name}")
        else:
            print(f"警告: 无法在 {fname} 中找到人脸，已跳过")
    return known_encodings, known_names

def annotate_frame(frame, boxes, names, font):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    for (box, name) in zip(boxes, names):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0) if name != "Unknown" else (255, 0, 0)
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
        text_pos = (x1, max(0, y1 - 25))
        draw.text(text_pos, name, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def main():
    args = parse_args()
    known_encs, known_names = load_known_faces(args.known_dir)
    # 加载字体
    if args.font_path and os.path.isfile(args.font_path):
        font = ImageFont.truetype(args.font_path, 24)
    else:
        try:
            font = ImageFont.truetype("simhei.ttf", 24)
        except IOError:
            font = ImageFont.load_default()
            print("警告: 未找到中文字体，使用默认字体，可能无法显示中文。可使用 --font_path 指定字体文件。")

    model = YOLO(args.model)

    def process_stream(src):
        track_params = {
            'source': src,
            'tracker': args.tracker,
            'persist': True,
            'conf': args.conf,
            'iou': args.iou,
            'stream': True,
        }
        # 保存Track ID对应的姓名
        id2name = {}
        cv2.namedWindow('Face Recognition', cv2.WINDOW_NORMAL)
        for frame_res in model.track(**track_params):
            frame = frame_res.orig_img.copy()
            # 获取boxes和对应track IDs
            boxes = frame_res.boxes.xyxy.cpu().numpy()
            try:
                ids = frame_res.boxes.id.cpu().numpy().astype(int)
            except AttributeError:
                ids = list(range(len(boxes)))  # 若无ID，则按顺序
            names = []
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for box, tid in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                locs = [(y1, x2, y2, x1)]
                encs = face_recognition.face_encodings(rgb, locs, num_jitters=1)
                if encs:
                    matches = face_recognition.compare_faces(known_encs, encs[0], tolerance=0.5)
                    if True in matches:
                        name = known_names[matches.index(True)]
                        id2name[tid] = name  # 更新缓存
                    else:
                        name = id2name.get(tid, "Unknown")
                else:
                    name = id2name.get(tid, "Unknown")
                names.append(name)
            out = annotate_frame(frame, boxes, names, font)
            cv2.imshow('Face Recognition', out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    if args.source.isdigit() or not os.path.isfile(args.source):
        src = int(args.source) if args.source.isdigit() else args.source
        process_stream(src)
    else:
        img = cv2.imread(args.source)
        if img is None:
            print(f"无法读取图像: {args.source}")
            return
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model.predict(source=rgb, conf=args.conf, iou=args.iou)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        names = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            locs = [(y1, x2, y2, x1)]
            encs = face_recognition.face_encodings(rgb, locs, num_jitters=1)
            name = "Unknown"
            if encs:
                matches = face_recognition.compare_faces(known_encs, encs[0], tolerance=0.5)
                if True in matches:
                    name = known_names[matches.index(True)]
            names.append(name)
        out = annotate_frame(img, boxes, names, font)
        cv2.imshow('Image Recognition', out)
        print("按任意键关闭窗口")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
