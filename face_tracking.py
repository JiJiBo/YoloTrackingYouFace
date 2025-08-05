# -*- coding: utf-8 -*-
"""
face_tracking.py

基于 YOLOv8 和 ByteTrack 的人脸检测与多目标追踪完整示例
依赖：
    pip install ultralytics opencv-python

用法示例：
    python face_tracking.py \
        --source 0 \
        --model yolov8n-face.pt \
        --tracker bytetrack.yaml \
        --conf 0.3 \
        --iou 0.5 \
        --output output.mp4
"""
import argparse
import os
import cv2
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Face Tracking Demo")
    parser.add_argument("--source", type=str, default="0",
                        help="输入源：0（摄像头）或视频文件路径")
    parser.add_argument("--model", type=str, required=True,
                        help="YOLOv8 人脸检测模型权重路径（e.g. yolov8n-face.pt）")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml",
                        help="追踪器配置文件（botsort.yaml 或 bytetrack.yaml）")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="检测置信度阈值")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="NMS IoU 阈值")
    parser.add_argument("--output", type=str, default=None,
                        help="输出视频文件路径，若不指定则不保存")
    parser.add_argument("--show", action="store_true", default=False,
                        help="实时显示窗口")
    return parser.parse_args()


def main():
    args = parse_args()
    # 解析输入源
    src = 0 if args.source.isdigit() else args.source

    # 加载模型
    model = YOLO(args.model)
    # 配置追踪
    track_params = {
        'source': src,
        'tracker': args.tracker,
        'persist': True,
        'conf': args.conf,
        'iou': args.iou
    }

    # 如果需要输出视频
    writer = None
    if args.output:
        # 打开输入流以获取视频参数
        cap = cv2.VideoCapture(src)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # 执行检测 + 跟踪
    results = model.track(**track_params)

    for frame_res in results:
        # 获取带标注的帧
        annotated = frame_res.plot()
        # 将 PIL->numpy
        frame = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

        # 写入输出
        if writer:
            writer.write(frame)
        # 实时显示
        if args.show:
            cv2.imshow('Face Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # 清理资源
    if writer:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
