```shell
pip install ultralytics opencv-python
```

```shell
    python face_tracking.py \
        --source 0 \
        --model yolov8n-face.pt \
        --tracker bytetrack.yaml \
        --conf 0.3 \
        --iou 0.5 \
        --output output.mp4
```