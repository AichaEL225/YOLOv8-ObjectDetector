# YOLOv8 Object Detector

Real-time object detection using **YOLOv8** and the **COCO dataset** (80 object classes).
Supports image, webcam, and video detection.

---

## Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/AichaEL225/YOLOv8-ObjectDetector.git
cd YOLOv8-ObjectDetector
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run detection
```bash
# Image
python detector.py

# Webcam
python webcam_detector.py

# Video
python video_detector.py
```

> Model downloads automatically on first run — no manual setup needed!

---

##  Configuration

Edit the config section at the top of each file:

| Setting | Options | Default |
|---|---|---|
| `MODEL_SIZE` | `yolov8n/s/m/l/x.pt` | `yolov8s.pt` |
| `CONFIDENCE` | `0.0` to `1.0` | `0.5` |
| `SHOW_FPS` | `True / False` | `True` |
| `SAVE_RESULT` | `True / False` | `True` |

---

## COCO Classes (80 Objects)

Person, bicycle, car, motorcycle, airplane, bus, train, truck, boat,
traffic light, fire hydrant, stop sign, parking meter, bench, bird,
cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack,
umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball,
kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket,
bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple,
sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair,
couch, potted plant, bed, dining table, toilet, tv, laptop, mouse,
remote, keyboard, cell phone, microwave, oven, toaster, sink,
refrigerator, book, clock, vase, scissors, teddy bear, hair drier,
toothbrush

---

##  Built With
- [YOLOv8 — Ultralytics](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [COCO Dataset](https://cocodataset.org/)