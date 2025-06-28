# Traffic-Analysis-Detection-Tracking-Counting-
Traffic Analysis(Detection + Tracking + Counting)
# 🚗 Vehicle Detection & Tracking Using YOLOv8 + SimpleTracker

This project implements real-time vehicle detection, tracking, and counting using [YOLOv8] (https://github.com/ultralytics/ultralytics) and a custom IoU-based tracker. It focuses on cars, motorcycles, trucks, and buses and uses real-time video processing using OpenCV.

---

##  Features

-  **Object Detection** using YOLOv8 (`yolov8n.pt`) for fast and accurate detection.
-  **Class-Specific Filtering** (Cars, Motorcycles, Buses, Trucks).
-  **SimpleTracker** for object tracking using Intersection-over-Union (IoU) instead of a deep learning-based tracker.
-  **Object Identity Preservation** across frames.
-  **Vehicle Counting** based on objects crossing a virtual horizontal line.
-  **Live Visualization** with bounding boxes, labels, and movement trails.
-  Optimized for performance and can run on CPU or GPU.

---

##  Use Cases

- Smart traffic analysis
- Traffic congestion monitoring
- Vehicle flow visualization
- Embedded vision projects

---

##  Tech Stack

| Component     | Description                                        |
|---------------|----------------------------------------------------|
| Python        | Main language                                      |
| OpenCV        | Frame manipulation, visualization, and video I/O  |
| Ultralytics YOLOv8 | Pretrained vehicle detection model             |
| NumPy         | Basic mathematical operations                      |

---

## 🚦 How It Works

### 1. **Detection**
- Uses `yolov8n.pt` model to detect objects in the frame.
- Only detects vehicle-related classes from the COCO dataset (IDs: 2, 3, 5, 7).

### 2. **Tracking**
- A lightweight IoU-based tracker assigns a persistent ID to each detected vehicle.
- Tracks vehicles across multiple frames using bounding box overlap.

### 3. **Counting**
- A horizontal red line is drawn at 70% of the frame height.
- When a vehicle crosses the line **from below**, it's counted.
- Avoids recounting the same ID using a set of `counted_ids`.

### 4. **Visualization**
- Bounding boxes are color-coded per class.
- Object IDs and class names are shown above each box.
- Movement trails for visual understanding of motion.




