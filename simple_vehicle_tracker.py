import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# ==================== INITIALIZATION ====================
# Load the YOLOv8 object detection model (nano version for speed)
# This model detects various object types; we will filter to vehicles only.
model = YOLO('yolov8n.pt')

# Try opening multiple possible video file paths (team may use different local files)
video_paths = [
    # "highway.mp4",
    "vidvid.mp4"
]

cap = None
for video_path in video_paths:
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        print(f"Successfully opened video: {video_path}")
        break

if not cap or not cap.isOpened():
    print("Error: Could not open any video file. Please ensure one of these files exists:")
    for path in video_paths:
        print(f"  - {path}")
    exit(1)

# ==================== VEHICLE DETECTION CONFIGURATION ====================
# Define the COCO dataset class IDs corresponding to vehicles
# These IDs are used to filter detection outputs from YOLO
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# Use different confidence thresholds per vehicle class
# This allows us to fine-tune detection reliability based on object size/speed
CLASS_CONFIDENCE_THRESHOLDS = {
    2: 0.5,  # Cars - higher threshold to reduce truck confusion
    3: 0.6,  # Motorcycles - higher threshold to reduce false positives
    5: 0.5,  # Buses - stricter threshold due to larger size
    7: 0.5   # Trucks - higher threshold to reduce car confusion
}

# Size-based classification to resolve car-truck conflicts
def classify_by_size(bbox, class_id, confidence):
    """Use bounding box size to help classify between cars and trucks"""
    x, y, w, h = bbox
    area = w * h
    
    # If confidence is low, use size to help classify
    if confidence < 0.7:
        # Large vehicles are more likely to be trucks
        if area > 15000:  # Large area threshold
            if class_id == 2:  # If classified as car but large
                return 7  # Reclassify as truck
        # Small vehicles are more likely to be cars
        elif area < 8000:  # Small area threshold
            if class_id == 7:  # If classified as truck but small
                return 2  # Reclassify as car
    
    return class_id

# ==================== SIMPLE TRACKER ====================
class SimpleTracker:
    """
    Lightweight object tracker using Intersection-over-Union (IoU) for tracking.
    Maintains identities of detected objects across frames based on box overlap.
    """
    def __init__(self):
        self.next_id = 1  # Unique ID for new tracks
        self.tracks = {}  # Active tracks
        self.max_age = 30  # Frames to retain a lost track before deleting
        self.iou_threshold = 0.3  # IOU minimum to consider a match

    def calculate_iou(self, bbox1, bbox2):
        """
        Compute IOU between two bounding boxes to assess overlap quality.
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def update(self, detections):
        """
        Match current detections with existing tracks or create new ones.
        Returns list of matched boxes with associated IDs and class labels.
        """
        if not detections:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['lost'] += 1
                if self.tracks[track_id]['lost'] > self.max_age:
                    del self.tracks[track_id]
            return []

        matched_tracks = set()
        matched_detections = set()
        boxes_ids = []

        for track_id in list(self.tracks.keys()):
            track = self.tracks[track_id]
            best_iou = 0
            best_detection_idx = -1

            for i, detection in enumerate(detections):
                if i in matched_detections:
                    continue

                iou = self.calculate_iou(track['bbox'], detection[:4])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_detection_idx = i

            if best_detection_idx != -1:
                # Update track but maintain class consistency
                new_class = detections[best_detection_idx][5]
                old_class = track['class']
                
                # If class changed, use size-based classification to resolve conflict
                if new_class != old_class and (new_class in [2, 7] and old_class in [2, 7]):
                    # Car-truck conflict - use size to decide
                    bbox = detections[best_detection_idx][:4]
                    confidence = detections[best_detection_idx][4]
                    resolved_class = classify_by_size(bbox, new_class, confidence)
                    
                    # Only change class if we're confident about the new classification
                    if resolved_class == new_class and confidence > 0.6:
                        final_class = new_class
                    else:
                        final_class = old_class  # Keep original class
                else:
                    final_class = new_class
                
                self.tracks[track_id].update({
                    'bbox': detections[best_detection_idx][:4],
                    'age': track['age'] + 1,
                    'lost': 0,
                    'class': final_class
                })
                matched_tracks.add(track_id)
                matched_detections.add(best_detection_idx)
                boxes_ids.append((self.tracks[track_id]['bbox'], track_id, self.tracks[track_id]['class']))
            else:
                self.tracks[track_id]['lost'] += 1
                if self.tracks[track_id]['lost'] > self.max_age:
                    del self.tracks[track_id]
                else:
                    boxes_ids.append((track['bbox'], track_id, track['class']))

        for i, detection in enumerate(detections):
            if i not in matched_detections:
                # Apply size-based classification for new tracks
                bbox = detection[:4]
                confidence = detection[4]
                original_class = detection[5]
                resolved_class = classify_by_size(bbox, original_class, confidence)
                
                self.tracks[self.next_id] = {
                    'bbox': detection[:4],
                    'age': 1,
                    'lost': 0,
                    'class': resolved_class
                }
                boxes_ids.append((detection[:4], self.next_id, resolved_class))
                self.next_id += 1

        return boxes_ids

# ==================== COUNTER SYSTEM ====================
class ObjectCounter:
    """
    Tracks objects crossing a defined horizontal line.
    Counts total vehicles and class-specific totals (cars, motorcycles).
    """
    def __init__(self):
        self.count = 0
        self.car_count = 0
        self.motorcycle_count = 0
        self.track_history = defaultdict(list)
        self.counted_ids = set()

    def update(self, tracks, counting_line_y):
        current_ids = set()
        for box, track_id, class_id in tracks:
            x, y, w, h = box
            center = (x + w // 2, y + h // 2)

            self.track_history[track_id].append(center)
            if len(self.track_history[track_id]) > 50:
                self.track_history[track_id].pop(0)

            if center[1] < counting_line_y and track_id not in self.counted_ids:
                self.count += 1
                self.counted_ids.add(track_id)

                if class_id == 2:
                    self.car_count += 1
                elif class_id == 3:
                    self.motorcycle_count += 1

            current_ids.add(track_id)

        for track_id in list(self.track_history.keys()):
            if track_id not in current_ids:
                del self.track_history[track_id]

# ==================== MAIN PROCESSING LOOP ====================
tracker = SimpleTracker()
counter = ObjectCounter()

print("Starting enhanced vehicle detection and tracking...")
print("Tracks cars and motorcycles using class-specific thresholds.")
print("Visual feedback includes bounding boxes, trails, and per-class counters.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video reached")
        break

    h, w = frame.shape[:2]
    counting_line_y = int(h * 0.7)

    margin = 0.2
    roi = frame[int(h * margin):h - int(h * margin), int(w * margin):w - int(w * margin)]

    results = model(roi, classes=[2, 3, 5, 7], verbose=False, conf=0.4, iou=0.5)
    detections = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf.item()
        class_id = int(box.cls.item())

        if confidence > CLASS_CONFIDENCE_THRESHOLDS.get(class_id, 0.4):
            detections.append([x1, y1, x2 - x1, y2 - y1, confidence, class_id])

    boxes_ids = tracker.update(detections)
    counter.update(boxes_ids, counting_line_y)

    cv2.line(frame, (0, counting_line_y), (w, counting_line_y), (0, 0, 255), 3)

    for (x, y, w, h), track_id, class_id in boxes_ids:
        if class_id == 2:
            color, label = (0, 255, 0), f"Car:{track_id}"
        elif class_id == 3:
            color, label = (255, 0, 0), f"Moto:{track_id}"
        elif class_id == 5:
            color, label = (0, 255, 255), f"Bus:{track_id}"
        else:
            color, label = (255, 0, 255), f"Truck:{track_id}"

        cv2.rectangle(roi, (x, y), (x + w, y + h), color, 2)
        cv2.putText(roi, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        history = counter.track_history.get(track_id, [])
        for i in range(1, len(history)):
            cv2.line(roi, history[i - 1], history[i], (0, 255, 255), 2)

    cv2.putText(frame, f"Total Vehicles: {counter.count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Cars: {counter.car_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Motorcycles: {counter.motorcycle_count}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f"Active Tracks: {len(tracker.tracks)}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Enhanced Vehicle Tracker - Cars & Motorcycles", frame)

    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Final Results:")
print(f"Total Vehicles: {counter.count}")
print(f"Cars: {counter.car_count}")
print(f"Motorcycles: {counter.motorcycle_count}")
