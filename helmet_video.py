import cv2
import cvzone
from ultralytics import YOLO
import math
import os

# -------------------- SETTINGS --------------------
VIDEO_PATH = "media/bike_3.mp4" # Switched to your new sample
MODEL_PATH = "weights/best.pt"

CONF_THRESHOLD = 0.5  # Kept moderate to ensure we catch actual riders
IOU_THRESHOLD = 0.3   # Aggressive overlap suppression

# -------------------- LOAD --------------------
cap = cv2.VideoCapture(VIDEO_PATH)
model = YOLO(MODEL_PATH)
classNames = ['With Helmet', 'Without Helmet']

# -------------------- MAIN LOOP --------------------
while True:
    success, img = cap.read()
    if not success:
        break

    # Using agnostic_nms=True to force a choice between classes
    results = model.predict(img, stream=True, conf=CONF_THRESHOLD, 
                            iou=IOU_THRESHOLD, agnostic_nms=True)

    for r in results:
        boxes = r.boxes
        detections = []

        # Step 1: Clean up and store detections
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            detections.append({'box': [x1, y1, x2, y2], 'conf': conf, 'cls': cls, 'center': (cx, cy)})

        # Step 2: Manual Proximity Filter (Prevents Parallel Boxes)
        # If two centers are within 30 pixels, we only keep the one with higher confidence
        final_detections = []
        detections.sort(key=lambda x: x['conf'], reverse=True) # Strongest first

        for d in detections:
            keep = True
            for f in final_detections:
                dist = math.hypot(d['center'][0] - f['center'][0], d['center'][1] - f['center'][1])
                if dist < 40: # If centers are closer than 40 pixels, it's the same head
                    keep = False
                    break
            if keep:
                final_detections.append(d)

        # Step 3: Draw the final cleaned-up results
        for d in final_detections:
            x1, y1, x2, y2 = d['box']
            cls = d['cls']
            conf = d['conf']
            
            # Color Coding: Green for Safe, Red for Violation
            color = (0, 255, 0) if cls == 0 else (0, 0, 255)

            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), colorR=color, t=3, rt=1)
            
            label = f"{classNames[cls]} {int(conf*100)}%"
            cvzone.putTextRect(img, label, (max(0, x1), max(35, y1)), 
                               scale=1.2, thickness=2, colorR=color, offset=10)

    # -------------------- DISPLAY --------------------
    cv2.imshow("Bike Helmet Detection System", img)

    if cv2.waitKey(3000) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()