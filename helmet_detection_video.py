import cv2
import cvzone
from ultralytics import YOLO
import math

# -------------------- SETTINGS --------------------
VIDEO_PATH = "media/bike_3.mp4"
MODEL_PATH = "weights/best.pt"

CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.3

# -------------------- LOAD --------------------
cap = cv2.VideoCapture(VIDEO_PATH)
model = YOLO(MODEL_PATH)
classNames = ['With Helmet', 'Without Helmet']

# 🔥 Frame skipping counter
frame_count = 0

# -------------------- MAIN LOOP --------------------
while True:
    success, img = cap.read()
    if not success:
        break

    # 🔥 Skip frames (2x speed boost)
    frame_count += 1
    if frame_count % 2 != 0:
        continue

    # 🔥 Resize frame (major speed boost)
    img = cv2.resize(img, (640, 480))

    # 🔥 Faster prediction (no stream, no logs)
    results = model.predict(
        img,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        agnostic_nms=True,
        verbose=False
    )

    for r in results:
        boxes = r.boxes
        detections = []

        # Step 1: Collect detections
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            w, h = x2 - x1, y2 - y1

            # 🔥 Filter small (far) objects
            if w < 60 or h < 60:
                continue

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            detections.append({
                'box': [x1, y1, x2, y2],
                'conf': conf,
                'cls': cls,
                'center': (cx, cy)
            })

        # Step 2: Remove overlapping detections
        final_detections = []
        detections.sort(key=lambda x: x['conf'], reverse=True)

        for d in detections:
            keep = True
            for f in final_detections:
                dist = math.hypot(
                    d['center'][0] - f['center'][0],
                    d['center'][1] - f['center'][1]
                )
                if dist < 50:
                    keep = False
                    break
            if keep:
                final_detections.append(d)

        # Step 3: Draw results
        for d in final_detections:
            x1, y1, x2, y2 = d['box']
            cls = d['cls']
            conf = d['conf']

            color = (0, 255, 0) if cls == 0 else (0, 0, 255)

            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), colorR=color, t=3, rt=1)

            label = f"{classNames[cls]} {int(conf*100)}%"
            cvzone.putTextRect(
                img,
                label,
                (max(0, x1), max(35, y1)),
                scale=1,
                thickness=2,
                colorR=color,
                offset=5
            )

    # -------------------- DISPLAY --------------------
    cv2.imshow("Helmet Detection (Fast)", img)

    # 🔥 Fast playback
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()