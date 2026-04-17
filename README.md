# Helmet Detection System
This project detects whether bike riders are wearing helmets using a trained YOLO model. It works on both images and videos.

## Files in this project
* `helmet_detection_image.py` → runs detection on a single image
* `helmet_detection_video.py` → optimized version for video
* `helmet_video.py` → alternative video detection script
* `media/` → contains sample images and videos
* `weights/best.pt` → trained YOLO model

## Requirements
Install the required libraries using:
```
pip install -r requirements.txt
```

## How to run
### For image detection
```
python helmet_detection_image.py
```
### For video detection (fast version)
```
python helmet_detection_video.py
```
### Alternative video script
```
python helmet_video.py
```

## Notes
* File paths are hardcoded inside the scripts. Change them if needed.
* The model file (`best.pt`) must be present inside the `weights` folder.

## Author
Banoth Ashok - 230285
Bandi Aditya - 230283
Ramavath Raju - 230842
