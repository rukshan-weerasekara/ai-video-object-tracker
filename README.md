# 🎬 AI-Powered Video Object Tracker (YOLOv8m)

A high-performance Computer Vision application designed for **Animators, VFX Artists, and Video Editors** to automate object identification and motion tracking.

## 🔗 Live Demo
Experience the tool here: [https://ai-video-object-tracker-rfy6qbvptmatqvtp5crzjf.streamlit.app/](https://ai-video-object-tracker-rfy6qbvptmatqvtp5crzjf.streamlit.app/)

## 🚀 The Creative Use Case
As a Video Editor/Animator, manual rotoscoping and object tracking take hours. This tool leverages **YOLOv8 (You Only Look Once)** to:
* **Auto-Detect:** Instantly identify over 80+ object classes (People, Vehicles, Gear, etc.).
* **Persistent Tracking:** Assign unique IDs to objects to maintain continuity across frames.
* **Workflow Boost:** Exporting these tracking paths can significantly speed up masking in After Effects or Blender.



## 🧠 Technical Architecture
The system uses the **YOLOv8 Medium (yolov8m.pt)** model, which offers a superior balance between inference speed and detection precision ($mAP$).

### Key Features:
* **Frame-by-Frame Inference:** Real-time processing of uploaded video assets.
* **Confidence Thresholding:** Adjustable slider to filter out noise and low-confidence detections.
* **Dynamic Visualization:** Overlays bounding boxes and tracking IDs directly on the UI.



## 🛠️ Tech Stack
* **Core AI:** YOLOv8 by Ultralytics
* **UI Framework:** Streamlit
* **Image Processing:** OpenCV, Pillow
* **Language:** Python 3.12

## 👨‍💻 Author
**Rukshan Weerasekara**
*Creative Technologist | Animator | AI Developer*
