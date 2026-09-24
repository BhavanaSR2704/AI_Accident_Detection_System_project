# AI_Accident_Detection_System_project

# 🚨 AI Accident Detection System

An AI-based accident detection system that analyzes CCTV/dashcam traffic videos and identifies possible accidents and abnormal vehicle behavior using computer vision, object detection, multi-object tracking, and temporal analysis.

---

## 📌 Project Overview

 accidents can happen within a few seconds, making continuous manual monitoring difficult.

This project aims to develop an automated system that analyzes traffic videos and detects accident-related events such as:

- 🚗 Vehicle collisions
- 🛑 Sudden vehicle stops
- 🔄 Sudden direction changes
- ⚠️ Abnormal vehicle movement
- 🚧 Vehicle interaction with roadside objects

The system processes the uploaded video frame-by-frame and uses vehicle tracking and temporal reasoning to determine whether an accident should be confirmed.

---

## 🎯 Objectives

The main objectives of this project are:

1. Detect vehicles in traffic videos.
2. Track vehicles across consecutive frames.
3. Analyze vehicle movement over time.
4. Identify collision and abnormal motion patterns.
5. Use temporal reasoning to reduce false accident detections.
6. Display the detected accident and its timestamp.
7. Provide a simple web interface for video analysis.
8. Generate an analysis summary for the processed video.

---

## 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| YOLO11n | Vehicle and object detection |
| ByteTrack | Multi-object tracking |
| OpenCV | Video processing and frame handling |
| NumPy | Numerical calculations |
| Pandas | Analysis results and tables |
| Streamlit | Web interface |
| CUDA / NVIDIA GPU | Accelerated model processing |

---

## 🔍 Detection Pipeline

The system follows the following pipeline:

```text
CCTV / Dashcam Video
            ↓
      Frame Extraction
            ↓
      Frame Preprocessing
            ↓
       YOLO11n Detection
            ↓
       ByteTrack Tracking
            ↓
    Vehicle Movement History
            ↓
      Event Detection
            ↓
 ┌───────────────────────────┐
 │ Vehicle Collision         │
 │ Sudden Vehicle Stop       │
 │ Direction Change           │
 │ Abnormal Motion            │
 │ Roadside Interaction      │
 └───────────────────────────┘
            ↓
    Temporal Accident Analysis
            ↓
      Accident Confirmation
            ↓
   Timestamp + Reason + Frame
            ↓
       Final Analysis
