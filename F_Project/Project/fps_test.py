import cv2
import time
from ultralytics import YOLO

# =========================
# SETTINGS
# =========================

VIDEO_PATH = r"C:\Users\shrey\Onedrive\Desktop\Project\accident_720x410_30fps.mp4"

# Keep the SAME model for every test
MODEL_PATH = "yolo11n.pt"

# Change this to 1, 2, 5 or 10
TARGET_FPS = 5

# =========================
# LOAD MODEL
# =========================

print("Loading YOLO11n...")
model = YOLO(MODEL_PATH)

# =========================
# OPEN VIDEO
# =========================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video")
    exit()

source_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("\nVideo information:")
print("Original FPS:", source_fps)
print("Total frames:", total_frames)
print("Testing FPS:", TARGET_FPS)

# Number of original frames to skip
frame_interval = source_fps / TARGET_FPS

frame_number = 0
processed_frames = 0

total_inference_time = 0

print("\nStarting test...\n")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Process only selected frames
    if frame_number >= round(processed_frames * frame_interval):

        start_time = time.time()

        # YOLO detection
        results = model(frame, verbose=False)

        end_time = time.time()

        inference_time = end_time - start_time
        total_inference_time += inference_time

        processed_frames += 1

        print(
            f"Frame: {frame_number} | "
            f"Processed: {processed_frames} | "
            f"Latency: {inference_time * 1000:.2f} ms"
        )

    frame_number += 1

cap.release()

# =========================
# RESULTS
# =========================

if processed_frames > 0:

    average_latency = (
        total_inference_time / processed_frames
    )

    processing_fps = 1 / average_latency

    print("\n==============================")
    print("       FPS TEST RESULT")
    print("==============================")

    print("Target sampling FPS:", TARGET_FPS)
    print("Original video FPS:", source_fps)
    print("Frames processed:", processed_frames)

    print(
        "Average latency:",
        round(average_latency * 1000, 2),
        "ms/frame"
    )

    print(
        "YOLO processing FPS:",
        round(processing_fps, 2),
        "FPS"
    )

    print("==============================")