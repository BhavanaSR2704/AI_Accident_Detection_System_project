import cv2
import time
from ultralytics import YOLO

# =========================
# SETTINGS
# =========================

VIDEO_PATH = r"C:\Users\shrey\Onedrive\Desktop\Project\accident_720x410_30fps.mp4"
MODEL_PATH = "yolo11n.pt"

# Keep selected FPS = 5
TARGET_FPS = 5

# CHANGE THIS FOR EACH TEST
# Options:
# None       = Original resolution
# (960, 540) = 960x540
# (640, 360) = 640x360

TARGET_RESOLUTION = (720,410)

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
source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("\n==============================")
print("VIDEO INFORMATION")
print("==============================")
print("Original FPS:", source_fps)
print("Original Resolution:", source_width, "x", source_height)
print("Sampling FPS:", TARGET_FPS)

if TARGET_RESOLUTION is None:
    print("Testing Resolution: ORIGINAL")
else:
    print(
        "Testing Resolution:",
        TARGET_RESOLUTION[0],
        "x",
        TARGET_RESOLUTION[1]
    )

# =========================
# FPS SAMPLING
# =========================

frame_interval = source_fps / TARGET_FPS

frame_number = 0
next_sample_frame = 0
processed_frames = 0

total_inference_time = 0

print("\nStarting test...\n")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Process only selected FPS frames
    if frame_number >= round(next_sample_frame):

        # =========================
        # RESIZE FRAME
        # =========================

        if TARGET_RESOLUTION is not None:
            frame = cv2.resize(
                frame,
                TARGET_RESOLUTION
            )

        # =========================
        # YOLO INFERENCE
        # =========================

        start_time = time.time()

        results = model(
            frame,
            verbose=False
        )

        end_time = time.time()

        inference_time = end_time - start_time

        total_inference_time += inference_time
        processed_frames += 1

        print(
            f"Frame: {frame_number} | "
            f"Processed: {processed_frames} | "
            f"Latency: {inference_time * 1000:.2f} ms"
        )

        next_sample_frame += frame_interval

    frame_number += 1

cap.release()

# =========================
# FINAL RESULTS
# =========================

if processed_frames > 0:

    average_latency = (
        total_inference_time /
        processed_frames
    )

    processing_fps = 1 / average_latency

    print("\n==============================")
    print("       RESOLUTION RESULT")
    print("==============================")

    if TARGET_RESOLUTION is None:
        print("Resolution: ORIGINAL")
    else:
        print(
            "Resolution:",
            TARGET_RESOLUTION[0],
            "x",
            TARGET_RESOLUTION[1]
        )

    print("Sampling FPS:", TARGET_FPS)
    print("Frames processed:", processed_frames)

    print(
        "Average latency:",
        round(
            average_latency * 1000,
            2
        ),
        "ms/frame"
    )

    print(
        "YOLO processing FPS:",
        round(
            processing_fps,
            2
        ),
        "FPS"
    )

    print("==============================")