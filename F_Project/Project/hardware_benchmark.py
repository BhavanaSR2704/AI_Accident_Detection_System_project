import time
import cv2
import psutil
from ultralytics import YOLO

# ---------------------------------------------------------
# Hardware benchmark for Stage-1 accident detection
# Model: YOLO11n
# Video: accident_test.mp4
# ---------------------------------------------------------

VIDEO_PATH = "accident_720x410_30fps.mp4"
MODEL_PATH = "yolo11n.pt"

print("=" * 60)
print("STAGE-1 HARDWARE BENCHMARK - YOLO11n")
print("=" * 60)

# Load video information
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError(
        f"Could not open video: {VIDEO_PATH}\n"
        "Make sure accident_test.mp4 is in the same folder as this script."
    )

video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"\nVideo resolution : {video_width} x {video_height}")
print(f"Original video FPS: {video_fps:.2f}")
print(f"Total frames      : {total_frames}")

cap.release()

# Load YOLO model
print("\nLoading YOLO11n...")
model = YOLO(MODEL_PATH)

# CPU/RAM baseline
process = psutil.Process()
process.cpu_percent(None)

print("Starting inference benchmark...\n")

frame_count = 0
inference_times = []

cap = cv2.VideoCapture(VIDEO_PATH)

start_total = time.perf_counter()

while True:
    success, frame = cap.read()

    if not success:
        break

    start_frame = time.perf_counter()

    # Run YOLO inference.
    # show=False keeps the benchmark focused on inference speed.
    model.predict(
        source=frame,
        verbose=False,
        conf=0.25
    )

    end_frame = time.perf_counter()

    inference_times.append(end_frame - start_frame)
    frame_count += 1

end_total = time.perf_counter()

cap.release()

# ---------------------------------------------------------
# Calculate results
# ---------------------------------------------------------
total_time = end_total - start_total

if frame_count == 0:
    raise RuntimeError("No video frames were processed.")

avg_latency = sum(inference_times) / len(inference_times)
inference_fps = 1 / avg_latency
overall_fps = frame_count / total_time

# RAM measurement
memory_mb = process.memory_info().rss / (1024 * 1024)

print("=" * 60)
print("BENCHMARK RESULTS")
print("=" * 60)

print(f"Frames processed       : {frame_count}")
print(f"Total benchmark time   : {total_time:.2f} seconds")
print(f"Average inference time : {avg_latency * 1000:.2f} ms/frame")
print(f"Inference FPS          : {inference_fps:.2f} FPS")
print(f"Overall processing FPS : {overall_fps:.2f} FPS")
print(f"Process RAM usage      : {memory_mb:.2f} MB")

print("\nHardware:")
print("CPU                    : AMD Ryzen 5 7520U")
print("RAM                    : 16 GB")
print("GPU                    : AMD Radeon(TM) Graphics")
print("Dedicated GPU memory   : 487 MB")
print("Shared GPU memory      : 7.6 GB")

print("\nStage-1 interpretation:")
if inference_fps >= video_fps:
    print("The measured inference speed is at least the source video FPS.")
else:
    print("The measured inference speed is below the source video FPS.")
    print("Lower input sampling (e.g., 1/2/5 FPS) can be evaluated for feasibility.")

print("=" * 60)
