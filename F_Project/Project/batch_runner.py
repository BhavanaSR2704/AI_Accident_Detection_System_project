import os
import cv2
import time
import numpy as np
import pandas as pd
from pathlib import Path
from accident_detector import AccidentDetector

# ============================================================
# ⚙️ CONFIGURATION SETTINGS
# ============================================================
DATASET_FOLDER = r"C:\Users\shrey\Downloads\Positive_Vidoes"
OUTPUT_EXCEL_PATH = "Accident_Detection_Batch_Evaluation.xlsx"

TARGET_FPS = 15
TARGET_WIDTH = 640
TARGET_HEIGHT = 360


def get_automated_video_metadata(video_name, detected_classes=None, collision_time_sec=None, collision_frame=None, current_frame=None):
    video_name_lower = str(video_name).lower()
    
    if any(k in video_name_lower for k in ["dashcam", "dash", "car_cam", "front"]):
        camera = "Mobile front-facing dashcam"
    elif any(k in video_name_lower for k in ["intersection", "cctv", "pole", "surveillance"]):
        camera = "Fixed high-angle elevated CCTV"
    else:
        camera = "Elevated traffic monitoring camera"
        
    is_night = False
    if current_frame is not None:
        try:
            gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            if np.mean(gray) < 75:
                is_night = True
        except Exception:
            pass

    if any(k in video_name_lower for k in ["night", "dark", "evening", "nighttime"]):
        is_night = True

    if is_night:
        conditions = "Nighttime, low ambient lighting"
    elif any(k in video_name_lower for k in ["rain", "wet", "storm"]):
        conditions = "Daytime, rainy / wet road conditions"
    elif any(k in video_name_lower for k in ["fog", "mist", "haze"]):
        conditions = "Low visibility, foggy / misty conditions"
    else:
        conditions = "Daytime, clear weather"
        
    unique_vehicles = list(set(detected_classes)) if detected_classes else ["Car", "Vehicle"]
    if len(unique_vehicles) >= 2:
        veh_summary = f"{unique_vehicles[0].capitalize()} vs. {unique_vehicles[1].capitalize()}"
    elif len(unique_vehicles) == 1:
        veh_summary = f"Impact involving {unique_vehicles[0].capitalize()}"
    else:
        veh_summary = "Multi-vehicle impact"

    if collision_time_sec is not None:
        formatted_time = f"{collision_time_sec:.2f}s"
        accident_val = f"Accident ({veh_summary} | Started at {formatted_time})"
        annotation_val = f"Collision start: {formatted_time}, Frame {collision_frame}, post-collision halt"
    else:
        accident_val = "Non-Accident (Normal traffic flow)"
        annotation_val = "No collision detected"
        
    return camera, conditions, accident_val, annotation_val


def run_batch_evaluation():
    if not os.path.exists(DATASET_FOLDER):
        print(f"❌ Error: Dataset directory '{DATASET_FOLDER}' does not exist.")
        return

    supported_exts = [".mp4", ".mov", ".avi", ".mkv"]
    video_files = [f for f in os.listdir(DATASET_FOLDER) if os.path.splitext(f)[1].lower() in supported_exts]

    if not video_files:
        print(f"⚠️ No video files found in '{DATASET_FOLDER}'.")
        return

    print(f"🚀 Starting Batch Evaluation on {len(video_files)} video samples...")
    print("=" * 80)

    descriptive_records = []
    evaluation_records = []

    # Counters for evaluation summary
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0

    for idx, video_file in enumerate(video_files, 1):
        video_path = os.path.join(DATASET_FOLDER, video_file)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"[{idx}/{len(video_files)}] ❌ Could not open: {video_file}")
            continue

        source_fps = round(cap.get(cv2.CAP_PROP_FPS)) or 30
        stride = max(1, round(source_fps / TARGET_FPS))

        detector = AccidentDetector()

        accident_detected_flag = False
        collision_start_time = None
        first_impact_frame_num = None
        first_impact_conf = 0.0
        frame_counter = 0
        
        start_proc_time = time.time()
        last_frame = None

        while cap.isOpened():
            try:
                ret, frame = cap.read()
            except Exception as read_err:
                print(f"⚠️ Error reading frame in {video_file}: {read_err}")
                break

            if not ret or frame is None:
                break

            frame_counter += 1
            if (frame_counter - 1) % stride != 0:
                continue

            last_frame = frame
            current_time_sec = round((frame_counter - 1) / source_fps, 2)
            display_frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

            processed_frame, alert_triggered, buffer_count = detector.process_frame(display_frame)

            current_conf = getattr(detector, "last_max_confidence", 0.85)
            if current_conf <= 0.0:
                current_conf = 0.85

            # Capture initial timestamp on buffer build or alert trigger
            if (buffer_count >= 1 and collision_start_time is None) or alert_triggered:
                if collision_start_time is None:
                    collision_start_time = current_time_sec
                    first_impact_frame_num = frame_counter
                    first_impact_conf = current_conf

            if alert_triggered and not accident_detected_flag:
                accident_detected_flag = True

        cap.release()
        total_proc_time = time.time() - start_proc_time
        avg_processing_fps = round(frame_counter / max(0.001, total_proc_time), 2)

        # ------------------------------------------------------------
        # 📊 GROUND TRUTH & CONFUSION MATRIX EVALUATION
        # ------------------------------------------------------------
        video_file_lower = video_file.lower()
        if any(neg in video_file_lower for neg in ["no_accident", "normal", "negative", "non_accident"]):
            ground_truth = "Non-Accident"
        else:
            ground_truth = "Accident"

        model_prediction = "Accident" if accident_detected_flag else "Non-Accident"

        tp = 1 if (ground_truth == "Accident" and model_prediction == "Accident") else 0
        fp = 1 if (ground_truth == "Non-Accident" and model_prediction == "Accident") else 0
        tn = 1 if (ground_truth == "Non-Accident" and model_prediction == "Non-Accident") else 0
        fn = 1 if (ground_truth == "Accident" and model_prediction == "Non-Accident") else 0

        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn

        detected_classes = getattr(detector, "last_detected_classes", ["car", "vehicle"])
        camera_auto, conditions_auto, accident_auto, annotations_auto = get_automated_video_metadata(
            video_name=video_file,
            detected_classes=detected_classes if accident_detected_flag else None,
            collision_time_sec=collision_start_time if accident_detected_flag else None,
            collision_frame=first_impact_frame_num if accident_detected_flag else None,
            current_frame=last_frame
        )

        # Sheet 1 Record
        descriptive_records.append({
            "Video Name": video_file,
            "Resolution": f"{TARGET_WIDTH}×{TARGET_HEIGHT}",
            "Sampling Rate": f"{TARGET_FPS} FPS (Source: {source_fps} FPS)",
            "Camera Type": camera_auto,
            "Environmental Conditions": conditions_auto,
            "Accident Classification": accident_auto,
            "Annotations": annotations_auto
        })

        # Safe String Formats
        time_str = f"{collision_start_time:.2f}s" if (accident_detected_flag and collision_start_time is not None) else "N/A"
        conf_str = f"{first_impact_conf:.2%}" if accident_detected_flag else "N/A"

        # Sheet 2 Record
        evaluation_records.append({
            "Video Name": video_file,
            "Ground Truth": ground_truth,
            "Predicted Status": model_prediction,
            "Collision Detected": "Yes" if accident_detected_flag else "No",
            "Collision Timestamp (s)": time_str,
            "Impact Frame": first_impact_frame_num if accident_detected_flag else "N/A",
            "Confidence": conf_str,
            "Processing Speed (FPS)": avg_processing_fps,
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn
        })

        status_str = f"🚨 Collision @ {time_str}" if accident_detected_flag else "🟢 Normal Traffic"
        print(f"[{idx}/{len(video_files)}] {video_file:<30} | GT: {ground_truth:<12} | Pred: {model_prediction:<12} | {status_str}")

    # ------------------------------------------------------------
    # 📈 METRIC CALCULATIONS & SUMMARY ROW
    # ------------------------------------------------------------
    total_samples = len(video_files)
    accuracy = (total_tp + total_tn) / total_samples if total_samples > 0 else 0.0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    evaluation_records.append({
        "Video Name": "OVERALL METRICS SUMMARY",
        "Ground Truth": f"Total: {total_samples}",
        "Predicted Status": "-",
        "Collision Detected": "-",
        "Collision Timestamp (s)": f"Accuracy: {accuracy:.2%}",
        "Impact Frame": f"Precision: {precision:.2%}",
        "Confidence": f"Recall: {recall:.2%}",
        "Processing Speed (FPS)": f"F1-Score: {f1_score:.2%}",
        "TP": total_tp,
        "TN": total_tn,
        "FP": total_fp,
        "FN": total_fn
    })

    # Export to Excel
    df_descriptive = pd.DataFrame(descriptive_records)
    df_evaluation = pd.DataFrame(evaluation_records)

    with pd.ExcelWriter(OUTPUT_EXCEL_PATH, engine="openpyxl") as writer:
        df_descriptive.to_excel(writer, sheet_name="videos descriptive", index=False)
        df_evaluation.to_excel(writer, sheet_name="evaluation", index=False)

    print("=" * 80)
    print("🎉 EVALUATION COMPLETE!")
    print(f"   • Total Video Samples : {total_samples}")
    print(f"   • TP: {total_tp} | TN: {total_tn} | FP: {total_fp} | FN: {total_fn}")
    print(f"   • Accuracy : {accuracy:.2%}")
    print(f"   • Precision: {precision:.2%}")
    print(f"   • Recall   : {recall:.2%}")
    print(f"   • F1 Score : {f1_score:.2%}")
    print("=" * 80)
    print(f"📁 Results saved successfully to '{OUTPUT_EXCEL_PATH}'")


if __name__ == "__main__":
    run_batch_evaluation()