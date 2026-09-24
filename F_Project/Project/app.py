import streamlit as st
import cv2
import tempfile
import os
import time
import pandas as pd

from accident_detector import AccidentDetector

# ============================================================
# FIXED DETECTION PARAMETERS
# ============================================================

YOLO_CONFIDENCE = 0.75
ACCIDENT_CONFIRMATION_FRAMES = 10
PROCESS_EVERY_N_FRAMES = 4

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Accident Detection",
    page_icon="🚨",
    layout="wide"
)

st.sidebar.markdown(
    """
    <div style="
        padding:20px;
        border-radius:15px;
        text-align:center;
        margin-top:20px;
        border:1px solid rgba(100,150,255,0.3);
    ">
        <div style="font-size:45px;">🚨</div>
        <h3 style="margin:5px 0;">AI Accident Detection safety</h3>
        <p style="font-size:13px;">
            Intelligent Accident Detection
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .team-box {
    position: auto;
    top: 80px;
    right: 25px;
    padding: 12px 20px;
    border-radius: 12px;
    border: 1px solid rgba(100, 150, 255, 0.4);
    background: rgba(20, 30, 60, 0.9);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    z-index: 9999;
    text-align: center;
}

.team-title {
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 5px;
}

.team-names {
    font-size: 14px;
    line-height: 1.5;
}


    .main-title {
        font-size: 40px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        opacity: 0.8;
        margin-bottom: 25px;
    }

    .metric-card {
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(100,100,100,0.2);
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">AI Accident Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'YOLO11n + ByteTrack + Temporal Accident Analysis'
    '</div>',
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="team-box">
        <div class="team-title">👥 Code and Construct Team</div>
        <div class="team-names">
            Shreya S<br>
            Tejaswini P<br>
            Varsha D N<br>
            Bhavana S R
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


st.write(
    "Upload a CCTV/dashcam video. "
    "The system analyzes vehicle movement, collisions, "
    "sudden stops, abnormal motion and roadside interactions."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "🎥 Upload Video"
)

uploaded_file = st.sidebar.file_uploader(
    "Choose video",
    type=[
        "mp4",
        "mov",
        "avi",
        "mkv"
    ]
)



# ============================================================
# MAIN
# ============================================================

if uploaded_file is not None:

    # ========================================================
    # SAVE TEMP VIDEO
    # ========================================================

    temp_video_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as temp_file:

            temp_file.write(
                uploaded_file.read()
            )

            temp_video_path = (
                temp_file.name
            )

        # ====================================================
        # OPEN VIDEO
        # ====================================================

        cap = cv2.VideoCapture(
            temp_video_path
        )

        if not cap.isOpened():

            st.error(
                "❌ Could not open the uploaded video."
            )

            st.stop()

        # ====================================================
        # VIDEO INFORMATION
        # ====================================================

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        if fps <= 0:

            fps = 30.0

        duration = (
            total_frames /
            fps
        )

        # ====================================================
        # VIDEO INFORMATION DISPLAY
        # ====================================================

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Resolution",
                f"{width} × {height}"
            )

        with col2:

            st.metric(
                "Video FPS",
                f"{fps:.2f}"
            )

        with col3:

            st.metric(
                "Frames",
                total_frames
            )

        with col4:

            st.metric(
                "Duration",
                f"{duration:.1f} s"
            )

        st.write("---")

        # ====================================================
        # LOAD DETECTOR
        # ====================================================

        with st.spinner(
            "Loading YOLO11n model..."
        ):

            detector = AccidentDetector(
                model_path="yolo11n.pt",

                conf_threshold=YOLO_CONFIDENCE,

                temporal_buffer_limit=ACCIDENT_CONFIRMATION_FRAMES
            )

        # ====================================================
        # UI
        # ====================================================

        st.subheader(
            "🎥 Live Video Analysis"
        )

        frame_placeholder = st.empty()

        st.write("---")

        status_col, progress_col = st.columns(
            [1, 2]
        )

        with status_col:

            status_placeholder = st.empty()

        with progress_col:

            progress_placeholder = st.progress(
                0
            )

        # ====================================================
        # DETECTION RESULT
        # ====================================================

        result_placeholder = st.empty()

        # ====================================================
        # VARIABLES
        # ====================================================

        frame_number = 0

        processed_frames = 0

        accident_detected = False

        accident_timestamp = None

        accident_reason = ""

        accident_frame = None

        detected_events_all = []

        start_time = time.time()

        # ====================================================
        # PROCESS VIDEO
        # ====================================================

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1

            # ------------------------------------------------
            # Skip frames according to setting
            # ------------------------------------------------

            if (
                frame_number %
                PROCESS_EVERY_N_FRAMES
                != 0
            ):

                continue

            processed_frames += 1

            # ------------------------------------------------
            # Timestamp
            # ------------------------------------------------

            timestamp = (
                frame_number /
                fps
            )

            # ------------------------------------------------
            # Resize for processing/display
            # ------------------------------------------------

            processing_frame = cv2.resize(
                frame,
                (854, 480)
            )

            # ------------------------------------------------
            # Run detector
            # ------------------------------------------------

            (
                processed_frame,
                alert_triggered,
                buffer_count,
                info
            ) = detector.process_frame(
                processing_frame,
                timestamp=timestamp
            )

            # =================================================
            # EVENTS
            # =================================================

            current_events = info[
                "events"
            ]

            for event in current_events:

                if event not in detected_events_all:

                    detected_events_all.append(
                        event
                    )

            # =================================================
            # PROGRESS
            # =================================================

            progress = min(
                1.0,
                frame_number /
                max(
                    1,
                    total_frames
                )
            )

            progress_placeholder.progress(
                progress
            )

            # =================================================
            # DISPLAY FRAME
            # =================================================

            rgb = cv2.cvtColor(
                processed_frame,
                cv2.COLOR_BGR2RGB
            )

            frame_placeholder.image(
                rgb,
                channels="RGB",
                width="stretch"
            )

            # =================================================
            # STATUS
            # =================================================

            if alert_triggered:

                status_placeholder.error(
                    "🚨 ACCIDENT CONFIRMED"
                )

                accident_detected = True

                accident_timestamp = (
                    info["timestamp"]
                )

                accident_reason = (
                    info["reason"]
                )

                accident_frame = (
                    processed_frame.copy()
                )

                result_placeholder.error(
                    f"""
                    🚨 ACCIDENT DETECTED

                    Time: {accident_timestamp:.2f} seconds

                    Reason:
                    {accident_reason}
                    """
                )

                # Stop at first confirmed accident
                break

            elif current_events:

                status_placeholder.warning(
                    "⚠️ Abnormal event being analyzed..."
                )

            else:

                status_placeholder.success(
                    "🟢 Normal traffic monitoring"
                )

            # =================================================
            # SMALL DELAY FOR UI
            # =================================================

            time.sleep(
                0.001
            )

        # ====================================================
        # RELEASE
        # ====================================================

        cap.release()

        total_processing_time = (
            time.time() -
            start_time
        )

        # ====================================================
        # FINAL RESULT
        # ====================================================

        st.write("---")

        st.subheader(
            "📊 Final Analysis"
        )

        if accident_detected:

            st.error(
                "🚨 ACCIDENT DETECTED"
            )

            # ------------------------------------------------
            # Accident details
            # ------------------------------------------------

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Accident Time",
                    f"{accident_timestamp:.2f} s"
                )

            with col2:

                st.metric(
                    "Processed Frames",
                    processed_frames
                )

            with col3:

                st.metric(
                    "Analysis Time",
                    f"{total_processing_time:.2f} s"
                )

            st.write(
                "**Detected event:**"
            )

            st.warning(
                accident_reason
            )

            # ------------------------------------------------
            # Accident frame
            # ------------------------------------------------

            if accident_frame is not None:

                st.subheader(
                    "🚨 Accident Frame"
                )

                accident_rgb = cv2.cvtColor(
                    accident_frame,
                    cv2.COLOR_BGR2RGB
                )

                st.image(
                    accident_rgb,
                    channels="RGB",
                    width="stretch"
                )

            # ------------------------------------------------
            # Data table
            # ------------------------------------------------

            analysis_data = [
                {
                    "Video":
                        uploaded_file.name,

                    "Resolution":
                        f"{width} × {height}",

                    "Original FPS":
                        f"{fps:.2f}",

                    "Processed Frames":
                        processed_frames,

                    "Accident":
                        "YES",

                    "Timestamp":
                        f"{accident_timestamp:.2f} s",

                    "Reason":
                        accident_reason
                }
            ]

            df = pd.DataFrame(
                analysis_data
            )

            st.dataframe(
                df,
                width="stretch"
            )

        else:

            st.success(
                "✅ No confirmed accident detected."
            )

            st.info(
                "The complete video was analyzed. "
                "The system did not observe enough "
                "evidence to confirm an accident."
            )

            # ------------------------------------------------
            # Events observed
            # ------------------------------------------------

            if detected_events_all:

                st.subheader(
                    "⚠️ Abnormal Events Observed"
                )

                for event in detected_events_all:

                    st.write(
                        f"• {event}"
                    )

            # ------------------------------------------------
            # Summary table
            # ------------------------------------------------

            analysis_data = [
                {
                    "Video":
                        uploaded_file.name,

                    "Resolution":
                        f"{width} × {height}",

                    "Original FPS":
                        f"{fps:.2f}",

                    "Processed Frames":
                        processed_frames,

                    "Accident":
                        "NO",

                    "Analysis Time":
                        f"{total_processing_time:.2f} s",

                    "Events Observed":
                        ", ".join(
                            detected_events_all
                        )
                        if detected_events_all
                        else "None"
                }
            ]

            df = pd.DataFrame(
                analysis_data
            )

            st.dataframe(
                df,
                width="stretch"
            )

    finally:

        # ====================================================
        # DELETE TEMP FILE
        # ====================================================

        if (
            temp_video_path is not None and
            os.path.exists(
                temp_video_path
            )
        ):

            try:

                os.remove(
                    temp_video_path
                )

            except Exception:

                pass


# ============================================================
# INFORMATION
# ============================================================

else:

    st.info(
        "👈 Upload a video from the sidebar to start analysis."
    )