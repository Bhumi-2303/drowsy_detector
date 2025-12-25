import streamlit as st
import cv2
import numpy as np
import tempfile
import os

import sys
import os

# Add project root to PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


# ===============================
# FLAGS (VERY IMPORTANT)
# ===============================
ENABLE_WEBCAM = True      # Set False when running in Docker
ENABLE_ALARM = True       # Set False in Docker / CI

# PATH SETUP

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(
    ROOT_DIR, "models", "shape_predictor_68_face_landmarks.dat"
)

BUZZER_PATH = os.path.join(
    ROOT_DIR, "assets", "buzzer.mp3"
)

# SAFE IMPORTS

from src.predict_ml import predict_image, predict_video
from src.detector import DrowsinessDetector

# OPTIONAL AUDIO

if ENABLE_ALARM:
    import pygame
    pygame.mixer.init()
    ALARM_ON = False

    def start_alarm():
        global ALARM_ON
        if not ALARM_ON:
            pygame.mixer.music.load(BUZZER_PATH)
            pygame.mixer.music.play(-1)
            ALARM_ON = True

    def stop_alarm():
        global ALARM_ON
        if ALARM_ON:
            pygame.mixer.music.stop()
            ALARM_ON = False
else:
    def start_alarm(): pass
    def stop_alarm(): pass

# FATIGUE SCORE

def compute_fatigue_score(ear, mar):
    EAR_TH = 0.25
    MAR_TH = 0.75

    eye_score = max(0, (EAR_TH - ear) / EAR_TH)
    mouth_score = max(0, (mar - MAR_TH) / MAR_TH)

    fatigue = (0.6 * eye_score + 0.4 * mouth_score) * 100
    return min(int(fatigue), 100)

# STREAMLIT UI

st.set_page_config(page_title="Drowsiness Detection", layout="centered")
st.title("🚗 Driver Drowsiness Detection System")

mode = st.sidebar.selectbox(
    "Select Mode",
    [
        "Image Upload (ML)",
        "Video Upload (ML)",
        "Live Webcam (Rule-based)",
        "About Project"
    ]
)

# IMAGE UPLOAD

if mode == "Image Upload (ML)":
    st.header("🖼️ Upload Driver Image")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        st.image(image, channels="BGR")

        prediction, confidence = predict_image(image)

        if prediction is not None:
            st.subheader("😴 DROWSY" if prediction == 1 else "🙂 NOT DROWSY")
            st.write(f"Confidence: {confidence*100:.2f}%")
        else:
            st.warning("No face detected.")

# VIDEO UPLOAD

elif mode == "Video Upload (ML)":
    st.header("🎞️ Upload Driver Video")

    uploaded_video = st.file_uploader(
        "Upload a video file",
        type=["mp4", "avi"]
    )

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        st.video(tfile.name)

        with st.spinner("Analyzing video..."):
            pred, ratio = predict_video(tfile.name)

        if pred is not None:
            st.subheader("😴 DROWSY" if pred == 1 else "🙂 NOT DROWSY")
            st.write(f"Drowsy frame ratio: {ratio*100:.2f}%")
        else:
            st.warning("No face detected.")

# WEBCAM (LOCAL ONLY)

elif mode == "Live Webcam (Rule-based)":
    if not ENABLE_WEBCAM:
        st.warning("Webcam disabled in this environment.")
    else:
        st.header("📷 Live Driver Monitoring")
        start_btn = st.button("Start Webcam")

        if start_btn:
            detector = DrowsinessDetector(MODEL_PATH)
            cap = cv2.VideoCapture(0)

            frame_slot = st.empty()
            fatigue_slot = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                ear, eye_alert, mar, yawn_alert = detector.process_frame(frame)

                if ear is not None:
                    fatigue = compute_fatigue_score(ear, mar)
                    fatigue_slot.progress(fatigue)

                    if eye_alert or yawn_alert:
                        start_alarm()
                else:
                    stop_alarm()

                frame_slot.image(frame, channels="BGR")

            cap.release()
            stop_alarm()

# ABOUT

elif mode == "About Project":
    st.markdown("""
    ### 🚗 Driver Drowsiness Detection System

    - Rule-based EAR + MAR
    - ML classification
    - Image & video support
    - Docker & Windows friendly
    """)
