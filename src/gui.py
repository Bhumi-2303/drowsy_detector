import streamlit as st
import cv2
import numpy as np
import tempfile
import pygame

from predict_ml import predict_image, predict_video
from detector import DrowsinessDetector

pygame.mixer.init()
ALARM_ON = False

def start_alarm():
    global ALARM_ON
    if not ALARM_ON:
        pygame.mixer.music.load("assets/buzzer.mp3")
        pygame.mixer.music.play(-1)  
        ALARM_ON = True

def stop_alarm():
    global ALARM_ON
    if ALARM_ON:
        pygame.mixer.music.stop()
        ALARM_ON = False



#fatigu score computation
def compute_fatigue_score(ear, mar):
    EAR_TH = 0.25
    MAR_TH = 0.75

    eye_score = max(0, (EAR_TH - ear) / EAR_TH)
    mouth_score = max(0, (mar - MAR_TH) / MAR_TH)

    fatigue = (0.6 * eye_score + 0.4 * mouth_score) * 100
    return min(int(fatigue), 100)


#streamlit configuration
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

# image upload prediction using ML
if mode == "Image Upload (ML)":
    st.header("🖼️ Upload Driver Image")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        st.image(image, channels="BGR", caption="Uploaded Image")

        prediction, confidence = predict_image(image)

        if prediction is not None:
            if prediction == 1:
                st.subheader("😴 DROWSY")
            else:
                st.subheader("🙂 NOT DROWSY")

            st.write(f"Confidence: {confidence*100:.2f}%")
        else:
            st.warning("No face detected in the image.")


# video upload prediction using ML
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
            if pred == 1:
                st.subheader("😴 DROWSY")
            else:
                st.subheader("🙂 NOT DROWSY")

            st.write(f"Drowsy frame ratio: {ratio*100:.2f}%")
        else:
            st.warning("No face detected in video.")


# live-webcam
elif mode == "Live Webcam (Rule-based)":
    st.header("📷 Live Driver Monitoring")

    start = st.button("Start Webcam")
    stop = st.button("Stop Webcam")

    if start:
        detector = DrowsinessDetector(
            "models/shape_predictor_68_face_landmarks.dat"
        )

        cap = cv2.VideoCapture(0)
        frame_slot = st.empty()
        fatigue_slot = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop:
                break

            ear, eye_alert, mar, yawn_alert = detector.process_frame(frame)

            if ear is not None:
                fatigue = compute_fatigue_score(ear, mar)

                # SHOW ALERT 
                if eye_alert or yawn_alert:
                    cv2.putText(
                        frame,
                        "😴 DROWSY",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3
                    )

                    cv2.putText(
                        frame,
                        f"Fatigue: {fatigue}%",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )

                    st.subheader("😴 DROWSY")
                    start_alarm()          
                else:
                    stop_alarm()           


                    cv2.putText(
                        frame,
                        f"Fatigue: {fatigue}%",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )

                    st.subheader("😴 DROWSY")

                #shows fatigue
                fatigue_slot.progress(fatigue)

            frame_slot.image(frame, channels="BGR")
        stop_alarm()

        cap.release()


#ABOUT
elif mode == "About Project":
    st.markdown("""
    ### 🚗 Driver Drowsiness Detection System

    **Features**
    - Real-time rule-based detection (EAR + MAR)
    - ML-based classification (RandomForest)
    - Image & video upload support
    - Fatigue score visualization

    **Dataset**
    - Trained on benchmark datasets (YawDD / NTHU)

    **Architecture**
    - Rule-based engine for live monitoring
    - ML engine for offline image/video analysis
    """)
