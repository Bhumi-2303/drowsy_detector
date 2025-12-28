import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import sys

# ===============================
# PATH SETUP
# ===============================

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.predict_ml import predict_image, predict_video

# ===============================
# STREAMLIT CONFIG
# ===============================

st.set_page_config(
    page_title="Driver Drowsiness Detection (ML)",
    layout="centered"
)

st.title("🚗 Driver Drowsiness Detection System")
st.caption("Image & Video Analysis using Machine Learning")

mode = st.sidebar.selectbox(
    "Select Mode",
    [
        "Image Upload",
        "Video Upload",
        "About Project"
    ]
)

# ===============================
# IMAGE UPLOAD
# ===============================

if mode == "Image Upload":
    st.header("🖼️ Upload Driver Image")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image_bytes = np.asarray(
            bytearray(uploaded_file.read()),
            dtype=np.uint8
        )
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        st.image(image, channels="BGR", caption="Uploaded Image")

        pred, conf = predict_image(image)

        if pred is not None:
            st.markdown(
                f"### {'😴 DROWSY' if pred == 1 else '🙂 ALERT'}"
            )
            st.write(f"Confidence: **{conf * 100:.2f}%**")
        else:
            st.warning("No face detected in the image.")

# ===============================
# VIDEO UPLOAD
# ===============================

elif mode == "Video Upload":
    st.header("🎥 Upload Driver Video")

    uploaded_video = st.file_uploader(
        "Upload a video file",
        type=["mp4", "avi"]
    )

    if uploaded_video:
        temp_video = tempfile.NamedTemporaryFile(delete=False)
        temp_video.write(uploaded_video.read())

        st.video(temp_video.name)

        with st.spinner("Analyzing video..."):
            pred, ratio = predict_video(temp_video.name)

        if pred is not None:
            st.markdown(
                f"### {'😴 DROWSY' if pred == 1 else '🙂 ALERT'}"
            )
            st.write(f"Drowsy Frame Ratio: **{ratio * 100:.2f}%**")
        else:
            st.warning("No face detected in the video.")

# ===============================
# ABOUT
# ===============================

else:
    st.markdown("""
    ### 🚗 Driver Drowsiness Detection System

    **Detection Modes**
    - Real-time webcam (OpenCV – rule-based + ML)
    - Image upload (ML)
    - Video upload (ML)

    **Key Concepts**
    - EAR & MAR feature extraction
    - Machine Learning classification
    - Hybrid decision logic

    **Technologies**
    - Python
    - OpenCV & dlib
    - Scikit-learn
    - Streamlit
    """)
