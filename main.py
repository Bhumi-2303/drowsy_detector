import cv2
import os
from src.utils import start_alarm, stop_alarm
from src.detector import DrowsinessDetector

# CONFIG FLAGS

ENABLE_DISPLAY = False   # ❌ set False in Docker / headless

# PATH SETUP

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    ROOT_DIR, "models", "shape_predictor_68_face_landmarks.dat"
)

BUZZER_PATH = os.path.join(
    ROOT_DIR, "assets", "buzzer.mp3"
)

# SAFETY CHECKS

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Predictor not found: {MODEL_PATH}")

# INITIALIZE DETECTOR

detector = DrowsinessDetector(MODEL_PATH)

# WEBCAM (LOCAL ONLY)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    ear, eye_alert, mar, yawn_alert = detector.process_frame(frame)

    if ear is None:
        stop_alarm()

        if ENABLE_DISPLAY:
            cv2.imshow("Drowsiness Detector", frame)
            if cv2.waitKey(1) == 27:
                break

        continue

    # EAR
    cv2.putText(
        frame, f"EAR: {ear:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # MAR
    cv2.putText(
        frame, f"MAR: {mar:.2f}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    # Alerts
    if eye_alert or yawn_alert:
        start_alarm(BUZZER_PATH)

        cv2.putText(
            frame,
            "DROWSINESS ALERT!",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3
        )

        if yawn_alert:
            cv2.putText(
                frame,
                "YAWNING DETECTED!",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 165, 255),
                3
            )
    else:
        stop_alarm()

    if ENABLE_DISPLAY:
        cv2.imshow("Drowsiness Detector", frame)
        if cv2.waitKey(1) == 27:
            break

cap.release()

if ENABLE_DISPLAY:
    cv2.destroyAllWindows()

stop_alarm()
