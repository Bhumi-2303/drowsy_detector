import cv2
import os
import time
import numpy as np
from collections import deque

from src.detector import DrowsinessDetector
from src.utils import start_alarm, stop_alarm

# =========================
# CONFIGURATION
# =========================
NO_FACE_FRAMES = 10
WINDOW_NAME = "Driver Drowsiness Detection"

CALIBRATION_FRAMES = 50        # time to learn your normal face
SMOOTH_WINDOW = 10             # smoothing
CONSEC_FRAMES = 20             # strong drowsiness requirement
COOLDOWN = 6                   # alarm cooldown

# =========================
# PATH SETUP
# =========================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTOR_PATH = os.path.join(ROOT_DIR, "models", "shape_predictor_68_face_landmarks.dat")
BUZZER_PATH = os.path.join(ROOT_DIR, "assets", "buzzer.mp3")

# =========================
# SAFETY CHECKS
# =========================
if not os.path.exists(PREDICTOR_PATH):
    raise FileNotFoundError("Face landmark model not found")
if not os.path.exists(BUZZER_PATH):
    raise FileNotFoundError("Buzzer audio not found")

# =========================
# INITIALIZATION
# =========================
detector = DrowsinessDetector(predictor_path=PREDICTOR_PATH)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam not accessible")

print("✅ Drowsiness Detection Started")
print("Keep your eyes OPEN for 2 seconds — calibrating...")
print("Press ESC to exit")

no_face_counter = 0
alarm_active = False
last_alarm_time = 0
counter = 0

# =========================
# CALIBRATION STORAGE
# =========================
ear_values = []
mar_values = []
ear_smooth = deque(maxlen=SMOOTH_WINDOW)
mar_smooth = deque(maxlen=SMOOTH_WINDOW)

ear_threshold = 0.22
mar_threshold = 0.70

# =========================
# UI CARD
# =========================
def draw_card(frame, x, y, w, h, title, value, header_color):
    cv2.rectangle(frame,(x,y),(x+w,y+h),(30,30,30),-1)
    cv2.rectangle(frame,(x,y),(x+w,y+30),header_color,-1)
    cv2.putText(frame,title,(x+10,y+22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
    cv2.putText(frame,value,(x+10,y+65),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

# =========================
# MAIN LOOP
# =========================
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    ear, eye_alert, mar, yawn_alert = detector.process_frame(frame)

    # =========================
    # NO FACE
    # =========================
    if ear is None:
        no_face_counter += 1
        cv2.putText(frame,"Face Not Detected",(30,40),
                    cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)

        if no_face_counter > NO_FACE_FRAMES:
            stop_alarm()
            alarm_active = False

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) == 27:
            break
        continue
    else:
        no_face_counter = 0

    frame_count += 1

    # =========================
    # CALIBRATION
    # =========================
    if frame_count <= CALIBRATION_FRAMES:
        ear_values.append(ear)
        mar_values.append(mar)

        cv2.putText(frame,"Calibrating... Keep eyes open.",
                    (30,40),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,0),2)

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    # Compute adaptive thresholds once
    if frame_count == CALIBRATION_FRAMES + 1:
        ear_threshold = max(0.15, np.mean(ear_values) * 0.72)
        mar_threshold = np.mean(mar_values) * 1.35
        print("🔧 Adaptive EAR Threshold =", round(ear_threshold,3))
        print("🔧 Adaptive MAR Threshold =", round(mar_threshold,3))

    # =========================
    # SMOOTHING
    # =========================
    ear_smooth.append(ear)
    mar_smooth.append(mar)

    ear_avg = np.mean(ear_smooth)
    mar_avg = np.mean(mar_smooth)

    # =========================
    # PURE RULE-BASED DECISION
    # =========================
    rule_alert = (ear_avg < ear_threshold) or (mar_avg > mar_threshold)

    if rule_alert:
        counter += 1
    else:
        counter = 0

    final_alert = counter >= CONSEC_FRAMES

    # =========================
    # DISPLAY
    # =========================
    draw_card(frame,20,20,180,90,"EAR",f"{ear_avg:.2f}",(0,180,0))
    draw_card(frame,220,20,180,90,"MAR",f"{mar_avg:.2f}",(0,180,180))

    # =========================
    # ALERT WITH COOLDOWN
    # =========================
    now = time.time()

    if final_alert:
        if not alarm_active and (now - last_alarm_time > COOLDOWN):
            start_alarm(BUZZER_PATH)
            alarm_active = True
            last_alarm_time = now

        h,w = frame.shape[:2]
        cv2.rectangle(frame,(0,h-80),(w,h),(0,0,255),-1)
        cv2.putText(frame,"⚠ DRIVER DROWSINESS DETECTED",
                    (40,h-30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),3)
    else:
        stop_alarm()
        alarm_active = False

    # =========================
    # FOOTER
    # =========================
    cv2.putText(frame,"Rule-Based Adaptive Detection (EAR + MAR)",
                (10,frame.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1)

    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
stop_alarm()
print("🛑 Stopped")
