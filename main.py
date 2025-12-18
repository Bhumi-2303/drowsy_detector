import cv2
from src.utils import start_alarm, stop_alarm
from src.detector import DrowsinessDetector

predictor_path = "models/shape_predictor_68_face_landmarks.dat"
detector = DrowsinessDetector(predictor_path)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    ear, eye_alert, mar, yawn_alert = detector.process_frame(frame)

    # If no face detected
    if ear is None:
        stop_alarm()
        cv2.imshow("Drowsiness Detector", frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    # Show EAR
    cv2.putText(
        frame, f"EAR: {ear:.2f}", (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )

    # Show MAR
    cv2.putText(
        frame, f"MAR: {mar:.2f}", (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
    )

    # Alerts
    if eye_alert or yawn_alert:
        start_alarm("assets/buzzer.mp3")

        cv2.putText(
            frame, "DROWSINESS ALERT!", (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3
        )

        if yawn_alert:
            cv2.putText(
                frame, "YAWNING DETECTED!", (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,165,255), 3
            )
    else:
        stop_alarm()

    cv2.imshow("Drowsiness Detector", frame)

    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
