import cv2
import dlib
import os
from imutils import face_utils
from src.ratio import eye_aspect_ratio, mouth_aspect_ratio

EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 15

MAR_THRESHOLD = 0.75
MAR_CONSEC_FRAMES = 15


class DrowsinessDetector:
    def __init__(self, predictor_path):
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(f"Predictor not found: {predictor_path}")

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

        self.counter = 0
        self.yawn_counter = 0

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)

        for rect in rects:
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            left_eye = shape[42:48]
            right_eye = shape[36:42]
            mouth = shape[48:68]

            ear = (
                eye_aspect_ratio(left_eye) +
                eye_aspect_ratio(right_eye)
            ) / 2.0

            if ear < EAR_THRESHOLD:
                self.counter += 1
            else:
                self.counter = 0

            mar = mouth_aspect_ratio(mouth)

            if mar > MAR_THRESHOLD:
                self.yawn_counter += 1
            else:
                self.yawn_counter = 0

            eye_alert = self.counter >= EAR_CONSEC_FRAMES
            yawn_alert = self.yawn_counter >= MAR_CONSEC_FRAMES

            return ear, eye_alert, mar, yawn_alert

        return None, False, None, False
