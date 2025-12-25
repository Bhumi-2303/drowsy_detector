import cv2
import csv
import os
import dlib
from imutils import face_utils
from ratio import eye_aspect_ratio, mouth_aspect_ratio

# Path Setup 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

DATASET_DIR = os.path.join(ROOT_DIR, "dataset", "raw", "videos")
OUTPUT_CSV = os.path.join(ROOT_DIR, "dataset", "processed", "features.csv")

PREDICTOR_PATH = os.path.join(
    ROOT_DIR, "models", "shape_predictor_68_face_landmarks.dat"
)

# Constants

EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.75

# Dlib Models

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# Video Processing

def process_video(video_path, gender, camera, writer):
    cap = cv2.VideoCapture(video_path)
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id % 5 != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            left_eye = shape[42:48]
            right_eye = shape[36:42]
            mouth = shape[48:68]

            ear = (eye_aspect_ratio(left_eye) +
                   eye_aspect_ratio(right_eye)) / 2.0
            mar = mouth_aspect_ratio(mouth)

            label = 1 if (ear < EAR_THRESHOLD or mar > MAR_THRESHOLD) else 0

            writer.writerow([ear, mar, label, gender, camera])

    cap.release()

# Main

def main():
    os.makedirs(
        os.path.join(ROOT_DIR, "dataset", "processed"),
        exist_ok=True
    )

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ear", "mar", "label", "gender", "camera"])

        for gender in ["male", "female"]:
            for camera in ["dash", "mirror"]:
                folder = os.path.join(DATASET_DIR, gender, camera)

                # ✅ SAFETY CHECK (CORRECT PLACE)
                if not os.path.exists(folder):
                    print(f"Skipping missing folder: {folder}")
                    continue

                for video in os.listdir(folder):
                    if video.lower().endswith((".avi", ".mp4")):
                        video_path = os.path.join(folder, video)
                        print(f"Processing {video_path}")
                        process_video(video_path, gender, camera, writer)

    print("Feature extraction completed")

if __name__ == "__main__":
    main()
