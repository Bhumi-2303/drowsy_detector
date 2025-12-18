import cv2
import dlib
import joblib
from imutils import face_utils
from ratio import eye_aspect_ratio, mouth_aspect_ratio

MODEL_PATH = "models/fatigue_classifier.pkl"
PREDICTOR_PATH = "models/shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
model = joblib.load(MODEL_PATH)

def predict_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

        prediction = model.predict([[ear, mar]])[0]
        confidence = model.predict_proba([[ear, mar]])[0].max()

        return prediction, confidence

    return None, None

def predict_video(video_path, frame_skip=5):
    cap = cv2.VideoCapture(video_path)
    predictions = []

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id % frame_skip != 0:
            continue

        pred, _ = predict_image(frame)
        if pred is not None:
            predictions.append(pred)

    cap.release()

    if not predictions:
        return None, None

    # Majority vote
    drowsy_ratio = sum(predictions) / len(predictions)
    final_pred = 1 if drowsy_ratio > 0.5 else 0

    return final_pred, drowsy_ratio

