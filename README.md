# 🚗 Driver Drowsiness Detection System

A hybrid **real-time + machine learning** based driver drowsiness detection system that monitors driver fatigue using facial cues such as **eye closure (EAR)** and **yawning (MAR)**.

---

## 📌 Overview

This project combines:

* **Rule-based real-time detection** for live webcam monitoring
* **Machine learning classification** for image and video uploads

The system raises **visual and audio alerts** when drowsiness is detected.

---

## 🧠 Architecture

### 🔹 Real-Time (Rule-Based)

* Eye Aspect Ratio (EAR) for eye closure
* Mouth Aspect Ratio (MAR) for yawning
* Fatigue score (0–100)
* Audio alarm on drowsiness

### 🔹 ML-Based (Offline Analysis)

* EAR & MAR feature extraction
* RandomForest classifier
* Image & video upload classification

### 🔹 UI

* Streamlit-based web interface
* Webcam, image, and video modes

---

## 🛠️ Tech Stack

* Python 3
* OpenCV, Dlib
* Scikit-learn
* Streamlit
* Pygame

---

## 📂 Project Structure

```
drowsy_detector/
├── src/          # core logic & UI
├── models/       # pretrained landmark model
├── assets/       # alarm sound
├── main.py       # realtime detection
├── README.md
└── .gitignore
```

---

## 📊 Dataset

Trained using public benchmark datasets:

* NTHU Driver Drowsiness Dataset

Labels are generated using EAR & MAR thresholds.

---

## ▶️ Run

```bash
streamlit run src/gui.py
```

---

## ✨ Features

* Real-time drowsiness detection
* Yawn detection
* Fatigue score visualization
* Audio alarm
* Image & video classification

---

## 👤 Author

**Bhavini Chauhan**
3rd Year IT Engineering Student

---

## 📜 License

Educational & research use only.
