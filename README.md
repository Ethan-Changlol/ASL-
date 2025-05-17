# ASL-
# Spell it with Sign Language: An ASL Typing Game

This project is a Python-based typing-style game that helps users practice American Sign Language (ASL) fingerspelling using real-time hand tracking and gesture recognition.

Players “type” letters by signing them into a webcam. The system uses [MediaPipe](https://developers.google.com/mediapipe) to detect hand landmarks, and a machine learning model (built with scikit-learn) to classify static signs. A second model built with PyTorch is used to recognize dynamic signs like “J” and “Z” using motion patterns.

## 🎮 Features

- Real-time gesture recognition using webcam input
- Supports all 26 ASL fingerspelling letters
- Two-model system for static and dynamic signs
- Real-time feedback and scoring
- Educational and beginner-friendly

## 📦 Requirements

- Python 3.8+
- mediapipe
- opencv-python
- scikit-learn
- torch
- numpy

Install dependencies using:

```bash
pip install -r requirements.txt
