import cv2
import mediapipe as mp
import time
import random
import joblib

clf = joblib.load('asl.joblib')

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

current_letter = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
predicted_letter = ""
score = 0
time_limit = 5  # seconds per letter
start_time = time.time()

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Camera not accessible")
            break

        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = [lm.x for lm in hand_landmarks.landmark] + \
                            [lm.y for lm in hand_landmarks.landmark] + \
                            [lm.z for lm in hand_landmarks.landmark]
                try:
                    predicted_letter = clf.predict([landmarks])[0]
                    if predicted_letter == current_letter:
                        score += 1
                        current_letter = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                        start_time = time.time()
                except:
                    predicted_letter = ""

        elapsed_time = time.time() - start_time
        if elapsed_time > time_limit:
            current_letter = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            start_time = time.time()

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, f"Letter: {current_letter}", (50, 50), font, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Your Sign: {predicted_letter}", (50, 100), font, 1, (255, 255, 0), 2)
        cv2.putText(image, f"Score: {score}", (50, 150), font, 1, (0, 0, 255), 2)
        cv2.putText(image, f"Time: {round(time_limit - elapsed_time, 1)}", (50, 200), font, 1, (255, 0, 0), 2)

        cv2.imshow('ASL Typing Game', image)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('w'):
            break

cap.release()
cv2.destroyAllWindows()
