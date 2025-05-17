import cv2
import mediapipe as mp
import time
import joblib

clf = joblib.load('asl.joblib')

# 初始化 MediaPipe 手部模型
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 設定攝影機
cap = cv2.VideoCapture(0)

# 控制變數
label = ""
landmarks = []

# 初始化 Hand 模型
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("無法讀取攝影機影像")
            break

        # 翻轉影像以符合鏡像效果
        frame = cv2.flip(frame, 1)

        # 轉換 BGR 影像到 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # 偵測手部
        results = hands.process(image)

        # 轉換回 BGR 以顯示
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 畫出手部關鍵點
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 如果開始錄製，儲存關鍵點座標
                landmarks = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)
                    landmarks.append(lm.z)

            try:
                label = clf.predict([landmarks])[0]
            except Exception as e:
                print(e)

        # 設定字型、大小和顏色
        font = cv2.FONT_HERSHEY_SIMPLEX    # 字型
        font_scale = 1.5                   # 字體大小
        color = (0, 0, 255)            # 文字顏色（白色）
        thickness = 2                      # 線條粗細

        # 取得文字尺寸（用來計算左下角位置）
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # 設定文字位置在圖片的左下角
        x = 10                            # 左邊留一點邊距
        y = text_height + 10              # 靠近圖片上緣，預留 10px 邊距

        # 將文字印到圖片上
        cv2.putText(image, label, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
        # 顯示影像
        cv2.imshow('Hand Tracking', image)

        # 鍵盤控制
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC 鍵結束程式
            break

# 釋放資源
cap.release()
cv2.destroyAllWindows()
