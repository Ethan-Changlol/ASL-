import cv2
import time
from ultralytics import YOLO

# 設定攝影機
cap = cv2.VideoCapture(0)

model = YOLO('asl.pt')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("無法讀取攝影機影像")
        break

    # 翻轉影像以符合鏡像效果
    frame = cv2.flip(frame, 1)

    # 辨識ASL
    results = model(frame)

    image = results[0].plot()

    # 顯示影像
    cv2.imshow('ASL', image)

    # 鍵盤控制
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC 鍵結束程式
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()