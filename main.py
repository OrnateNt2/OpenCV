import cv2
import numpy as np
import sys
from collections import deque

video_path = "input/an1.mp4"
output_path = "output/an1-1.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Не удалось открыть видео.")
    sys.exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 24  # если не удалось определить fps, возьмём 24

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if total_frames <= 0:
    total_frames = None

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Параметры
k = 1.5         # коэффициент для порога яркости
box_size = 10    # размер половины стороны квадрата вокруг яркой точки
max_no_detection_time = 0.5
max_no_detection_frames = int(fps * max_no_detection_time)

no_detection_counter = max_no_detection_frames + 1
frame_count = 0

# Инициализация фильтра Калмана
kalman = cv2.KalmanFilter(4, 2)  # 4 состояния (x, y, dx, dy), 2 измерения (x, y)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                      [0, 1, 0, 0]], dtype=np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                     [0, 1, 0, 1],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]], dtype=np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
kalman.errorCovPost = np.eye(4, dtype=np.float32)
kalman.statePost = np.array([0, 0, 0, 0], dtype=np.float32).reshape(-1, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    meanVal = np.mean(gray)
    stdVal = np.std(gray)
    threshold_val = meanVal + k * stdVal

    # Находим самую яркую точку
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(gray)

    if maxVal > threshold_val:
        # Обновляем фильтр Калмана измерением
        measurement = np.array([[np.float32(maxLoc[0])], [np.float32(maxLoc[1])]])
        kalman.correct(measurement)

        no_detection_counter = 0
    else:
        # Если объект не найден, используем предсказание Калмана
        no_detection_counter += 1

    if no_detection_counter > max_no_detection_frames:
        # Нет дрона, показываем сообщение
        cv2.putText(frame, "No drone detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        # Предсказание фильтра Калмана
        prediction = kalman.predict()
        cx, cy = int(prediction[0]), int(prediction[1])

        x1 = max(cx - box_size, 0)
        y1 = max(cy - box_size, 0)
        x2 = min(cx + box_size, width - 1)
        y2 = min(cy + box_size, height - 1)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Drone: x={x1}, y={y1}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    out.write(frame)

    # Прогресс-бар
    if total_frames is not None and total_frames > 0:
        progress = (frame_count / total_frames) * 100
        sys.stdout.write("\rProcessing: {:.2f}%".format(progress))
        sys.stdout.flush()
    else:
        sys.stdout.write("\rFrame: {}".format(frame_count))
        sys.stdout.flush()

cap.release()
out.release()
print("\nОбработка завершена. Видео сохранено в", output_path)
