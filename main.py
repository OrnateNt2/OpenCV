import cv2
import numpy as np
import sys
from collections import deque

video_path = "input/an1.mp4"
output_path = "output/an1.mp4"

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

# Очередь для хранения последних N позиций яркой точки
N = 5  # количество кадров для сглаживания
positions_x = deque(maxlen=N)
positions_y = deque(maxlen=N)

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
        # Добавляем новую позицию в очередь
        positions_x.append(maxLoc[0])
        positions_y.append(maxLoc[1])

        if len(positions_x) == N:
            # Если набрали N кадров, вычисляем медиану координат
            median_x = int(np.median(positions_x))
            median_y = int(np.median(positions_y))

            x1 = max(median_x - box_size, 0)
            y1 = max(median_y - box_size, 0)
            x2 = min(median_x + box_size, width - 1)
            y2 = min(median_y + box_size, height - 1)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Drone: x={x1}, y={y1}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            no_detection_counter = 0
        else:
            # Пока не набрали N кадров - просто рисуем по текущей точке
            cx, cy = maxLoc
            x1 = max(cx - box_size, 0)
            y1 = max(cy - box_size, 0)
            x2 = min(cx + box_size, width - 1)
            y2 = min(cy + box_size, height - 1)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Drone: x={x1}, y={y1}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            no_detection_counter = 0
    else:
        # Нет яркой точки выше порога
        no_detection_counter += 1
        if no_detection_counter > max_no_detection_frames:
            cv2.putText(frame, "No drone detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # Можно реализовать last known, но пока просто "No drone detected"
            cv2.putText(frame, "No drone detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Если нет дрона, очищаем очереди позиций
        positions_x.clear()
        positions_y.clear()

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
