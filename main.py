import cv2
import numpy as np
from collections import deque
import sys

# Путь к вашему видео
video_path = "2.mp4"
output_path = "2test2.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Не удалось открыть видео.")
    sys.exit(1)

# Получаем свойства видео
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 24  # если не удалось определить, ставим 24 по умолчанию
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if total_frames <= 0:
    total_frames = None

print(f"Video properties: width={width}, height={height}, fps={fps}, total_frames={total_frames}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Параметры
N = 15                     # Кол-во кадров для медианного фона
k = 1.5                    # Коэффициент для адаптивного порога (mean + k*std)
morph_kernel_size = 3      # Размер ядра для морфологии
enlarge_factor = 0.05       # Уменьшим область вокруг дрона
max_no_detection_time = 0.5 # 0.5 сек держим last known
max_no_detection_frames = int(fps * max_no_detection_time)
threshold_min_size = 2      # минимальный размер объекта

# Трекер
tracker_type = "CSRT"
def create_tracker(tracker_type="CSRT"):
    if cv2.__version__.startswith('4.'):
        if tracker_type == 'CSRT':
            return cv2.legacy.TrackerCSRT_create()
        elif tracker_type == 'KCF':
            return cv2.legacy.TrackerKCF_create()
        else:
            return cv2.legacy.TrackerCSRT_create()
    else:
        # OpenCV 3.x с contrib
        if tracker_type == 'CSRT':
            return cv2.TrackerCSRT_create()
        elif tracker_type == 'KCF':
            return cv2.TrackerKCF_create()
        else:
            return cv2.TrackerCSRT_create()

frames_queue = deque(maxlen=N)

#  первые N кадров
initial_frames_read = 0
while initial_frames_read < N:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frames_queue.append(gray)
    initial_frames_read += 1

if len(frames_queue) < N:
    print("Недостаточно кадров для вычисления фона")
    cap.release()
    out.release()
    sys.exit(1)

background = np.median(np.array(frames_queue), axis=0).astype(np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))

cap.set(cv2.CAP_PROP_POS_FRAMES, N)
frame_count = N
tracker = None
tracking = False
last_drone_rect = None
no_detection_counter = max_no_detection_frames + 1

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Адаптивный порог
    meanVal = np.mean(gray)
    stdVal = np.std(gray)
    dynamic_threshold = meanVal + k * stdVal

    diff = cv2.absdiff(gray, background)
    _, thresh = cv2.threshold(diff, dynamic_threshold, 255, cv2.THRESH_BINARY)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    ok = False

    if tracking and tracker is not None:
        # Попытка обновить трекер
        ok, bbox = tracker.update(frame)
        if ok:
            x, y, w, h = [int(v) for v in bbox]
            dw = int(w * enlarge_factor)
            dh = int(h * enlarge_factor)
            nx = max(x - dw, 0)
            ny = max(y - dh, 0)
            nw = min(w + 2*dw, width - nx)
            nh = min(h + 2*dh, height - ny)

            last_drone_rect = (nx, ny, nw, nh)
            no_detection_counter = 0
        else:
            tracking = False
            tracker = None
            no_detection_counter += 1
    else:
        # Детекция дрона
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_contour = None
        best_brightness = -1
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < threshold_min_size or h < threshold_min_size:
                continue
            # Вычисляем среднюю яркость
            mask = np.zeros_like(gray, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_val = cv2.mean(gray, mask=mask)[0]
            if mean_val > best_brightness:
                best_brightness = mean_val
                best_contour = cnt

        if best_contour is not None:
            x, y, w, h = cv2.boundingRect(best_contour)
            dw = int(w * enlarge_factor)
            dh = int(h * enlarge_factor)
            nx = max(x - dw, 0)
            ny = max(y - dh, 0)
            nw = min(w + 2*dw, width - nx)
            nh = min(h + 2*dh, height - ny)

            last_drone_rect = (nx, ny, nw, nh)
            no_detection_counter = 0

            tracker = create_tracker(tracker_type)
            tracker.init(frame, (nx, ny, nw, nh))
            tracking = True
        else:
            no_detection_counter += 1

    # Отображение
    if no_detection_counter <= max_no_detection_frames and last_drone_rect is not None and tracking and ok:
        dx, dy, dw, dh = last_drone_rect
        cv2.rectangle(frame, (dx, dy), (dx+dw, dy+dh), (0, 255, 0), 2)
        cv2.putText(frame, f"Drone: x={dx}, y={dy}, w={dw}, h={dh}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No drone detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    out.write(frame)

    frames_queue.append(gray)
    if len(frames_queue) == N and frame_count % 30 == 0:
        background = np.median(np.array(frames_queue), axis=0).astype(np.uint8)

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
