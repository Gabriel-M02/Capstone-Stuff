import cv2
import numpy as np
import psutil
import time
from ultralytics import YOLO
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# --- Optimization ---
cv2.setUseOptimized(True)
cv2.setNumThreads(2)

# --- Load YOLO model ---
print("[INFO] Loading YOLO model...")
model = YOLO("EVST_DataModelPrototypemk1/runs/detect/train/weights/best.torchscript")

# --- Battery HUD (with fallback) ---
def draw_battery_status(frame, x_offset, y_offset, scale=1.0):
    try:
        battery = psutil.sensors_battery()
        if battery is None:
            text = "EXT PWR"
            percent = 100
            charging = True
        else:
            percent = battery.percent
            charging = battery.power_plugged
            text = f"{int(percent)}%{'' if charging else ''}"
    except Exception:
        text = "EXT PWR"
        percent = 100
        charging = True

    width, height = int(120 * scale), int(30 * scale)
    outline_color = (255, 255, 255)
    fill_color = (0, 255, 0) if percent > 20 else (0, 0, 255)

    cv2.rectangle(frame, (x_offset, y_offset),
                  (x_offset + width, y_offset + height), outline_color, 2)
    cv2.rectangle(frame, (x_offset + width, y_offset + int(0.25 * height)),
                  (x_offset + width + int(6 * scale), y_offset + int(0.75 * height)), outline_color, -1)

    fill_width = int((percent / 100) * (width - 4))
    cv2.rectangle(frame, (x_offset + 2, y_offset + 2),
                  (x_offset + 2 + fill_width, y_offset + height - 2), fill_color, -1)

    cv2.putText(frame, text, (x_offset - int(60 * scale), y_offset + int(22 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (0, 255, 255), 2)

# --- HUD Overlay ---
def draw_hud(display_frame, fps):
    h, w, _ = display_frame.shape
    font_scale = max(0.6, w / 1280)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Title
    cv2.putText(display_frame, f"EVST NVG HUD MK2 - {time.strftime('%H:%M:%S')}",
                (int(0.03 * w), int(0.08 * h)), font, font_scale, (0, 255, 255), 2)
    # FPS
    cv2.putText(display_frame, f"FPS: {int(fps)}",
                (int(0.03 * w), int(0.95 * h)), font, font_scale, (0, 255, 0), 2)
    # Battery
    draw_battery_status(display_frame, int(0.82 * w), int(0.05 * h), scale=font_scale * 1.4)

# --- Detection Function ---
def detect_guns(frame, threshold=0.6):
    frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=15)
    results = model.predict(source=frame, imgsz=320, conf=threshold, verbose=False)
    boxes = results[0].boxes

    detections = []
    hostile_detected = False

    if boxes is not None and boxes.data.numel() > 0:
        data = boxes.data.cpu().numpy()
        for det in data:
            x1, y1, x2, y2, conf, cls = det
            detections.append((int(x1), int(y1), int(x2), int(y2), float(conf), int(cls)))
            # Only mark hostile if a Nerf gun (class 0) is detected
            if int(cls) == 0 and float(conf) > threshold:
                hostile_detected = True

    return hostile_detected, detections

# --- Initialize Camera ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Camera could not be opened.")
    exit()

print("[INFO] System Ready. Press 'Q' to exit.")
prev_time = time.time()
frame_count = 0
prev_gray = None
tracking_box = None
tracking_points = None

# --- Hostility stabilization parameters ---
HOSTILE_HOLD_TIME = 1.5  # seconds to keep HOSTILE after last detection
last_hostile_time = 0
is_hostile = False

# --- Main Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    display = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_count += 1

    # Run YOLO detection every 8 frames
    if frame_count % 8 == 0 or tracking_points is None:
        hostile_now, detections = detect_guns(frame)
        if hostile_now:
            last_hostile_time = time.time()
            is_hostile = True
        elif time.time() - last_hostile_time > HOSTILE_HOLD_TIME:
            is_hostile = False

        if len(detections) > 0:
            x1, y1, x2, y2, conf, cls = detections[0]
            tracking_box = (x1, y1, x2, y2)
            mask = np.zeros_like(gray)
            mask[y1:y2, x1:x2] = 255
            tracking_points = cv2.goodFeaturesToTrack(gray, mask=mask, maxCorners=80, qualityLevel=0.3, minDistance=7)
    else:
        detections = []

    # Optical Flow tracking
    if prev_gray is not None and tracking_points is not None:
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, tracking_points, None)
        good_new = new_points[status == 1]
        good_old = tracking_points[status == 1]
        if len(good_new) > 0 and tracking_box:
            movement = np.mean(good_new - good_old, axis=0)
            x_shift, y_shift = int(movement[0]), int(movement[1])
            x1, y1, x2, y2 = tracking_box
            tracking_box = (x1 + x_shift, y1 + y_shift, x2 + x_shift, y2 + y_shift)

    prev_gray = gray.copy()
    tracking_points = cv2.goodFeaturesToTrack(gray, mask=None, maxCorners=80, qualityLevel=0.3, minDistance=7)

    # Draw detection box with stabilized status
    if tracking_box:
        x1, y1, x2, y2 = tracking_box
        color = (0, 0, 255) if is_hostile else (0, 255, 0)
        label = "HOSTILE" if is_hostile else "FRIENDLY"
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # FPS and HUD
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    draw_hud(display, fps)

    cv2.namedWindow("Hybrid NVG Tracker (Final)", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Hybrid NVG Tracker (Final)", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Hybrid NVG Tracker (Final)", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
