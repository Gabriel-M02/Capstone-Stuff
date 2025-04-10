import cv2
import numpy as np
import psutil
import time
from ultralytics import YOLO
import mediapipe as mp

# Load custom YOLOv8 model trained to detect guns
model = YOLO("NerfPrototype/runs/detect/train3/weights/best.pt")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

# Battery HUD
def draw_battery_status(frame, x=540, y=10):
    battery = psutil.sensors_battery()
    if battery is None:
        return
    percent = battery.percent
    charging = battery.power_plugged
    width, height = 80, 30
    outline_color = (255, 255, 255)
    fill_color = (0, 255, 0) if percent > 20 else (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x + width, y + height), outline_color, 2)
    cv2.rectangle(frame, (x + width, y + 8), (x + width + 6, y + height - 8), outline_color, -1)
    fill_width = int((percent / 100) * (width - 4))
    cv2.rectangle(frame, (x + 2, y + 2), (x + 2 + fill_width, y + height - 2), fill_color, -1)
    text = f"{int(percent)}%{' ' if charging else ''}"
    cv2.putText(frame, text, (x - 60, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

# Night vision effect
def apply_night_vision_effect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    colored = cv2.applyColorMap(enhanced, cv2.COLORMAP_OCEAN)
    return colored

def detect_guns(frame, threshold=0.8):
    results = model(frame)
    detections = results[0].boxes.data.cpu().numpy() if results[0].boxes.data.numel() > 0 else []
    
    if len(detections) == 0:
        print("[DEBUG] No detections")
        return False, []  # No objects at all

    for det in detections:
        cls = int(det[5])
        conf = float(det[4])
        print(f"[DEBUG] Class: {cls}, Confidence: {conf:.2f}")
        if cls == 0 and conf >= threshold:  # Class 0 = gun
            return True, detections

    return False, detections  # Detections present, but no valid guns


# Initialize capture
cap = cv2.VideoCapture(0)
smoothed_bbox = None
tracking_with_pose = False
last_pose_timestamp = time.time()
prev_time = time.time()

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (700, 450))
    nv_frame = apply_night_vision_effect(frame.copy())
    h, w, _ = frame.shape
    person_box = None

    # Pose detection
    pose_results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if pose_results.pose_landmarks:
        tracking_with_pose = True
        last_pose_timestamp = time.time()

        x_min, y_min = w, h
        x_max, y_max = 0, 0
        for landmark in pose_results.pose_landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)

        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        person_box = (x_min, y_min, x_max, y_max)

    # Use last box if pose unavailable for 1+ sec
    elif time.time() - last_pose_timestamp > 1.0:
        tracking_with_pose = False
        person_box = smoothed_bbox

    # Smooth tracking box
    if person_box:
        x1, y1, x2, y2 = person_box
        if smoothed_bbox is None:
            smoothed_bbox = (x1, y1, x2, y2)
        else:
            sx1, sy1, sx2, sy2 = smoothed_bbox
            alpha = 0.4
            x1 = int(alpha * x1 + (1 - alpha) * sx1)
            y1 = int(alpha * y1 + (1 - alpha) * sy1)
            x2 = int(alpha * x2 + (1 - alpha) * sx2)
            y2 = int(alpha * y2 + (1 - alpha) * sy2)
            smoothed_bbox = (x1, y1, x2, y2)

    # Run gun detection
    hostile, detections = detect_guns(frame)

    # Draw rectangle + label around tracked subject
    if smoothed_bbox:
        x1, y1, x2, y2 = smoothed_bbox
        color = (0, 0, 255) if hostile else (0, 255, 0)
        label = "HOSTILE" if hostile else "FRIENDLY"
        cv2.rectangle(nv_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(nv_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Optional debug: draw YOLO boxes
    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])
        cv2.rectangle(nv_frame, (x1, y1), (x2, y2), (255, 255, 0), 1)

    # HUD overlays
    draw_battery_status(nv_frame)
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(nv_frame, f"FPS: {int(fps)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(nv_frame, f"EVST NVG HUD MK 2 - {time.strftime('%H:%M:%S')}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Display frame
    cv2.imshow("Hybrid NVG Tracker", nv_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
