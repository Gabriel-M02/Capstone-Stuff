# Separate program for Threat Detection
# Finished Program is to be combined with FaceTracker.py
# Starter File Created by Gabriel M

import cv2
import numpy as np
from ultralytics import YOLO
import psutil
import random
import time

# Load YOLOv8 model for person detection
model = YOLO("yolov8n.pt")
assigned_label = None

# Simulated hostility classifier
def is_hostile(person_id):
    global assigned_label
    if assigned_label is None:
        assigned_label = random.choice([True, False])
    return assigned_label

# Global variable to hold the smoothed bounding box
smoothed_box = None
SMOOTHING_ALPHA = 0.3

def draw_hud(frame, detections):
    global smoothed_box

    if len(detections) == 0:
        return

    # Sort detections by confidence (descending)
    detections = sorted(detections, key=lambda d: d[4], reverse=True)

    # Use only the most confident detection
    x1, y1, x2, y2, conf, cls = detections[0][:6]

    # Create a stable ID (not strictly necessary with just one detection)
    person_id = hash((int(x1), int(y1), int(x2), int(y2)))
    hostile = is_hostile(person_id)

    # Smooth the bounding box
    if smoothed_box is None:
        smoothed_box = (x1, y1, x2, y2)
    else:
        sx1, sy1, sx2, sy2 = smoothed_box
        x1 = SMOOTHING_ALPHA * x1 + (1 - SMOOTHING_ALPHA) * sx1
        y1 = SMOOTHING_ALPHA * y1 + (1 - SMOOTHING_ALPHA) * sy1
        x2 = SMOOTHING_ALPHA * x2 + (1 - SMOOTHING_ALPHA) * sx2
        y2 = SMOOTHING_ALPHA * y2 + (1 - SMOOTHING_ALPHA) * sy2
        smoothed_box = (x1, y1, x2, y2)

    # Draw rectangle and label
    color = (0, 0, 255) if hostile else (0, 255, 0)
    label = "HOSTILE" if hostile else "FRIENDLY"

    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    cv2.putText(frame, label, (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Add HUD time
    cv2.putText(frame, f"Night Vision HUD - {time.strftime('%H:%M:%S')}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

# Display Battery Status
def draw_battery_status(frame, x=540, y=10):
    battery = psutil.sensors_battery()
    if battery is None:
        return  # Not available on this system

    percent = battery.percent
    charging = battery.power_plugged

    # Battery outline
    width = 80
    height = 30
    outline_color = (255, 255, 255)
    fill_color = (0, 255, 0) if percent > 20 else (0, 0, 255)

    # Main battery body
    cv2.rectangle(frame, (x, y), (x + width, y + height), outline_color, 2)

    # Battery tip
    cv2.rectangle(frame, (x + width, y + 8), (x + width + 6, y + height - 8), outline_color, -1)

    # Fill level
    fill_width = int((percent / 100) * (width - 4))
    cv2.rectangle(frame, (x + 2, y + 2), (x + 2 + fill_width, y + height - 2), fill_color, -1)

    # Percentage text
    text = f"{int(percent)}%{' ' if charging else ''}"
    cv2.putText(frame, text, (x - 60, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

# Simulate night vision effect
def apply_night_vision_effect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    colored = cv2.applyColorMap(enhanced, cv2.COLORMAP_SUMMER)
    return colored

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for performance
    small_frame = cv2.resize(frame, (700, 450))

    # Night vision effect
    nv_frame = apply_night_vision_effect(small_frame)

    # YOLO detection
    results = model(nv_frame, classes=[0])  # Class 0 = 'person'
    detections = results[0].boxes.data.cpu().numpy() if results[0].boxes.data.numel() > 0 else []

    draw_battery_status(nv_frame)

    # Draw HUD
    draw_hud(nv_frame, detections)

    # Display
    cv2.imshow("EVST NVG Prototype MK 1", nv_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
