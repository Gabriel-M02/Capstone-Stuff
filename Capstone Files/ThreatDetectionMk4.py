import cv2
import numpy as np
import psutil
import time
from ultralytics import YOLO

# Load the custom YOLOv8 model
model = YOLO("EVST_DataModelPrototypemk1/runs/detect/train/weights/best.pt")

# Battery HUD overlay - Jacob O'Hearon
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

# Gun and person detection function
def detect_guns_and_people(frame, threshold=0.8):
    results = model(frame)
    detections = results[0].boxes.data.cpu().numpy() if results[0].boxes.data.numel() > 0 else []
    person_box = None
    gun_detected = False

    for det in detections:
        x1, y1, x2, y2, conf, cls = det[:6]
        cls = int(cls)
        if cls == 0 and conf >= threshold:
            gun_detected = True
        elif cls == 1 and conf >= 0.5:  # Class 1 = person
            person_box = (int(x1), int(y1), int(x2), int(y2))

    return gun_detected, person_box, detections

# Initialize camera and variables
cap = cv2.VideoCapture(0)
smoothed_bbox = None
prev_time = time.time()

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (700, 450))
    display_frame = frame.copy()

    # Detection step
    gun, person_box, detections = detect_guns_and_people(frame)

    # Bounding box smoothing
    if person_box:
        x1, y1, x2, y2 = person_box
        if smoothed_bbox is None:
            smoothed_bbox = person_box
        else:
            sx1, sy1, sx2, sy2 = smoothed_bbox
            alpha = 0.4
            x1 = int(alpha * x1 + (1 - alpha) * sx1)
            y1 = int(alpha * y1 + (1 - alpha) * sy1)
            x2 = int(alpha * x2 + (1 - alpha) * sx2)
            y2 = int(alpha * y2 + (1 - alpha) * sy2)
            smoothed_bbox = (x1, y1, x2, y2)

    # Display threat classification
    if smoothed_bbox:
        x1, y1, x2, y2 = smoothed_bbox
        color = (0, 0, 255) if gun else (0, 255, 0)
        label = "HOSTILE" if gun else "FRIENDLY"
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        cv2.putText(display_frame, "NO PERSON DETECTED", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Optional: draw all YOLO detections (debug)
    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 255, 0), 1)

    # Battery and HUD overlays
    draw_battery_status(display_frame)
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(display_frame, f"FPS: {int(fps)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(display_frame, f"EVST NVG HUD MK 2 - {time.strftime('%H:%M:%S')}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show final display
    cv2.imshow("Hybrid NVG Tracker", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
