import cv2
import mediapipe as mp
import numpy as np

# EVST Tactical Solutions Threat Sensor and Tracking Program
# Full Body Tracking Code Created by Gabriel Montemayor
# Optimized for Stability and Accuracy

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

# Initialize OpenCV webcam capture
cap = cv2.VideoCapture(0)

# Initialize bounding box variables for smoothing
smoothed_x_min, smoothed_y_min = None, None
smoothed_x_max, smoothed_y_max = None, None
alpha = 0.2  # Smoothing factor for EMA

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Pose Estimation
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        h, w, _ = frame.shape  # Get frame dimensions
        x_min, y_min = w, h  # Initialize min values
        x_max, y_max = 0, 0  # Initialize max values
        confidence_threshold = 0.5  # Confidence filter

        # Loop through detected keypoints
        for landmark in results.pose_landmarks.landmark:
            if landmark.visibility > confidence_threshold:  # Filter low-confidence points
                x, y = int(landmark.x * w), int(landmark.y * h)

                # Expand bounding box
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

        # Ensure bounding box is within frame limits
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)

        # Add margin to the bounding box for better framing
        padding = 20  # Pixels
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        # Apply Exponential Moving Average (EMA) smoothing
        if smoothed_x_min is None:
            smoothed_x_min, smoothed_y_min = x_min, y_min
            smoothed_x_max, smoothed_y_max = x_max, y_max
        else:
            smoothed_x_min = alpha * x_min + (1 - alpha) * smoothed_x_min
            smoothed_y_min = alpha * y_min + (1 - alpha) * smoothed_y_min
            smoothed_x_max = alpha * x_max + (1 - alpha) * smoothed_x_max
            smoothed_y_max = alpha * y_max + (1 - alpha) * smoothed_y_max

        # Convert floating point values back to integers
        x_min, y_min, x_max, y_max = map(int, [smoothed_x_min, smoothed_y_min, smoothed_x_max, smoothed_y_max])

        # Draw a stable rectangle around the detected person
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (140, 255, 0), 2)

    # Display the frame
    cv2.imshow('Stable Full Body Tracker - OID', frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
