import cv2
import mediapipe as mp
import numpy as np

# Initialize the MediaPipe Face Detection and Drawing tools
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Set up face detection
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect faces
        results = face_detection.process(rgb_frame)

        # If faces are detected, scramble their regions and highlight them
        if results.detections:
            for detection in results.detections:
                # Draw bounding box around the detected face
                mp_drawing.draw_detection(frame, detection)

                # Get the bounding box coordinates
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                # Ensure bounding box coordinates are within the frame bounds
                x = max(0, x)
                y = max(0, y)
                w_box = min(w - x, w_box)
                h_box = min(h - y, h_box)

                # Extract the face region
                face = frame[y:y+h_box, x:x+w_box]

                # Place the scrambled face back into the frame
                frame[y:y+h_box, x:x+w_box] = face
        else:
            # Optionally handle the case where no faces are detected
            pass

        # Display the scrambled frame with the highlighted face
        cv2.imshow('Subject Tracker Mk2', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
