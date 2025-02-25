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

        # If faces are detected, scramble their regions
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                # Extract the face region
                face = frame[y:y+h_box, x:x+w_box]

                # Scramble the face by resizing it to a very small size and then scaling it back up
                face = cv2.resize(face, (16, 16), interpolation=cv2.INTER_LINEAR)
                face = cv2.resize(face, (w_box, h_box), interpolation=cv2.INTER_NEAREST)

                # Place the scrambled face back into the frame
                frame[y:y+h_box, x:x+w_box] = face

        # Display the scrambled frame
        cv2.imshow('Person Tracking with Face Scrambling', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
