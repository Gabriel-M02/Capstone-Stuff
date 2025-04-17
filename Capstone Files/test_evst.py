import cv2
import os
from ThreatDetectionMk2MAC import detect_guns

# Functional Test Code

def test_weapon_detection_on_sample():
    # Dynamically build the path to your test image (safe across OS)
    img_path = os.path.join("EVST_DataModelPrototypemk1", "train", "images",
                            "WIN_20250410_00_15_46_Pro_jpg.rf.c9c0a4d678a9fff622ea1becfd5409f6.jpg")

    frame = cv2.imread(img_path)
    assert frame is not None, f"Test image not found at: {img_path}"

    # Run detection
    hostile, detections = detect_guns(frame, threshold=0.5)

    # Test pass criteria
    assert hostile is True or len(detections) > 0, "Gun should be detected in test image."
