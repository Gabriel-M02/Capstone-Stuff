import time
import cv2
import os
from ThreatDetectionMk2MAC import detect_guns


# Non functional Tester Code
def test_detection_speed_under_100ms():
    # Use the same test image as your functional test
    img_path = os.path.join("EVST_DataModelPrototypemk1", "train", "images",
                            "WIN_20250410_00_15_46_Pro_jpg.rf.c9c0a4d678a9fff622ea1becfd5409f6.jpg")

    frame = cv2.imread(img_path)
    assert frame is not None, f"Image not found at path: {img_path}"

    # Time the detection
    start = time.time()
    detect_guns(frame)
    duration = time.time() - start

    # Test passes if inference takes less than 100ms
    assert duration < 0.1, f"Detection took too long: {duration:.3f}s"