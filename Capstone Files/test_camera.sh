#!/bin/bash

LOGFILE="camera_test_log.txt"
DEVICE="/dev/video0"
RESOLUTION="1280x720"
FRAMERATE="30"

echo "===== CAMERA TEST SCRIPT =====" | tee $LOGFILE
echo "Date: $(date)" | tee -a $LOGFILE
echo "" | tee -a $LOGFILE

# Step 1: Check if camera device exists
if [ ! -e "$DEVICE" ]; then
    echo "[ERROR] $DEVICE not found." | tee -a $LOGFILE
    exit 1
fi
echo "[INFO] Camera detected at $DEVICE" | tee -a $LOGFILE

# Step 2: Show camera format capabilities
echo "" | tee -a $LOGFILE
echo "[INFO] Listing supported formats:" | tee -a $LOGFILE
v4l2-ctl --device=$DEVICE --list-formats-ext | tee -a $LOGFILE

# Step 3: Preview video using ffplay (non-blocking)
echo "" | tee -a $LOGFILE
echo "[INFO] Attempting live preview with ffplay..." | tee -a $LOGFILE
ffplay -f v4l2 -framerate $FRAMERATE -video_size $RESOLUTION -i $DEVICE &>/dev/null &

# Wait a few seconds to let preview show up
sleep 5

# Step 4: Capture image using ffmpeg
echo "" | tee -a $LOGFILE
echo "[INFO] Capturing test image..." | tee -a $LOGFILE
ffmpeg -f v4l2 -framerate $FRAMERATE -video_size $RESOLUTION -i $DEVICE -frames 1 test_capture.jpg -y &>> $LOGFILE

if [ -f "test_capture.jpg" ]; then
    echo "[SUCCESS] Image captured: test_capture.jpg" | tee -a $LOGFILE
else
    echo "[ERROR] Failed to capture image" | tee -a $LOGFILE
fi

echo "" | tee -a $LOGFILE
echo "[DONE] Camera test complete. See $LOGFILE for details." | tee -a $LOGFILE
