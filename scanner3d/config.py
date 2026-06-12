"""
Configuration settings for the 3D Scanner project.
"""

import os

# Camera Mode: "droidcam" (local webcam via DroidCam) or "ipwebcam" (IP Webcam app)
CAMERA_MODE = "droidcam"

# DroidCam settings (local webcam)
# DroidCam typically appears at index 0, 1, or 2 depending on your system.
# If index 0 is your built-in webcam, try index 1 or 2 for DroidCam.
# Run `python main.py --list-cameras` to find the correct index.
DROIDCAM_INDEX = 1  # Usually 0 or 1 for DroidCam virtual camera

# IP Webcam settings (alternative)
CAMERA_URL = "http://10.18.155.15:8080"
CAMERA_SOURCE = f"{CAMERA_URL}/video"  # MJPEG video stream
SNAPSHOT_URL = f"{CAMERA_URL}/shot.jpg"  # Snapshot endpoint

# Scanning settings
NUM_ANGLES = 50  # Number of photos to extract from video (default: every 10 degrees)
CAPTURE_DELAY = 0.5  # Seconds between captures (legacy, not used in video mode)

# Video recording settings
VIDEO_RECORD_FPS = 30.0  # FPS for recording preview
EXTRACT_AFTER_RECORD = True  # Extract frames from recorded video after recording

# Resolution
RESOLUTION = (640, 480)  # Capture resolution

# Image processing
USE_FEATURE_MATCHING = False
MIN_FEATURE_MATCHES = 10

# Output
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "captures")
MODEL_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")