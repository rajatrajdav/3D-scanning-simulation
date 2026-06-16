"""
Configuration settings for the 3D Scanner project.
"""

import os

# DroidCam settings
# The IP shown in DroidCam phone app (for reference / web remote control)
DROIDCAM_IP_URL = "http://10.141.200.242:4747"

# DroidCam PC Client creates a virtual camera on your system.
# Auto-detect by trying indices 0, 1, 2, 3, 4 in sequence.
# Or set a specific index manually (0 = built-in webcam, 1+ = DroidCam)
DROIDCAM_INDEX = -1  # -1 = auto-detect, or set to 0, 1, 2 etc.

# Scanning settings
# HIGH QUALITY: 70 angles = every ~5 degrees for better reconstruction
NUM_ANGLES = 70  # Number of photos to extract from video
CAPTURE_DELAY = 0.2  # Seconds between captures (legacy, not used in video mode)

# Video recording settings
VIDEO_RECORD_FPS = 60.0  # FPS for recording preview
EXTRACT_AFTER_RECORD = True  # Extract frames from recorded video after recording

# HIGH QUALITY: Higher resolution for better feature matching
RESOLUTION = (1280, 720)  # HD resolution (was 640x480)

# Image processing
USE_FEATURE_MATCHING = False
MIN_FEATURE_MATCHES = 10

# Output
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "captures")
MODEL_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# Kiri Engine API settings for cloud 3D reconstruction
# Sign up at https://kiriengine.app/ to get your API key
KIRI_BASE_URL = "https://api.kiriengine.app/api/"
KIRI_API_KEY = "kiri_fY_P0PneyULXY8u0TDrf-Lf6o5iy0-1DgkK9Ad430rs"