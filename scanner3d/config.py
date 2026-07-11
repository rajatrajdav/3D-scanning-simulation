"""
Configuration settings for the 3D Scanner project.
"""

import os

# ─── IP WEBCAM SETTINGS ───────────────────────────────────────────────────
# Your phone running IP Webcam app on the same WiFi network
# Download IP Webcam from Play Store: https://play.google.com/store/apps/details?id=com.pas.webcam
# Open the app and tap "Start Server" - it shows the URL
IP_WEBCAM_URL = "http://10.138.159.186:8080"

# The MJPEG video stream path (default for IP Webcam)
IP_WEBCAM_STREAM_PATH = "/video"

# The snapshot/photo path (for taking still images)
IP_WEBCAM_SNAPSHOT_PATH = "/photo.jpg"

# IP Webcam API endpoints
IP_WEBCAM_API_FOCUS = "/focus"
IP_WEBCAM_API_TORCH = "/torch"
IP_WEBCAM_API_SETTINGS = "/settings"
IP_WEBCAM_API_INFO = "/info"

# ─── DROIDCAM SETTINGS (fallback) ─────────────────────────────────────────
# Only used if IP Webcam is not available
DROIDCAM_IP_URL = "http://10.138.159.186:8080"
DROIDCAM_INDEX = -

# Camera source preference
USE_PHONE_CAMERA = True  # Uses IP Webcam or Phone camera

# ─── SCANNING SETTINGS ────────────────────────────────────────────────────
NUM_ANGLES = 70  # Number of photos to extract from video (every ~5°)
CAPTURE_DELAY = 0.2
VIDEO_RECORD_FPS = 60.0
EXTRACT_AFTER_RECORD = True
RESOLUTION = (1280, 720)

# ─── IMAGE PROCESSING ─────────────────────────────────────────────────────
USE_FEATURE_MATCHING = False
MIN_FEATURE_MATCHES = 10

# ─── OUTPUT ───────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "captures")
MODEL_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# ─── KIRI ENGINE API ──────────────────────────────────────────────────────
KIRI_BASE_URL = "https://api.kiriengine.app/api/"
KIRI_API_KEY = "kiri_6ehOvZRTDonIKaEltkltR0xbae2dCmXKPauyG9P3zYA"