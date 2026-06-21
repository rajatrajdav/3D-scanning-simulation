# 3D Scanner - Capture 3D scan images via DroidCam video recording + frame extraction
# + Cloud 3D reconstruction via Kiri Engine API
# + CPU-optimized background removal + ArUco marker dimension tracking

from .scanner import Scanner3D
from .camera import Camera
from .config import (
    NUM_ANGLES,
    OUTPUT_DIR,
    DROIDCAM_INDEX,
    KIRI_API_KEY,
    KIRI_BASE_URL,
)
from .segmentation import BackgroundRemover
from .aruco_tracker import ArUcoDimensionTracker

__all__ = [
    "Scanner3D",
    "Camera",
    "BackgroundRemover",
    "ArUcoDimensionTracker",
    "NUM_ANGLES",
    "OUTPUT_DIR",
    "DROIDCAM_INDEX",
    "KIRI_API_KEY",
    "KIRI_BASE_URL",
]
