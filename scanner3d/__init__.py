# 3D Scanner - Capture 3D scan images via DroidCam video recording + frame extraction
# + Cloud 3D reconstruction via Kiri Engine API
# + CPU-optimized background removal + ArUco marker dimension tracking
# + Live ORB feature tracking + point cloud visualization (LiveTracker3D)

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

# Lazy import: LiveTracker3D requires open3d which is optional.
# Import it only when explicitly accessed to avoid breaking the app
# when open3d is not installed.
def _get_live_tracker():
    """Lazy import of LiveTracker3D (requires open3d)."""
    from .scanner import LiveTracker3D
    return LiveTracker3D

__all__ = [
    "Camera",
    "BackgroundRemover",
    "ArUcoDimensionTracker",
    "NUM_ANGLES",
    "OUTPUT_DIR",
    "DROIDCAM_INDEX",
    "KIRI_API_KEY",
    "KIRI_BASE_URL",
]
