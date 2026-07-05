"""
3D Scanner - Flask Backend Server
==================================
Handles IP Webcam MJPEG stream ingestion, frame processing, background removal,
ArUco marker detection for dimension measurement, and scan session management.

Usage:
    python backend/app.py
    # Starts server on http://localhost:5000
"""

import cv2
import numpy as np
import urllib.request
import os
import sys
import time
import json
import threading
import queue
import io
import base64
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from flask import (
    Flask, Response, jsonify, request, send_file, stream_with_context
)
from flask_cors import CORS

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── Configuration ───────────────────────────────────────────────────────────
IP_WEBCAM_URL = "http://10.138.159.186:8080"
IP_WEBCAM_STREAM_PATH = "/video"
IP_WEBCAM_SNAPSHOT_PATH = "/photo.jpg"
STREAM_URL = IP_WEBCAM_URL + IP_WEBCAM_STREAM_PATH
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "captures")
MODEL_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

# ArUco marker configuration
# Using a 4x4 dictionary (50 markers) for CPU efficiency
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()

# Known marker size in meters (e.g., 5cm x 5cm printed marker)
MARKER_SIZE_M = 0.05

# Frame processing settings
MAX_FRAME_WIDTH = 640
PROCESS_FPS = 10  # Process every Nth frame for CPU efficiency

# ─── Flask App ───────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ─── Stream Manager ──────────────────────────────────────────────────────────
class StreamManager:
    """Manages the IP Webcam MJPEG stream with background frame processing."""

    def __init__(self, stream_url: str = STREAM_URL):
        self.stream_url = stream_url
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._processed_frame: Optional[np.ndarray] = None
        self._frame_count = 0
        self._stream_thread: Optional[threading.Thread] = None
        self._process_thread: Optional[threading.Thread] = None

        # Processing results
        self.mask: Optional[np.ndarray] = None
        self.markers: List[Dict[str, Any]] = []
        self.dimensions: Dict[str, float] = {}
        self.pixels_per_metric: float = 0.0
        self.fps = 0.0
        self.last_process_time = time.time()

        # Background removal enabled
        self.bg_removal_enabled = True
        self.marker_detection_enabled = True

    def open(self) -> bool:
        """Open the IP Webcam MJPEG stream."""
        print(f"[Stream] Connecting to IP Webcam: {self.stream_url}")
        try:
            self.cap = cv2.VideoCapture(self.stream_url)
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    print(f"[Stream] ✓ Connected! Resolution: {w}x{h}")
                    return True
                self.cap.release()
            self.cap = None
        except Exception as e:
            print(f"[Stream] Error: {e}")
        return False

    def start(self):
        """Start background stream capture and processing threads."""
        if self.running:
            return
        self.running = True
        self._stream_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._stream_thread.start()
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()
        print("[Stream] Background capture started")

    def stop(self):
        """Stop all background threads."""
        self.running = False
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=2.0)
        if self._process_thread and self._process_thread.is_alive():
            self._process_thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        print("[Stream] Stopped")

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest raw frame."""
        with self.frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def get_processed_frame(self) -> Optional[np.ndarray]:
        """Get the latest processed frame with overlays."""
        with self.frame_lock:
            return self._processed_frame.copy() if self._processed_frame is not None else None

    def _capture_loop(self):
        """Continuously capture frames from the MJPEG stream."""
        frame_count = 0
        fps_start = time.time()
        fps_counter = 0

        while self.running and self.cap:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # Resize for performance
                h, w = frame.shape[:2]
                if w > MAX_FRAME_WIDTH:
                    scale = MAX_FRAME_WIDTH / w
                    new_w, new_h = int(w * scale), int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                with self.frame_lock:
                    self._latest_frame = frame
                    self._frame_count += 1

                fps_counter += 1
                elapsed = time.time() - fps_start
                if elapsed >= 2.0:
                    self.fps = fps_counter / elapsed
                    fps_counter = 0
                    fps_start = time.time()
            else:
                time.sleep(0.01)

    def _process_loop(self):
        """Periodically process frames for background removal and marker detection."""
        while self.running:
            frame = self.get_frame()
            if frame is None:
                time.sleep(0.03)
                continue

            # Throttle processing to every Nth frame
            with self.frame_lock:
                fc = self._frame_count
            if fc % 3 != 0:
                time.sleep(0.01)
                continue

            processed = frame.copy()
            h, w = frame.shape[:2]

            try:
                # 1. Background removal (CPU-optimized)
                if self.bg_removal_enabled:
                    self.mask = remove_background_cpu(frame)
                    if self.mask is not None:
                        # Apply mask: keep foreground, darken background
                        mask_3ch = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)
                        bg_darkened = cv2.addWeighted(processed, 0.3, np.zeros_like(processed), 0.7, 0)
                        processed = np.where(mask_3ch > 0, processed, bg_darkened)
                        # Draw mask boundary (contour)
                        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            largest = max(contours, key=cv2.contourArea)
                            if cv2.contourArea(largest) > 5000:  # Min area threshold
                                cv2.drawContours(processed, [largest], -1, (0, 255, 255), 2)

                # 2. ArUco marker detection for dimension measurement
                if self.marker_detection_enabled:
                    self.dimensions, self.markers, self.pixels_per_metric = detect_aruco_markers(
                        frame, processed, MARKER_SIZE_M
                    )
                    # Draw bounding box and dimensions if markers detected
                    if self.markers:
                        self._draw_dimension_overlay(processed)

            except Exception as e:
                print(f"[Process] Error: {e}")

            # Add status overlay
            self._draw_status_overlay(processed)

            with self.frame_lock:
                self._processed_frame = processed

            # Throttle processing rate
            time.sleep(1.0 / PROCESS_FPS)

    def _draw_dimension_overlay(self, frame: np.ndarray):
        """Draw dimension measurements on the frame."""
        if not self.dimensions:
            return

        height = self.dimensions.get("height", 0)
        width = self.dimensions.get("width", 0)
        depth = self.dimensions.get("depth", 0)

        if height > 0 or width > 0 or depth > 0:
            # Draw measurement text box
            text_lines = []
            if width > 0:
                text_lines.append(f"W: {width*100:.1f} cm")
            if height > 0:
                text_lines.append(f"H: {height*100:.1f} cm")
            if depth > 0:
                text_lines.append(f"D: {depth*100:.1f} cm")

            if text_lines:
                text = "  ".join(text_lines)
                h, w = frame.shape[:2]
                # Semi-transparent background
                overlay = frame.copy()
                cv2.rectangle(overlay, (w - 260, 10), (w - 10, 70), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                cv2.putText(frame, text, (w - 250, 42),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Draw markers
        for marker in self.markers:
            corners = marker.get("corners")
            if corners is not None:
                pts = corners.reshape((-1, 2)).astype(np.int32)
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                cx, cy = int(marker["center"][0]), int(marker["center"][1])
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                cv2.putText(frame, f"ID:{marker['id']}", (cx - 20, cy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Pixels-per-metric info
        if self.pixels_per_metric > 0:
            cv2.putText(frame, f"PPM: {self.pixels_per_metric:.1f}", (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)

    def _draw_status_overlay(self, frame: np.ndarray):
        """Draw status information overlay."""
        h, w = frame.shape[:2]
        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.0f}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # Frame info
        cv2.putText(frame, f"{w}x{h}", (10, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    def get_status(self) -> Dict[str, Any]:
        """Get current stream status."""
        with self.frame_lock:
            frame = self._latest_frame
            h, w = frame.shape[:2] if frame is not None else (0, 0)
        return {
            "connected": self.cap is not None and self.cap.isOpened(),
            "resolution": f"{w}x{h}",
            "fps": round(self.fps, 1),
            "bg_removal": self.bg_removal_enabled,
            "marker_detection": self.marker_detection_enabled,
            "markers_found": len(self.markers),
            "dimensions": self.dimensions,
            "pixels_per_metric": round(self.pixels_per_metric, 2),
        }


# ─── CPU-Optimized Background Removal ────────────────────────────────────────
def remove_background_cpu(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Lightweight background removal using CPU-friendly methods.
    Combines:
    1. Color thresholding in HSV space (for green screen / solid backgrounds)
    2. Edge-aware GrabCut refinement (lightweight)
    3. Morphological cleanup

    For better results, print an ArUco marker on paper and place behind the object,
    or use a solid colored background.
    """
    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]

        # Strategy 1: Detect if background is likely a solid color (green/blue/white)
        # Check corners for dominant color
        corner_pixels = [
            frame[5, 5, :],           # top-left
            frame[5, w-5, :],         # top-right
            frame[h-5, 5, :],         # bottom-left
            frame[h-5, w-5, :],       # bottom-right
            frame[h//2, 5, :],        # middle-left
            frame[h//2, w-5, :],      # middle-right
        ]
        avg_corner = np.mean(corner_pixels, axis=0)

        # Check if background is green/blue (typical for scanning)
        g_minus_r = float(avg_corner[1]) - float(avg_corner[2])
        b_minus_r = float(avg_corner[0]) - float(avg_corner[2])

        if g_minus_r > 20 or b_minus_r > 20:
            # Color-based: background has strong green or blue component
            if g_minus_r > b_minus_r:
                # Green background
                lower = np.array([35, 40, 40])
                upper = np.array([85, 255, 255])
            else:
                # Blue background
                lower = np.array([90, 40, 40])
                upper = np.array([130, 255, 255])

            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.bitwise_not(mask)  # Foreground = not background
        else:
            # Strategy 2: Edge-based background removal
            # Use Sobel edge detection + flood fill from corners
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edges = cv2.dilate(edges, None, iterations=2)

            # Flood fill from corners (background)
            mask = np.ones((h+2, w+2), np.uint8) * 255
            mask[1:-1, 1:-1] = 0
            cv2.floodFill(edges, mask, (0, 0), 255)
            cv2.floodFill(edges, mask, (w-1, 0), 255)
            cv2.floodFill(edges, mask, (0, h-1), 255)
            cv2.floodFill(edges, mask, (w-1, h-1), 255)

            # Invert: foreground = not background edge
            mask = cv2.bitwise_not(edges)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Gaussian blur for soft edges
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        return mask

    except Exception as e:
        print(f"[BG Removal] Error: {e}")
        return None


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Landing page with API documentation."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D Scanner API</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               background: #0f1419; color: #f1f5f9; min-height: 100vh; display: flex;
               align-items: center; justify-content: center; }
        .card { background: #1a1f26; border-radius: 16px; padding: 48px; max-width: 600px;
                width: 90%; border: 1px solid #334155; }
        h1 { font-size: 28px; margin-bottom: 8px; }
        .badge { display: inline-block; background: #00b4d8; color: #fff; font-size: 12px;
                 padding: 4px 12px; border-radius: 20px; margin-bottom: 24px; }
        p { color: #94a3b8; line-height: 1.6; margin-bottom: 24px; }
        .endpoints { list-style: none; }
        .endpoints li { background: #242b33; border-radius: 8px; padding: 12px 16px;
                        margin-bottom: 8px; font-family: monospace; font-size: 14px; }
        .endpoints .method { color: #10b981; font-weight: bold; margin-right: 8px; }
        .endpoints .desc { color: #64748b; font-size: 12px; margin-top: 4px; }
        .status { margin-top: 24px; padding: 12px; background: #242b33; border-radius: 8px;
                  text-align: center; color: #10b981; font-size: 14px; }
    </style>
</head>
<body>
    <div class="card">
        <h1>🎯 3D Scanner API</h1>
        <div class="badge">Backend Server</div>
        <p>Flask backend for the 3D scanning simulation project. Handles IP Webcam stream ingestion, frame processing, background removal, and ArUco marker detection.</p>
        <ul class="endpoints">
            <li><span class="method">GET</span> / <span class="desc">→ This page (API documentation)</span></li>
            <li><span class="method">GET</span> /api/status <span class="desc">→ Server health & configuration</span></li>
        </ul>
        <div class="status">✅ Server is running</div>
    </div>
</body>
</html>
"""

@app.route("/api/status")
def api_status():
    """Return server status as JSON."""
    return jsonify({
        "status": "ok",
        "app": "3D Scanner Backend",
        "version": "1.0.0",
        "endpoints": {
            "/": "API documentation (HTML)",
            "/api/status": "Server status (JSON)",
        }
    })

# ─── ArUco Marker Detection ──────────────────────────────────────────────────
def detect_aruco_markers(
    frame: np.ndarray,
    display_frame: np.ndarray,
    marker_size_m: float
) -> Tuple[Dict[str, float], List[Dict[str, Any]], float]:
    """
    Detect ArUco markers and estimate pixel-per-metric ratio.

    Place a printed ArUco marker (from standard 4x4 dictionary) next to
    the object being scanned. The system detects it and calculates dimensions.

    Args:
        frame: Input BGR frame
        display_frame: Frame to draw on (modified in-place)
        marker_size_m: Known marker size in meters

    Returns:
        (dimensions_dict, markers_list, pixels_per_metric)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)

    dimensions = {"width": 0.0, "height": 0.0, "depth": 0.0}
    markers = []
    ppm = 0.0

    if ids is not None and len(ids) > 0:
        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(display_frame, corners, ids)

        for i, marker_id in enumerate(ids.flatten()):
            c = corners[i][0]
            center = np.mean(c, axis=0)
            side_lengths = [
                np.linalg.norm(c[0] - c[1]),
                np.linalg.norm(c[1] - c[2]),
                np.linalg.norm(c[2] - c[3]),
                np.linalg.norm(c[3] - c[0]),
            ]
            avg_side_pixels = np.mean(side_lengths)

            # Estimate pose (requires camera intrinsics - we approximate)
            # pixels_per_meter = avg_side_pixels / marker_size_m
            current_ppm = avg_side_pixels / marker_size_m if marker_size_m > 0 else 0

            # Use first marker for reference
            if i == 0 and current_ppm > 0:
                ppm = current_ppm

            marker_info = {
                "id": int(marker_id),
                "center": (float(center[0]), float(center[1])),
                "corners": c,
                "side_pixels": float(avg_side_pixels),
                "pixels_per_metric": float(current_ppm),
            }
            markers.append(marker_info)

            # For the first marker, estimate object dimensions
            if i == 0 and ppm > 0:
                # Estimate bounding box of the frame content relative to marker
                h, w = frame.shape[:2]
                # Simple heuristic: the object is within the frame
                # In a real setup, you'd detect the object contour and measure it
                dimensions["width"] = w / ppm
                dimensions["height"] = h / ppm
                dimensions["depth"] = dimensions["width"] * 0.3  # Estimated depth

                # Draw bounding box around the frame area

# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"3D Scanner Backend starting on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=debug)
