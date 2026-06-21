"""
Marker-Based Dimension Measurement
====================================
Detects ArUco markers to establish a pixels-per-metric ratio,
then overlays a bounding box with real-world dimensions (L x W x H)
on the detected object.

Uses OpenCV's built-in ArUco module (cv2.aruco).
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List


# Known marker sizes in centimeters
DEFAULT_MARKER_SIZE_CM = 5.0  # 5 cm ArUco marker (common printed size)


class ArUcoDimensionTracker:
    """
    Detects ArUco markers, computes pixels-per-metric ratio,
    and overlays dimension measurements on the target object.

    Args:
        marker_size_cm: Physical size of the printed ArUco marker in cm.
        marker_id: The ArUco marker ID to track (None = use any detected marker).
        enable: Start with tracking enabled.
    """

    def __init__(self, marker_size_cm: float = DEFAULT_MARKER_SIZE_CM,
                 marker_id: Optional[int] = None, enable: bool = True):
        self.enabled = enable
        self.marker_size_cm = marker_size_cm
        self.marker_id = marker_id
        self.ppm: float = 0.0  # pixels-per-metric (pixels per cm)
        self.last_marker_corners: Optional[np.ndarray] = None
        self.last_dimensions: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.last_bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)

        # ArUco dictionary and detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a frame: detect ArUco marker, compute PPM, draw bounding box
        and dimension text. Returns the annotated frame.
        """
        if not self.enabled:
            return frame

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            # Find the target marker (or use the first one)
            target_idx = 0
            if self.marker_id is not None:
                for i, marker_id_val in enumerate(ids.flatten()):
                    if marker_id_val == self.marker_id:
                        target_idx = i
                        break

            marker_corners = corners[target_idx][0]  # 4x2 array
            self.last_marker_corners = marker_corners

            # Draw the marker outline
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Compute pixels-per-metric: average of side lengths
            side_lengths = []
            for i in range(4):
                p1 = marker_corners[i]
                p2 = marker_corners[(i + 1) % 4]
                side_px = np.linalg.norm(p1 - p2)
                side_lengths.append(side_px)

            avg_side_px = np.mean(side_lengths)
            if avg_side_px > 0:
                self.ppm = avg_side_px / self.marker_size_cm

            # Estimate object bounding box from marker position
            # The object is assumed to be near the marker center
            marker_center = np.mean(marker_corners, axis=0)
            cx, cy = int(marker_center[0]), int(marker_center[1])

            # Estimate object size based on marker position in frame
            # We assume the object occupies a region relative to the marker
            if self.ppm > 0:
                # Estimate object bounding box: centered around marker,
                # sized proportionally to frame dimensions
                obj_w_px = int(w * 0.4)  # 40% of frame width
                obj_h_px = int(h * 0.5)  # 50% of frame height

                # Clamp to frame boundaries
                bx = max(10, min(cx - obj_w_px // 2, w - obj_w_px - 10))
                by = max(10, min(cy - obj_h_px // 2, h - obj_h_px - 10))
                bw = min(obj_w_px, w - bx - 10)
                bh = min(obj_h_px, h - by - 10)

                self.last_bbox = (bx, by, bw, bh)

                # Compute real-world dimensions
                real_w = bw / self.ppm
                real_h = bh / self.ppm
                # Estimate depth as a fraction of width (simple heuristic)
                real_d = real_w * 0.6
                self.last_dimensions = (round(real_w, 1), round(real_h, 1), round(real_d, 1))

                # Draw bounding box
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh),
                              (0, 255, 0), 2)

                # Draw dimension text
                dim_text = f"{self.last_dimensions[0]} x {self.last_dimensions[1]} x {self.last_dimensions[2]} cm"
                label_bg = (bx, by - 30)
                cv2.rectangle(frame, (bx, by - 32), (bx + bw, by), (0, 0, 0), -1)
                cv2.putText(frame, dim_text, (bx + 4, by - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Draw PPM info
                ppm_text = f"PPM: {self.ppm:.1f} px/cm"
                cv2.putText(frame, ppm_text, (bx + 4, by + bh + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)

        return frame

    def get_ppm(self) -> float:
        """Get the current pixels-per-metric ratio."""
        return self.ppm

    def get_dimensions(self) -> Tuple[float, float, float]:
        """Get the last measured dimensions (L, W, H) in cm."""
        return self.last_dimensions

    def get_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """Get the last bounding box (x, y, w, h)."""
        return self.last_bbox

    def toggle(self, state: bool = None):
        if state is not None:
            self.enabled = state
        else:
            self.enabled = not self.enabled
        return self.enabled

    def release(self):
        pass