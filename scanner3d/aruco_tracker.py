"""
Marker-Based Dimension Measurement
====================================
Detects ArUco markers to establish a pixels-per-metric ratio, then detects
the actual object near the marker (via contour analysis) and overlays a
bounding box with real-world dimensions (L x W x H) on it.

Requires opencv-contrib-python (cv2.aruco lives in the contrib module):
    pip install opencv-contrib-python numpy

Notes on accuracy:
  - PPM (pixels-per-metric) calibration from the marker is reliable.
  - Width/height are measured from the object's actual silhouette, found
    via Canny edges + contours in the region around the marker (with the
    marker itself masked out so its own square edges aren't mistaken for
    the object). This assumes a reasonably plain/contrasting background,
    same assumption used by most single-camera marker+contour measuring
    tools - it isn't a substitute for stereo or depth-camera input.
  - "Depth" (the 3rd dimension) cannot be measured from a single 2D image
    at all. It is reported as a heuristic estimate (a configurable ratio
    of the measured width) and is clearly not a real measurement - treat
    it as a placeholder until a second view or a depth sensor is added.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List


# Known marker sizes in centimeters
DEFAULT_MARKER_SIZE_CM = 5.0  # 5 cm ArUco marker (common printed size)


class ArUcoDimensionTracker:
    """
    Detects ArUco markers, computes pixels-per-metric ratio, detects the
    actual object near the marker, and overlays measured dimensions.

    Args:
        marker_size_cm: Physical size of the printed ArUco marker in cm.
        marker_id: The ArUco marker ID to track (None = use any detected marker).
        enable: Start with tracking enabled.
        depth_estimate_ratio: Heuristic multiplier used to *estimate* the
            unmeasurable 3rd dimension as a fraction of measured width.
            This is not a real measurement (see module docstring).
        min_object_area_frac: Minimum contour area, as a fraction of frame
            area, to be considered "the object" rather than noise.
    """

    def __init__(self, marker_size_cm: float = DEFAULT_MARKER_SIZE_CM,
                 marker_id: Optional[int] = None, enable: bool = True,
                 depth_estimate_ratio: float = 0.6,
                 min_object_area_frac: float = 0.01):
        if marker_size_cm <= 0:
            raise ValueError("marker_size_cm must be positive")

        self.enabled = enable
        self.marker_size_cm = marker_size_cm
        self.marker_id = marker_id
        self.depth_estimate_ratio = depth_estimate_ratio
        self.min_object_area_frac = min_object_area_frac

        self.ppm: float = 0.0  # pixels-per-metric (pixels per cm)
        self.last_marker_corners: Optional[np.ndarray] = None
        self.last_dimensions: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.last_bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)

        # ArUco dictionary and detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

    # ------------------------------------------------------------------ #
    # Object detection (the part that was previously just a guessed box)
    # ------------------------------------------------------------------ #
    def _detect_object_bbox(self, gray: np.ndarray,
                             marker_corners: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Find the actual object's bounding box via Canny edges + contours,
        searching the whole frame but with the marker's own footprint
        masked out so its square border isn't mistaken for the object.

        Returns:
            (x, y, w, h) of the largest qualifying contour, or None if
            nothing big enough was found.
        """
        h, w = gray.shape[:2]

        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edged = cv2.Canny(blurred, 50, 150)
        edged = cv2.dilate(edged, None, iterations=2)
        edged = cv2.erode(edged, None, iterations=1)

        # Mask out the marker region (plus a small margin) so its own
        # edges can't be picked up as "the object".
        marker_poly = marker_corners.astype(np.int32)
        margin = int(0.15 * np.linalg.norm(marker_corners[0] - marker_corners[2]))
        marker_center = np.mean(marker_corners, axis=0)
        expanded_poly = marker_center + (marker_poly - marker_center) * (
            1.0 + margin / max(np.linalg.norm(marker_corners[0] - marker_center), 1.0)
        )
        cv2.fillPoly(edged, [expanded_poly.astype(np.int32)], 0)

        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        min_area = self.min_object_area_frac * w * h
        candidates = [c for c in contours if cv2.contourArea(c) >= min_area]
        if not candidates:
            return None

        largest = max(candidates, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest)
        return (x, y, bw, bh)

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a frame: detect ArUco marker, compute PPM, detect the
        object near the marker, and draw the bounding box + dimension
        text. Returns the annotated frame.
        """
        if not self.enabled:
            return frame

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            return frame

        # Find the target marker. If a specific marker_id was requested but
        # isn't present in this frame, don't silently fall back to a
        # different marker - skip the frame instead, since using the wrong
        # marker would silently corrupt the PPM calibration.
        target_idx = None
        if self.marker_id is not None:
            for i, marker_id_val in enumerate(ids.flatten()):
                if marker_id_val == self.marker_id:
                    target_idx = i
                    break
            if target_idx is None:
                return frame
        else:
            target_idx = 0

        marker_corners = corners[target_idx][0]  # 4x2 array
        self.last_marker_corners = marker_corners

        # Draw the marker outline
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Compute pixels-per-metric: average of side lengths
        side_lengths = [
            np.linalg.norm(marker_corners[i] - marker_corners[(i + 1) % 4])
            for i in range(4)
        ]
        avg_side_px = np.mean(side_lengths)
        if avg_side_px <= 0:
            return frame
        self.ppm = avg_side_px / self.marker_size_cm

        # Detect the actual object instead of guessing a fixed-size box
        obj_bbox = self._detect_object_bbox(gray, marker_corners)
        if obj_bbox is None:
            # Marker is calibrated but no distinct object was found this
            # frame - don't draw a fabricated box.
            self.last_bbox = None
            self.last_dimensions = (0.0, 0.0, 0.0)
            return frame

        bx, by, bw, bh = obj_bbox
        self.last_bbox = obj_bbox

        # Real-world width/height are measured; depth is an explicit estimate.
        real_w = bw / self.ppm
        real_h = bh / self.ppm
        real_d = real_w * self.depth_estimate_ratio
        self.last_dimensions = (round(real_w, 1), round(real_h, 1), round(real_d, 1))

        # Draw bounding box
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)

        # Draw dimension text (clamped so the label stays on-screen even
        # when the object touches the top edge of the frame)
        dim_text = f"{self.last_dimensions[0]} x {self.last_dimensions[1]} x {self.last_dimensions[2]} cm (~ est.)"
        label_y_top = max(by - 32, 0)
        label_y_bottom = max(by, label_y_top + 22)
        cv2.rectangle(frame, (bx, label_y_top), (bx + bw, label_y_bottom), (0, 0, 0), -1)
        cv2.putText(frame, dim_text, (bx + 4, label_y_bottom - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw PPM info just below the box (clamped to stay in-frame)
        ppm_text = f"PPM: {self.ppm:.1f} px/cm"
        ppm_y = min(by + bh + 18, h - 5)
        cv2.putText(frame, ppm_text, (bx + 4, ppm_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)

        return frame

    def get_ppm(self) -> float:
        """Get the current pixels-per-metric ratio."""
        return self.ppm

    def get_dimensions(self) -> Tuple[float, float, float]:
        """Get the last measured dimensions (L, W, H) in cm. The 3rd value
        is a heuristic estimate, not a measurement - see module docstring."""
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