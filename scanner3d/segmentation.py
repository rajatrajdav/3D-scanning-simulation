"""
Object Segmentation Module
==========================
CPU-optimized object segmentation using contour detection + GrabCut.
Works on ANY object (not just humans), suitable for 3D scanning.
Uses edge detection + morphological operations to find object boundaries.

Pipeline:
  1. Convert frame to grayscale
  2. Apply Gaussian blur + Canny edge detection
  3. Morphological close to fill gaps
  4. Find largest contour = object boundary
  5. Optionally apply GrabCut for refinement
  6. Return mask for grid overlay
"""

import cv2
import numpy as np
from typing import Optional


class BackgroundRemover:
    """
    Extracts foreground object from background using contour detection.
    Works on any object type - identifies the dominant object in frame.

    Args:
        enable: Start with removal enabled.
        min_area_ratio: Minimum object area relative to frame (0-1)
        use_grabcut: Use GrabCut refinement for better edges (slower)
    """

    def __init__(self, enable: bool = True, min_area_ratio: float = 0.02,
                 use_grabcut: bool = False):
        self.enabled = enable
        self.min_area_ratio = min_area_ratio
        self.use_grabcut = use_grabcut
        self._last_mask: Optional[np.ndarray] = None
        self._last_contour: Optional[np.ndarray] = None
        self._bg_color = (0, 0, 0)
        print("[Segmentation] Object segmentation initialized (contour-based)")

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Process frame: detect object, remove background, return frame with
        background blacked out and object preserved.
        """
        if not self.enabled:
            self._last_mask = None
            self._last_contour = None
            return fram

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Step 1: Edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)

        # Step 2: Morphological close to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)

        # Step 3: Dilate to fill interior
        dilated = cv2.dilate(closed, kernel, iterations=2)

        # Step 4: Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            self._last_mask = None
            self._last_contour = None
            return frame

        # Step 5: Find largest contour (the object)
        min_area = int(h * w * self.min_area_ratio)
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

        if not valid_contours:
            self._last_mask = None
            self._last_contour = None
            return frame

        largest_contour = max(valid_contours, key=cv2.contourArea)
        self._last_contour = largest_contour

        # Step 6: Create mask from contour
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        cv2.drawContours(mask, [largest_contour], -1, 255, 2)

        # Step 7: Use GrabCut for refinement if enabled
        if self.use_grabcut:
            try:
                # Use bounding rect as initial rectangle for GrabCut
                bx, by, bw, bh = cv2.boundingRect(largest_contour)
                # Expand slightly
                bx = max(0, bx - 10)
                by = max(0, by - 10)
                bw = min(w - bx, bw + 20)
                bh = min(h - by, bh + 20)

                if bw > 20 and bh > 20:
                    rect = (bx, by, bw, bh)
                    gc_mask = np.zeros((h, w), dtype=np.uint8)
                    bgd_model = np.zeros((1, 65), np.float64)
                    fgd_model = np.zeros((1, 65), np.float64)

                    cv2.grabCut(frame, gc_mask, rect, bgd_model, fgd_model,
                                5, cv2.GC_INIT_WITH_RECT)

                    # Use contour as probable foreground
                    gc_mask2 = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
                    mask = gc_mask2
            except Exception:
                pass  # Fall back to contour mask

        # Step 8: Smooth mask edges
        mask = cv2.medianBlur(mask, 5)
        kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_smooth, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_smooth, iterations=1)

        self._last_mask = mask.astype(bool)

        # Step 9: Apply mask to frame (black background)
        output = np.zeros_like(frame, dtype=np.uint8)
        output[mask > 0] = frame[mask > 0]

        return output

    def get_mask(self) -> Optional[np.ndarray]:
        """Get the latest segmentation mask (bool array) for contour extraction."""
        return self._last_mask

    def get_contour(self) -> Optional[np.ndarray]:
        """Get the latest object contour for grid overlay."""
        return self._last_contour

    def set_threshold(self, t: float):
        pass  # Not used in contour-based approach

    def toggle(self, state: bool = None):
        if state is not None:
            self.enabled = state
        else:
            self.enabled = not self.enabled
        return self.enabled

    def release(self):
        self._last_mask = None
        self._last_contour = None