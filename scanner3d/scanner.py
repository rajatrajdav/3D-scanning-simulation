"""
Real-Time Visual Odometry & Point Cloud Visualization Module
==============================================================

Live tracking, video capture guidance, and point-cloud visualization for
the 3D scanner project. Tracks ORB features via Lucas-Kanade optical
flow, estimates relative depth from pixel-motion magnitude, and renders
a live, persistent point cloud in Open3D while overlaying an HUD on the
OpenCV camera window.

Architecture
------------
1. Video Ingestion      : cv2.VideoCapture (webcam index or video file path)
2. Feature Detection    : cv2.ORB_create
3. Feature Tracking     : cv2.calcOpticalFlowPyrLK (sparse Lucas-Kanade)
4. 2D -> 3D Projection  : depth ~ 1 / ||(dx, dy)||  (relative, simulated)
5. Visualization        : o3d.visualization.Visualizer (non-blocking loop)
6. HUD / Guidance       : cv2.drawKeypoints + cv2.putText overlay
"""

import time

import cv2
import numpy as np
import open3d as o3d


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
VIDEO_SOURCE = 0          # 0 = default webcam, or a path to an .mp4 file
MAX_POINTS = 20_000       # Hard cap on point-cloud size (memory guard)
MIN_TRACK_FEATURES = 50   # Re-detect ORB features once tracked set thins below this
ORB_FEATURES = 500        # Max ORB features requested per detection pass
DEPTH_SCALE = 5000.0      # Tunes how "deep" small pixel motions appear
MIN_DISPLACEMENT = 0.5    # px, floor for displacement magnitude (avoid div-by-~0)

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


class LiveTracker3D:
    """
    Real-time ORB + Lucas-Kanade feature tracker that back-projects 2D
    pixel motion into a relative 3D point cloud, rendered live in Open3D
    alongside an annotated OpenCV preview window.
    """

    def __init__(self, video_source=VIDEO_SOURCE, max_points=MAX_POINTS):
        self.video_source = video_source
        self.max_points = max_points

        # --- OpenCV capture setup ---
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            raise RuntimeError(f"[LiveTracker3D] Could not open video source: {video_source}")

        # --- ORB detector ---
        self.orb = cv2.ORB_create(nfeatures=ORB_FEATURES)

        # --- Tracking state ---
        self.prev_gray = None
        self.prev_pts = None  # Nx1x2 float32, points currently tracked

        # --- Point-cloud accumulation buffers ---
        self.points_xyz = np.empty((0, 3), dtype=np.float32)   # Nx3
        self.points_rgb = np.empty((0, 3), dtype=np.float64)   # Nx3 (Open3D wants float64 colors)

        # --- Open3D visualizer setup (non-blocking) ---
        self.pcd = o3d.geometry.PointCloud()
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="3D Scanner - Live Point Cloud", width=960, height=720)
        self.vis.add_geometry(self.pcd)

        # FPS bookkeeping
        self._last_tick = time.time()
        self._fps = 0.0

    # ------------------------------------------------------------------ #
    # Feature detection / re-seeding
    # ------------------------------------------------------------------ #
    def _detect_features(self, gray_frame):
        """Detect ORB keypoints on a grayscale frame and return them as an
        Nx1x2 float32 array usable by cv2.calcOpticalFlowPyrLK, alongside
        the raw cv2.KeyPoint list (used for the green HUD overlay)."""
        keypoints = self.orb.detect(gray_frame, None)
        if not keypoints:
            return None, []
        pts = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)
        return pts, keypoints

    # ------------------------------------------------------------------ #
    # 2D motion -> relative 3D back-projection
    # ------------------------------------------------------------------ #
    def _backproject(self, prev_pts, curr_pts, frame_shape):
        """
        Convert 2D pixel displacement vectors into relative (X, Y, Z)
        coordinates.

        X, Y : current pixel position, recentered around the frame center
               and normalized to a roughly [-1, 1] range.
        Z    : estimated depth, modeled as inversely proportional to the
               magnitude of the optical-flow displacement vector. Large
               pixel motion -> object assumed closer (smaller Z); small
               motion -> object assumed farther (larger Z). This is a
               *relative*, simulated depth -- not a metric measurement.
        """
        h, w = frame_shape[:2]
        cx, cy = w / 2.0, h / 2.0

        prev_xy = prev_pts.reshape(-1, 2)
        curr_xy = curr_pts.reshape(-1, 2)

        disp = curr_xy - prev_xy                  # (dx, dy) per point
        mag = np.linalg.norm(disp, axis=1)         # displacement magnitude
        mag = np.maximum(mag, MIN_DISPLACEMENT)    # guard against div-by-0

        # Inverse relationship: small motion -> far (large Z), large motion -> near (small Z)
        z = DEPTH_SCALE / mag

        # Recenter & normalize X, Y by frame dimensions for a stable relative scale
        x = (curr_xy[:, 0] - cx) / max(w, h)
        y = (curr_xy[:, 1] - cy) / max(w, h)

        xyz = np.stack([x, y, z], axis=1).astype(np.float32)
        return xyz, mag

    @staticmethod
    def _depth_to_color(z_values):
        """
        Map depth (Z) values to an RGB gradient (Green = near, Red = far)
        for the Open3D point cloud, normalized per-batch.
        """
        if len(z_values) == 0:
            return np.empty((0, 3), dtype=np.float64)
        z_min, z_max = float(np.min(z_values)), float(np.max(z_values))
        spread = max(z_max - z_min, 1e-6)
        norm = (z_values - z_min) / spread  # 0 (near) -> 1 (far)
        red = norm
        green = 1.0 - norm
        blue = np.zeros_like(norm)
        return np.stack([red, green, blue], axis=1).astype(np.float64)

    # ------------------------------------------------------------------ #
    # Point-cloud buffer management (memory-capped sliding window)
    # ------------------------------------------------------------------ #
    def _append_points(self, new_xyz, new_rgb):
        """Append new points to the persistent cloud, trimming the oldest
        entries once max_points is exceeded (FIFO sliding window) to keep
        memory bounded and rendering fast."""
        self.points_xyz = np.vstack([self.points_xyz, new_xyz])
        self.points_rgb = np.vstack([self.points_rgb, new_rgb])

        if len(self.points_xyz) > self.max_points:
            excess = len(self.points_xyz) - self.max_points
            self.points_xyz = self.points_xyz[excess:]
            self.points_rgb = self.points_rgb[excess:]

    def _refresh_geometry(self):
        """Push the latest point/colour buffers into the Open3D point cloud
        and notify the visualizer renderer."""
        self.pcd.points = o3d.utility.Vector3dVector(self.points_xyz.astype(np.float64))
        self.pcd.colors = o3d.utility.Vector3dVector(self.points_rgb)
        self.vis.update_geometry(self.pcd)

    # ------------------------------------------------------------------ #
    # HUD overlay
    # ------------------------------------------------------------------ #
    def _draw_hud(self, frame, keypoints, tracking_count):
        """Draw tracked keypoints (green) and an informational HUD text
        overlay onto the OpenCV preview frame."""
        if keypoints:
            frame = cv2.drawKeypoints(
                frame, keypoints, None,
                color=(0, 255, 0),
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )

        cv2.putText(frame, f"Points Mapped: {len(self.points_xyz)}",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Tracking Features: {tracking_count}",
                    (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS: {self._fps:.1f}",
                    (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit",
                    (15, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        return frame

    def _update_fps(self):
        now = time.time()
        dt = now - self._last_tick
        self._last_tick = now
        if dt > 0:
            instant_fps = 1.0 / dt
            # Light exponential smoothing so the readout isn't jumpy
            self._fps = self._fps * 0.9 + instant_fps * 0.1 if self._fps else instant_fps

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self):
        """Main capture/track/visualize loop. Runs until 'q' is pressed or
        the video source is exhausted, then releases all resources."""
        print("[LiveTracker3D] Starting live tracking. Press 'q' in the video window to quit.")

        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    print("[LiveTracker3D] End of stream / capture failed.")
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                display_keypoints = []

                if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) < MIN_TRACK_FEATURES:
                    # (Re)seed features on the first frame, or whenever the
                    # tracked set has thinned out too much.
                    self.prev_pts, display_keypoints = self._detect_features(gray)
                    self.prev_gray = gray
                    tracking_count = len(display_keypoints)

                else:
                    # Track existing points forward with pyramidal Lucas-Kanade.
                    curr_pts, status, _err = cv2.calcOpticalFlowPyrLK(
                        self.prev_gray, gray, self.prev_pts, None, **LK_PARAMS
                    )

                    if curr_pts is not None and status is not None:
                        status = status.reshape(-1)
                        good_prev = self.prev_pts.reshape(-1, 2)[status == 1]
                        good_curr = curr_pts.reshape(-1, 2)[status == 1]

                        if len(good_curr) > 0:
                            xyz, _mag = self._backproject(good_prev, good_curr, frame.shape)
                            rgb = self._depth_to_color(xyz[:, 2])
                            self._append_points(xyz, rgb)

                            # Build KeyPoint objects purely for the green HUD overlay
                            display_keypoints = [cv2.KeyPoint(float(p[0]), float(p[1]), 5)
                                                  for p in good_curr]

                        # Carry forward only the successfully tracked points
                        self.prev_pts = (good_curr.reshape(-1, 1, 2).astype(np.float32)
                                          if len(good_curr) > 0 else None)

                    tracking_count = 0 if self.prev_pts is None else len(self.prev_pts)
                    self.prev_gray = gray

                # --- Refresh both UIs ---
                self._refresh_geometry()
                self.vis.poll_events()
                self.vis.update_renderer()

                self._update_fps()
                annotated = self._draw_hud(frame, display_keypoints, tracking_count)
                cv2.imshow("3D Scanner - Live Tracking", annotated)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[LiveTracker3D] Quit requested by user.")
                    break

        finally:
            self._shutdown()

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #
    def _shutdown(self):
        """Release every system resource opened by this tracker."""
        self.cap.release()
        cv2.destroyAllWindows()
        self.vis.destroy_window()
        print(f"[LiveTracker3D] Shutdown complete. Total points captured: {len(self.points_xyz)}")


def run_live_tracking(video_source=VIDEO_SOURCE):
    """Convenience entry point, mirroring the style of quick_scan() in
    scanner_core.py -- instantiate and run the tracker end-to-end."""
    tracker = LiveTracker3D(video_source=video_source)
    tracker.run()


if __name__ == "__main__":
    run_live_tracking()