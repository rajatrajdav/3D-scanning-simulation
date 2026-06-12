"""
Camera module supporting DroidCam (local webcam) and IP Webcam.
New workflow:
  1. Connect to camera (DroidCam or IP Webcam)
  2. Record a video while user rotates the object 360°
  3. Extract N evenly-spaced frames from the video as scan images
"""

import cv2
import numpy as np
import time
import os
from typing import Optional, Tuple, List
from pathlib import Path

from .config import (
    CAMERA_MODE,
    DROIDCAM_INDEX,
    CAMERA_SOURCE,
    SNAPSHOT_URL,
    RESOLUTION,
    VIDEO_RECORD_FPS,
    OUTPUT_DIR,
)


class Camera:
    """Unified camera interface supporting DroidCam (local) and IP Webcam."""

    def __init__(self, resolution: Tuple[int, int] = RESOLUTION):
        self.resolution = resolution
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_streaming = False
        self.mode = CAMERA_MODE

    def open(self) -> bool:
        """Open connection to the camera based on configured mode."""
        if self.mode == "droidcam":
            return self._open_droidcam()
        else:
            return self._open_ipwebcam()

    def _open_droidcam(self) -> bool:
        """Open DroidCam virtual camera (local webcam).
        Auto-detects camera index by trying 0, 1, 2, 3, 4 in sequence
        and using the first second one that successfully opens.
        """
        indices_to_try = [DROIDCAM_INDEX] if DROIDCAM_INDEX >= 0 else []
        if DROIDCAM_INDEX >= 0:
            # Also try other indices as fallback
            for i in range(5):
                if i not in indices_to_try:
                    indices_to_try.append(i)
        else:
            indices_to_try = list(range(5))

        for index in indices_to_try:
            self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(index)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                self.is_streaming = True
                print(f"[Camera] Camera opened at index {index}")
                return True
            self.cap = None

        print("[Camera] Failed to open any camera (tried indices 0-4)")
        return False

    @staticmethod
    def list_cameras(max_indices: int = 10) -> list:
        """Scan camera indices and return list of (index, name) that work."""
        import platform
        available = []
        print("Scanning for cameras...")
        for i in range(max_indices):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Try to get camera name (DirectShow backends support this)
                name = f"Camera {i}"
                try:
                    backend = cap.getBackendName() if hasattr(cap, 'getBackendName') else "Unknown"
                    name = f"Camera {i} (backend: {backend})"
                except:
                    pass
                print(f"  ✓ Index {i}: {name}")
                available.append((i, name))
                cap.release()
            else:
                print(f"  ✗ Index {i}: no camera")
        return available

    def _open_ipwebcam(self) -> bool:
        """Open IP Webcam HTTP MJPEG stream."""
        import requests  # only needed for IP Webcam mode

        self.cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(CAMERA_SOURCE)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.is_streaming = True
            print(f"[Camera] IP Webcam opened: {CAMERA_SOURCE}")
            return True
        return False

    def read_frame(self) -> Optional[np.ndarray]:
        """Read a single frame from the video stream."""
        if self.cap is None or not self.is_streaming:
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def capture_image(self, output_path: str = None) -> Optional[str]:
        """
        Open a live preview and capture a single still image.
        
        Args:
            output_path: Path to save the image. If None, saves to captures/ directory
                         with timestamp filename.
            
        Returns:
            Path to the saved image, or None if capture failed/cancelled.
        """
        if not self.open():
            print("[Camera] Failed to open camera.")
            return None

        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_dir = Path(OUTPUT_DIR) / f"capture_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / "capture.jpg")

        print("=" * 60)
        print("IMAGE CAPTURE MODE")
        print("=" * 60)
        print("  Press 'c' to capture an image")
        print("  Press 'q' or ESC to quit without capturing")
        print(f"  Save path: {output_path}")
        print()

        saved_path = None

        try:
            while self.is_streaming:
                frame = self.read_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue

                display = self._draw_guides(frame.copy())
                cv2.putText(display, "Press 'c' to capture, 'q' to quit",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow("3D Scanner - Image Capture", display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('c'):
                    # Capture the current frame
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    cv2.imwrite(output_path, frame)
                    print(f"\n✓ Image captured and saved to: {output_path}")
                    saved_path = output_path
                    # Show confirmation
                    cv2.putText(display, "IMAGE CAPTURED! Press 'q' to exit",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.imshow("3D Scanner - Image Capture", display)
                    cv2.waitKey(1500)  # Show confirmation for 1.5s
                    break

                if key == ord('q') or key == 27:
                    print("\nCapture cancelled by user.")
                    break

        finally:
            self.release()
            cv2.destroyAllWindows()

        return saved_path

    def show_preview(self, window_name: str = "3D Scanner - Camera Preview") -> None:
        """Open a live preview window. Press 'q' or ESC to exit."""
        if not self.open():
            print("[Camera] Failed to open camera.")
            return

        print(f"[Camera] Preview started. Press 'q' / ESC to quit.")

        try:
            while self.is_streaming:
                frame = self.read_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue

                # Draw alignment guides
                display = self._draw_guides(frame.copy())

                cv2.imshow(window_name, display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:
                    break

        finally:
            self.release()
            cv2.destroyAllWindows()

    def record_video(self, output_path: str, duration_seconds: float = 30.0,
                     window_name: str = "3D Scanner - Recording") -> str:
        """
        Record a video while user rotates the object 360°.
        
        Args:
            output_path: Full path to save the .avi video file (without extension)
            duration_seconds: Max recording duration
            window_name: OpenCV window name
            
        Returns:
            Path to the recorded video file
        """
        if not self.open():
            print("[Camera] Cannot record: camera not opened.")
            return ""

        video_path = f"{output_path}.avi"

        # Get actual frame dimensions
        frame = self.read_frame()
        if frame is None:
            print("[Camera] Could not read test frame.")
            self.release()
            return ""

        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(video_path, fourcc, VIDEO_RECORD_FPS, (w, h))

        if not out.isOpened():
            print("[Camera] Failed to create video writer.")
            self.release()
            return ""

        print("=" * 60)
        print("VIDEO RECORDING MODE")
        print("=" * 60)
        print(f"  Duration: up to {duration_seconds} seconds")
        print(f"  Saving to: {video_path}")
        print()
        print("Instructions:")
        print("  1. Start rotating the object SLOWLY and STEADILY")
        print("  2. Complete one full 360° rotation")
        print("  3. Press 'q' or ESC to stop recording early")
        print("  4. Recording auto-stops after the duration")
        print()

        input("Press Enter when ready to START recording...")
        print("\nRecording started! Rotate the object 360°...\n")

        start_time = time.time()
        frame_count = 0
        recording = True

        try:
            while recording and self.is_streaming:
                frame = self.read_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                out.write(frame)
                frame_count += 1

                # Show live preview with recording indicator
                display = frame.copy()
                cv2.putText(display, "REC", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                elapsed = time.time() - start_time
                remaining = max(0, duration_seconds - elapsed)
                cv2.putText(display, f"{remaining:.0f}s", (w - 120, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                cv2.imshow(window_name, display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:
                    print("\nRecording stopped by user.")
                    recording = False

                if elapsed >= duration_seconds:
                    print("\nRecording duration reached.")
                    recording = False

        finally:
            out.release()
            self.release()
            cv2.destroyAllWindows()

        actual_fps = frame_count / (time.time() - start_time) if frame_count > 0 else 0
        print(f"\nRecording complete: {frame_count} frames, {actual_fps:.1f} FPS")
        print(f"Video saved: {video_path}")

        return video_path

    def extract_frames_from_video(self, video_path: str, num_frames: int = 36,
                                  output_dir: str = ".") -> List[str]:
        """
        Extract evenly-spaced frames from a recorded video.
        
        Args:
            video_path: Path to the .avi video file
            num_frames: Number of frames to extract (default 36 = every 10°)
            output_dir: Directory to save extracted frames
            
        Returns:
            List of paths to extracted frame images
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[Camera] Could not open video: {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if total_frames == 0:
            print("[Camera] Video has no frames!")
            cap.release()
            return []

        # Calculate evenly spaced frame indices
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        print(f"\nExtracting {num_frames} frames from video ({total_frames} total frames)...")

        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []

        for i, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                print(f"  Warning: Could not read frame at index {idx}")
                continue

            filename = f"frame_{i:03d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            saved_paths.append(filepath)

        cap.release()

        print(f"Extracted {len(saved_paths)} frames to: {output_dir}")
        return saved_paths

    def video_scan(self, duration_seconds: float = 30.0,
                   num_angles: int = 36,
                   output_dir: str = ".") -> List[str]:
        """
        Full video-based scan: record video while rotating object,
        then extract evenly-spaced frames.
        
        Args:
            duration_seconds: Recording duration
            num_angles: Number of frames to extract
            output_dir: Output directory for extracted frames
            
        Returns:
            List of paths to extracted frame images
        """
        # Record video
        session_dir = Path(output_dir)
        session_dir.mkdir(parents=True, exist_ok=True)
        video_path = str(session_dir / "scan_video")
        
        recorded = self.record_video(video_path, duration_seconds=duration_seconds)
        if not recorded:
            return []

        # Extract frames
        extracted = self.extract_frames_from_video(recorded, num_frames=num_angles,
                                                    output_dir=output_dir)

        print(f"\nScan complete! {len(extracted)} images saved.")
        return extracted

    def _draw_guides(self, frame: np.ndarray) -> np.ndarray:
        """Draw alignment guides on the frame."""
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        color = (0, 255, 0)
        thickness = 1
        cv2.line(frame, (cx - 30, cy), (cx + 30, cy), color, thickness)
        cv2.line(frame, (cx, cy - 30), (cx, cy + 30), color, thickness)

        margin = 30
        corner_len = 40
        cv2.line(frame, (margin, margin), (margin + corner_len, margin), color, thickness)
        cv2.line(frame, (margin, margin), (margin, margin + corner_len), color, thickness)
        cv2.line(frame, (w - margin, margin), (w - margin - corner_len, margin), color, thickness)
        cv2.line(frame, (w - margin, margin), (w - margin, margin + corner_len), color, thickness)
        cv2.line(frame, (margin, h - margin), (margin + corner_len, h - margin), color, thickness)
        cv2.line(frame, (margin, h - margin), (margin, h - margin - corner_len), color, thickness)
        cv2.line(frame, (w - margin, h - margin), (w - margin - corner_len, h - margin), color, thickness)
        cv2.line(frame, (w - margin, h - margin), (w - margin, h - margin - corner_len), color, thickness)

        return frame

    def release(self) -> None:
        """Release the camera resource."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_streaming = False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()