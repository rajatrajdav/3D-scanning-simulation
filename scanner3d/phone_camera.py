"""
IP Webcam Phone Camera Module.

Connects to the IP Webcam app running on your phone.
IP Webcam provides a proper MJPEG stream that OpenCV can open directly.

How to use:
  1. Install "IP Webcam" from Play Store on your phone
  2. Open the app and tap "Start Server"
  3. Note the URL shown (e.g., http://10.138.159.186:8080)
  4. Run this app - it connects automatically!

IP Webcam features:
  - MJPEG video stream at /video
  - Snapshots at /photo.jpg
  - Focus control at /focus
  - Torch/LED control at /torch
  - Settings at /settings

Usage:
    from scanner3d.phone_camera import PhoneCamera
    cam = PhoneCamera()
    if cam.open():
        frame = cam.read_frame()
"""

import cv2
import numpy as np
import urllib.request
import time
import os
from typing import Optional, Tuple, List
from pathlib import Path

from .config import (
    IP_WEBCAM_URL,
    IP_WEBCAM_STREAM_PATH,
    IP_WEBCAM_SNAPSHOT_PATH,
    RESOLUTION,
    VIDEO_RECORD_FPS,
    OUTPUT_DIR,
)


class PhoneCamera:
    """
    Captures video from your phone's IP Webcam app over WiFi.
    
    IP Webcam provides a standard MJPEG stream that OpenCV handles
    natively. Just start the server on your phone and this connects.
    """

    def __init__(self, stream_url: str = None, resolution: Tuple[int, int] = RESOLUTION):
        """
        Args:
            stream_url: Full URL to the MJPEG stream.
                        Default: http://10.138.159.186:8080/video
            resolution: Desired frame resolution
        """
        if stream_url is None:
            self.stream_url = IP_WEBCAM_URL + IP_WEBCAM_STREAM_PATH
            self.base_url = IP_WEBCAM_URL
        else:
            self.stream_url = stream_url
            self.base_url = stream_url.rsplit("/", 1)[0]

        self.resolution = resolution
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_streaming = False
        self.source_name = ""

    def open(self) -> bool:
        """Connect to the IP Webcam stream on your phone."""
        print(f"[PhoneCamera] Connecting to IP Webcam: {self.stream_url}")
        
        try:
            # OpenCV can open IP Webcam MJPEG streams directly
            self.cap = cv2.VideoCapture(self.stream_url)
            
            if not self.cap.isOpened():
                # Try alternative paths
                for alt_path in ["/mjpeg", "/videofeed", "/video.mjpg"]:
                    alt_url = self.base_url + alt_path
                    print(f"[PhoneCamera] Trying: {alt_url}")
                    self.cap = cv2.VideoCapture(alt_url)
                    if self.cap.isOpened():
                        self.stream_url = alt_ul
                        break
            
            if self.cap and self.cap.isOpened():
                # Test frame read
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    self.is_streaming = True
                    self.source_name = f"IP Webcam ({w}x{h})"
                    print(f"[PhoneCamera] ✓ Connected! Resolution: {w}x{h}")
                    print(f"[PhoneCamera]   Stream: {self.stream_url}")
                    return True
                else:
                    print("[PhoneCamera] ✗ Connected but no frames received")
                    self.cap.release()
                    self.cap = None
            else:
                print("[PhoneCamera] ✗ Could not open stream")
                
        except Exception as e:
            print(f"[PhoneCamera] ✗ Error: {e}")
            if self.cap:
                self.cap.release()
                self.cap = None
        
        # Failed - show instructions
        print()
        print("=" * 60)
        print("IP WEBCAM SETUP")
        print("=" * 60)
        print()
        print("1. Install IP Webcam from Play Store on your phone")
        print("   https://play.google.com/store/apps/details?id=com.pas.webcam")
        print()
        print("2. Open the app and tap 'Start Server'")
        print(f"   Your URL: {IP_WEBCAM_URL}")
        print()
        print("3. Make sure your phone is on the SAME WiFi network")
        print()
        print("4. Run this app again - it will connect automatically!")
        print()
        print("   Or test with: python -m scanner3d.phone_camera")
        print("=" * 60)
        print()
        
        return False

    def read_frame(self) -> Optional[np.ndarray]:
        """Read a single frame from the video stream."""
        if self.cap is None or not self.is_streaming:
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def capture_snapshot(self) -> Optional[np.ndarray]:
        """
        Capture a high-res still image from the phone camera.
        Uses the /photo.jpg endpoint for better quality.
        """
        try:
            url = self.base_url + IP_WEBCAM_SNAPSHOT_PATH
            resp = urllib.request.urlopen(url, timeout=5)
            data = resp.read()
            resp.close()
            
            if len(data) > 100 and data[0:2] == b'\xff\xd8':
                arr = np.frombuffer(data, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                return img
        except Exception:
            pass
        return None

    def show_preview(self, window_name: str = "3D Scanner - Phone Camera") -> None:
        """Open a live preview window. Press 'q' or ESC to exit."""
        if not self.open():
            print("[PhoneCamera] Failed to open camera.")
            return

        print(f"[PhoneCamera] Live preview started. Press 'q' / ESC to quit.")
        print(f"[PhoneCamera] Source: {self.source_name}")

        try:
            while self.is_streaming:
                frame = self.read_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue

                display = self._draw_guides(frame.copy())
                cv2.putText(display, f"IP Webcam - Press 'q' to quit",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow(window_name, display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:
                    break

        finally:
            self.release()
            cv2.destroyAllWindows()

    def capture_image(self, output_path: str = None) -> Optional[str]:
        """Open a live preview and capture a single still image."""
        if not self.open():
            print("[PhoneCamera] Failed to open camera.")
            return None

        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_dir = Path(OUTPUT_DIR) / f"capture_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / "capture.jpg")

        print("=" * 60)
        print("PHONE CAMERA - IMAGE CAPTURE")
        print("=" * 60)
        print("  Press 'c' to capture an image")
        print("  Press 'q' or ESC to quit")
        print(f"  Save path: {output_path}")
        print()

        saved_path = None
        try:
            while self.is_streaming:
                frame = self.read_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue

                display = frame.copy()
                cv2.putText(display, "Press 'c' to capture, 'q' to quit",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow("Phone Camera - Capture", display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('c'):
                    # Try snapshot first for better quality
                    snapshot = self.capture_snapshot()
                    if snapshot is not None:
                        frame = snapshot
                    
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    cv2.imwrite(output_path, frame)
                    print(f"\n✓ Image captured: {output_path}")
                    saved_path = output_path
                    break

                if key == ord('q') or key == 27:
                    print("\nCapture cancelled.")
                    break

        finally:
            self.release()
            cv2.destroyAllWindows()

        return saved_path

    def record_video(self, output_path: str, duration_seconds: float = 30.0,
                     window_name: str = "3D Scanner - Recording") -> str:
        """Record video from the phone camera while rotating object 360°."""
        if not self.open():
            print("[PhoneCamera] Cannot record: camera not opened.")
            return ""

        video_path = f"{output_path}.avi"
        
        frame = self.read_frame()
        if frame is None:
            print("[PhoneCamera] Could not get initial frame.")
            self.release()
            return ""

        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(video_path, fourcc, VIDEO_RECORD_FPS, (w, h))

        if not out.isOpened():
            print("[PhoneCamera] Failed to create video writer.")
            self.release()
            return ""

        print("=" * 60)
        print("PHONE CAMERA - RECORDING MODE")
        print("=" * 60)
        print(f"  Duration: up to {duration_seconds} seconds")
        print(f"  Saving to: {video_path}")
        print()
        print("Instructions:")
        print("  1. Place your object in front of the phone camera")
        print("  2. Rotate the object SLOWLY through 360°")
        print("  3. Press 'q' or ESC to stop early")
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

                display = frame.copy()
                cv2.putText(display, "REC - IP Webcam", (10, 40),
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
        """Extract evenly-spaced frames from a recorded video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[PhoneCamera] Could not open video: {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            print("[PhoneCamera] Video has no frames!")
            cap.release()
            return []

        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        print(f"\nExtracting {num_frames} frames from video ({total_frames} total frames)...")

        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []

        for i, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            filename = f"frame_{i:03d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            saved_paths.append(filepath)

        cap.release()
        print(f"Extracted {len(saved_paths)} frames to: {output_dir}")
        return saved_paths

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


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("IP Webcam Connection Test")
    print("=" * 60)
    print(f"Connecting to: {IP_WEBCAM_URL}")
    print()
    
    cam = PhoneCamera()
    if cam.open():
        print(f"\n✓ Connected! Source: {cam.source_name}")
        print("  Starting preview (press 'q' to quit)...")
        cam.show_preview()
    else:
        print("\n✗ Connection failed.")
        print()
        print("Make sure:")
        print(f"  1. IP Webcam is running on your phone")
        print(f"  2. URL is correct: {IP_WEBCAM_URL}")
        print(f"  3. Phone and PC are on the same WiFi network")