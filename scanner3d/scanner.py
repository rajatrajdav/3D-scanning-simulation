"""
3D Scanner core module.
New workflow:
  1. Preview camera to position object
  2. Record a video while rotating the object 360°
  3. Extract evenly-spaced frames from the video
  4. Save extracted frames as scan images
"""

import cv2
import numpy as np
import os
import time
from typing import List, Optional
from pathlib import Path

from .config import NUM_ANGLES, OUTPUT_DIR
from .camera import Camera


class Scanner3D:
    """
    Orchestrates the 3D scanning process using video-based capture.
    Records a video while user rotates the object, then extracts
    evenly-spaced frames for 3D reconstruction input.
    """

    def __init__(self, camera: Optional[Camera] = None, output_dir: str = OUTPUT_DIR):
        self.camera = camera or Camera()
        self.output_dir = Path(output_dir)
        self.session_id: str = ""
        self.extracted_paths: List[str] = []

    def run_preview(self) -> None:
        """Open camera preview with guides for positioning the object."""
        print("Camera preview mode.")
        print("Position the object in the center, then press 'q' to close.")
        self.camera.show_preview()

    def run_video_scan(self, num_angles: int = NUM_ANGLES,
                       duration_seconds: float = 30.0) -> List[str]:
        """
        Run a video-based 360° scan.
        
        Args:
            num_angles: Number of frames to extract (default 36 = every 10°)
            duration_seconds: Recording duration in seconds
            
        Returns:
            List of file paths to extracted frame images
        """
        self.session_id = time.strftime("%Y%m%d_%H%M%S")
        session_dir = self.output_dir / self.session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print("3D VIDEO SCAN SESSION")
        print("=" * 60)
        print(f"Session ID: {self.session_id}")
        print(f"Angles to extract: {num_angles}")
        print(f"Output: {session_dir}")
        print()

        # Step 1: Preview
        print("[Step 1] Position your object in the camera view")
        print("Close the preview window (press 'q') when ready.")
        self.camera.show_preview()

        # Step 2: Record video while rotating the object
        print("\n[Step 2] Recording video - rotate the object 360°")
        video_path = os.path.join(str(session_dir), "scan_video")
        recorded = self.camera.record_video(
            video_path,
            duration_seconds=duration_seconds
        )
        if not recorded:
            print("[Scanner] Video recording failed.")
            return []

        # Step 3: Extract evenly-spaced frames
        print("\n[Step 3] Extracting frames from video...")
        self.extracted_paths = self.camera.extract_frames_from_video(
            recorded,
            num_frames=num_angles,
            output_dir=str(session_dir)
        )

        # Clean up video file to save space
        if os.path.exists(recorded):
            os.remove(recorded)
            print(f"[Scanner] Deleted temporary video: {recorded}")

        print(f"\nScan complete! {len(self.extracted_paths)} images saved.")
        return self.extracted_paths

    def run_preprocess(self) -> List[str]:
        """
        Apply preprocessing (denoise, enhance contrast) to extracted frames.
        Saves processed versions alongside originals.
        
        Returns:
            List of paths to processed images
        """
        if not self.extracted_paths:
            print("[Scanner] No frames to preprocess.")
            return []

        processed_paths = []
        for img_path in self.extracted_paths:
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            processed = self._preprocess_frame(frame)

            # Save processed version with _processed suffix
            p = Path(img_path)
            proc_path = str(p.parent / f"{p.stem}_processed{p.suffix}")
            cv2.imwrite(proc_path, processed)
            processed_paths.append(proc_path)

        print(f"[Scanner] Preprocessed {len(processed_paths)} images.")
        return processed_paths

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a captured frame:
        - Mild denoising
        - Contrast enhancement (CLAHE)
        """
        denoised = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return enhanced

    def cleanup(self) -> None:
        """Clear extracted paths from memory."""
        self.extracted_paths.clear()


def quick_scan():
    """
    Convenience function to run a quick video-based scan end-to-end.
    """
    scanner = Scanner3D()
    try:
        # Step 1: Preview
        print("Step 1: Position your object in the camera view")
        scanner.run_preview()

        # Step 2: Video scan
        print("\nStep 2: Scanning the object via video...")
        paths = scanner.run_video_scan()

        # Step 3: Preprocess
        print("\nStep 3: Preprocessing images...")
        processed = scanner.run_preprocess()

        print(f"\n✓ Scan complete! {len(paths)} images captured.")
        return paths
    finally:
        scanner.camera.release()
        scanner.cleanup()


if __name__ == "__main__":
    quick_scan()