"""
3D Scanner - Main Entry Point

Captures 3D scan images using your phone camera (via DroidCam or IP Webcam).
Workflow:
  1. Preview: Position your object in the camera frame
  2. Record: Rotate the object 360° while recording a video
  3. Extract: The tool extracts evenly-spaced frames at N angles

Usage:
    python main.py                    - Full scan (preview + record + extract)
    python main.py --preview-only     - Just show camera preview to position object
    python main.py --scan-only        - Record video + extract frames (no preview)

Configuration:
    Edit scanner3d/config.py to set:
    - CAMERA_MODE: "droidcam" (default) or "ipwebcam"
    - DROIDCAM_INDEX: Camera index (default 0)
    - NUM_ANGLES: Number of frames to extract (default 36)
"""

import argparse
import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scanner3d.scanner import Scanner3D, quick_scan
from scanner3d.camera import Camera
from scanner3d.config import OUTPUT_DIR, NUM_ANGLES, DROIDCAM_INDEX, CAMERA_MODE


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="3D Scanner - Capture images of an object from multiple angles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                        # Full scan (preview → record → extract)
  python main.py --preview-only         # Just test camera connection
  python main.py --scan-only            # Record + extract without preview
  python main.py --angles 60            # Extract 60 frames (every 6 degrees)
  python main.py --duration 45          # Record for 45 seconds
  python main.py --output myscan        # Save to custom directory

Quick Start:
  1. Install DroidCam on your phone and PC, connect them
  2. Run: python main.py --preview-only
  3. If camera works, run: python main.py
  4. Rotate the object slowly when prompted
        """
    )

    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Open camera preview window without scanning"
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Skip preview, go straight to recording + extracting"
    )
    parser.add_argument(
        "--angles",
        type=int,
        default=NUM_ANGLES,
        help=f"Number of angles to extract (default: {NUM_ANGLES})"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Recording duration in seconds (default: 30)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output directory for captured images"
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Scan and list all available camera indices"
    )
    parser.add_argument(
        "--capture-image",
        action="store_true",
        help="Open camera preview and capture a single still image (press 'c')"
    )
    parser.add_argument(
        "--capture-video",
        action="store_true",
        help="Record a video from the DroidCam and save it as .avi file"
    )

    return parser.parse_args()


def run_preview():
    """Just open camera preview to position the object."""
    print("=" * 60)
    print("3D SCANNER - Camera Preview")
    print("=" * 60)
    print("Connecting to camera via DroidCam...")
    print("Make sure DroidCam is connected on your phone.")
    print()

    cam = Camera()
    try:
        cam.show_preview()
    except Exception as e:
        print(f"Error: {e}")
        print()
        print("Troubleshooting:")
        print("1. Make sure DroidCam is running on your phone and PC")
        print("2. Check DROIDCAM_INDEX in scanner3d/config.py")
        print("3. Try setting CAMERA_MODE = 'ipwebcam' if using IP Webcam instead")
    finally:
        cam.release()


def run_scan(num_angles: int, duration: float, output_dir: str = None):
    """Run the full video-based scanning session."""
    print("=" * 60)
    print("3D SCANNER - Video Scan Session")
    print("=" * 60)

    scanner = Scanner3D()
    if output_dir:
        scanner.output_dir = Path(output_dir)

    try:
        # Step 1: Preview (optional)
        print("\n[Step 1] Position your object in the camera view")
        print("Close the preview window (press 'q') when ready.")
        scanner.run_preview()

        # Step 2: Video recording + frame extraction
        print("\n[Step 2] Recording video - rotate the object 360°...")
        paths = scanner.run_video_scan(
            num_angles=num_angles,
            duration_seconds=duration
        )

        if paths:
            print(f"\n✓ Scan complete! {len(paths)} images saved.")
            session_dir = scanner.output_dir / scanner.session_id
            print(f"  Location: {session_dir}")
        else:
            print("\n✗ Scan failed. Check camera connection.")

        return paths

    finally:
        scanner.camera.release()
        scanner.cleanup()


def run_capture_image():
    """Open preview and capture a single still image."""
    cam = Camera()
    try:
        saved = cam.capture_image()
        if saved:
            print(f"\n✓ Image saved to: {saved}")
    except Exception as e:
        print(f"Error: {e}")
        print()
        print("Troubleshooting:")
        print("1. Make sure DroidCam is running on your phone and PC")
        print("2. Check DROIDCAM_INDEX in scanner3d/config.py")
    finally:
        cam.release()


def run_capture_video(duration: float = 30.0):
    """Record a video from the DroidCam."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(OUTPUT_DIR) / f"video_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = str(output_dir / "recording")

    cam = Camera()
    try:
        recorded = cam.record_video(video_path, duration_seconds=duration)
        if recorded:
            print(f"\n✓ Video saved to: {recorded}")
    except Exception as e:
        print(f"Error: {e}")
        print()
        print("Troubleshooting:")
        print("1. Make sure DroidCam is running on your phone and PC")
        print("2. Check DROIDCAM_INDEX in scanner3d/config.py")
    finally:
        cam.release()


def main():
    """Main entry point."""
    args = parse_args()

    if args.list_cameras:
        print("=" * 60)
        print("3D SCANNER - Camera Detection")
        print("=" * 60)
        print("DroidCam must be connected on your phone and PC.")
        print("Look for which index has your phone camera feed.\n")
        cameras = Camera.list_cameras(max_indices=10)
        print(f"\nFound {len(cameras)} camera(s).")
        if cameras:
            print("\nTo use a specific camera, set DROIDCAM_INDEX in config.py")
            print(f"Currently configured: DROIDCAM_INDEX = {DROIDCAM_INDEX}")
        return

    if args.capture_image:
        run_capture_image()
        return

    if args.capture_video:
        run_capture_video(duration=args.duration)
        return

    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    if args.preview_only:
        run_preview()

    elif args.scan_only:
        # Scan without preview
        scanner = Scanner3D()
        try:
            paths = scanner.run_video_scan(
                num_angles=args.angles,
                duration_seconds=args.duration
            )
            if paths:
                print(f"\n✓ Scan complete! {len(paths)} images saved.")
        finally:
            scanner.camera.release()
            scanner.cleanup()

    else:
        # Full scan with preview
        run_scan(args.angles, args.duration, output_dir=args.output)


if __name__ == "__main__":
    main()