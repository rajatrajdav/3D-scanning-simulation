"""
3D Scanner - Main Entry Point

Captures 3D scan images using DroidCam (phone camera via DroidCam PC Client).
Workflow:
  1. Preview: Position your object in the camera frame
  2. Record: Rotate the object 360° while recording a video
  3. Extract: The tool extracts evenly-spaced frames at N angles
  4. Reconstruct: Upload to Kiri Engine API for cloud 3D reconstruction

Usage:
    python main.py                          - Full scan (preview + record + extract)
    python main.py --preview-only           - Show camera preview to position object
    python main.py --scan-only              - Record video + extract frames (no preview)
    python main.py --reconstruct            - Upload latest session to Kiri Engine for 3D reconstruction
    python main.py --list-cameras           - Detect available camera indices

Configuration:
    Edit scanner3d/config.py to set:
    - DROIDCAM_INDEX: Camera index (-1 = auto-detect, or 0, 1, 2...)
    - NUM_ANGLES: Number of frames to extract (default: 70)
    - KIRI_API_KEY: Your Kiri Engine API key
"""

import argparse
import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scanner3d.scanner import Scanner3D
from scanner3d.camera import Camera
from scanner3d.config import OUTPUT_DIR, NUM_ANGLES
from scanner3d.kiri_reconstructor import reconstruct as run_reconstruction


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="3D Scanner - Capture images of an object from multiple angles using DroidCam",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                        # Full scan (preview -> record -> extract)
  python main.py --preview-only         # Just test DroidCam connection
  python main.py --scan-only            # Record + extract without preview
  python main.py --list-cameras         # List available camera indices
  python main.py --reconstruct          # Reconstruct latest session via Kiri Engine
  python main.py --reconstruct 20260612_103734  # Reconstruct specific session
  python main.py --angles 60            # Extract 60 frames (every 6 degrees)
  python main.py --duration 45          # Record for 45 seconds
  python main.py --output myscan        # Save to custom directory

Quick Start:
  1. Install DroidCam on your phone and DroidCam PC Client on this PC
  2. Connect PC Client to your phone (same WiFi network)
  3. Run: python main.py --list-cameras
  4. Note which index has video (typically 0, 1, or 2)
  5. Set that index in scanner3d/config.py as DROIDCAM_INDEX
  6. Run: python main.py --preview-only
  7. If camera works, run: python main.py
  8. Rotate the object slowly when prompted
  9. Run: python main.py --reconstruct (uploads to Kiri Engine for 3D model)
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
        help="Scan and list all available camera indices on this system"
    )
    parser.add_argument(
        "--capture-image",
        action="store_true",
        help="Open camera preview and capture a single still image (press 'c')"
    )
    parser.add_argument(
        "--capture-video",
        action="store_true",
        help="Record a video from DroidCam and save it as .avi file"
    )
    parser.add_argument(
        "--reconstruct",
        nargs="?",
        const="auto",
        default=None,
        help="Run 3D reconstruction on captured session via Kiri Engine (default: latest). Pass session ID for specific."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Custom name for the output 3D model"
    )
    parser.add_argument(
        "--quality",
        type=str,
        default="high",
        choices=["draft", "medium", "high", "ultra"],
        help="Kiri Engine reconstruction quality (default: high)"
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=15,
        help="Seconds between Kiri Engine status checks (default: 15)"
    )

    return parser.parse_args()


def run_preview():
    """Just open camera preview to position the object."""
    print("=" * 60)
    print("3D SCANNER - Camera Preview")
    print("=" * 60)
    print("Connecting to DroidCam...")
    print("Make sure DroidCam PC Client is connected to your phone.")
    print()

    cam = Camera()
    try:
        cam.show_preview()
    except Exception as e:
        print(f"Error: {e}")
        print()
        print("Troubleshooting:")
        print("1. Make sure DroidCam PC Client is installed and running")
        print("2. Connect PC Client to your phone (same WiFi network)")
        print("3. Run: python main.py --list-cameras to find the correct camera index")
        print("4. Set DROIDCAM_INDEX in scanner3d/config.py")
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
            print(f"\nScan complete! {len(paths)} images saved.")
            session_dir = scanner.output_dir / scanner.session_id
            print(f"  Location: {session_dir}")
            print(f"\nNext step: Run 'python main.py --reconstruct' to create 3D model via Kiri Engine")
        else:
            print("\nScan failed. Check DroidCam connection.")

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
            print(f"\nImage saved to: {saved}")
    except Exception as e:
        print(f"Error: {e}")
        print()
        print("Troubleshooting:")
        print("1. Make sure DroidCam PC Client is running and connected")
        print("2. Run: python main.py --list-cameras to find the correct camera index")
    finally:
        cam.release()


def run_capture_video(duration: float = 30.0):
    """Record a video from DroidCam."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(OUTPUT_DIR) / f"video_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = str(output_dir / "recording")

    cam = Camera()
    try:
        recorded = cam.record_video(video_path, duration_seconds=duration)
        if recorded:
            print(f"\nVideo saved to: {recorded}")
    except Exception as e:
        print(f"Error: {e}")
        print()
        print("Troubleshooting:")
        print("1. Make sure DroidCam PC Client is running and connected")
        print("2. Run: python main.py --list-cameras to find the correct camera index")
    finally:
        cam.release()


def main():
    """Main entry point."""
    args = parse_args()

    if args.list_cameras:
        print("=" * 60)
        print("3D SCANNER - Camera Detection")
        print("=" * 60)
        print("Make sure DroidCam PC Client is connected to your phone.\n")
        cameras = Camera.list_cameras(max_indices=10)
        print(f"\nFound {len(cameras)} camera(s).")
        if cameras:
            print("\nTo use a specific camera:")
            print("  Set DROIDCAM_INDEX in scanner3d/config.py to the desired index")
            print("  Or set it to -1 for auto-detect (tries 0, 1, 2, 3, 4)")
        return

    if args.capture_image:
        run_capture_image()
        return

    if args.capture_video:
        run_capture_video(duration=args.duration)
        return

    if args.reconstruct:
        session_id = None if args.reconstruct == "auto" else args.reconstruct
        run_reconstruction(
            session_id=session_id,
            output_name=args.model_name,
            quality=args.quality,
            poll_interval=args.poll_interval
        )
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
                print(f"\nScan complete! {len(paths)} images saved.")
                print(f"\nNext step: Run 'python main.py --reconstruct' to create 3D model via Kiri Engine")
        finally:
            scanner.camera.release()
            scanner.cleanup()

    else:
        # Full scan with preview
        run_scan(args.angles, args.duration, output_dir=args.output)


if __name__ == "__main__":
    main()