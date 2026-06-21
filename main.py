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
    python main.py --reconstruct --file-format STL  - Reconstruct to STL format
    python main.py --list-cameras           - Detect available camera indices
    python main.py --live-tracking          - Live ORB feature tracking with point cloud visualization
    python main.py --capture-image          - Capture a single still image

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

from scanner3d.camera import Camera
from scanner3d.config import OUTPUT_DIR, NUM_ANGLES
from scanner3d.kiri_reconstructor import reconstruct as run_reconstruction

# Lazy import: LiveTracker3D requires open3d which is optional.
# We import it only when --live-tracking is used to avoid breaking
# the rest of the app when open3d is not installed.
_LIVE_TRACKER_CLASS = None  # Cached reference to LiveTracker3D class

def _get_live_tracker():
    """Lazy import of LiveTracker3D. Returns the class or None if unavailable."""
    global _LIVE_TRACKER_CLASS
    if _LIVE_TRACKER_CLASS is not None:
        return _LIVE_TRACKER_CLASS
    try:
        from scanner3d.scanner import LiveTracker3D
        _LIVE_TRACKER_CLASS = LiveTracker3D
        return _LIVE_TRACKER_CLASS
    except (ImportError, ModuleNotFoundError):
        _LIVE_TRACKER_CLASS = False
        return None


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
  python main.py --reconstruct --file-format STL  # Reconstruct to STL
  python main.py --reconstruct 20260612_103734    # Reconstruct specific session
  python main.py --angles 60            # Extract 60 frames (every 6 degrees)
  python main.py --duration 45          # Record for 45 seconds
  python main.py --output myscan        # Save to custom directory
  python main.py --live-tracking        # Live ORB feature tracking + point cloud

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
        "--live-tracking",
        action="store_true",
        help="Start live ORB feature tracking with Open3D point cloud visualization (replaces old Scanner3D)"
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
        "--file-format",
        type=str,
        default="OBJ",
        choices=["OBJ", "STL", "GLB", "FBX", "PLY"],
        help="Output 3D format for Kiri Engine reconstruction (default: OBJ)"
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
    parser.add_argument(
        "--no-view",
        action="store_true",
        help="Skip opening the 3D viewer after reconstruction"
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


def run_scan(num_angles: int, duration: float, output_dir: str = None, with_preview: bool = True):
    """
    Run the full video-based scanning session using Camera directly.

    NOTE: The old Scanner3D class has been replaced by LiveTracker3D (for live
    ORB point-cloud tracking). For the capture workflow used here (record video
    + extract frames), Camera is used directly since Scanner3D no longer exists.

    Args:
        num_angles: Number of frames to extract from the video.
        duration: Recording duration in seconds.
        output_dir: Optional custom output directory.
        with_preview: If True, show a preview window before recording.

    Returns:
        List of paths to extracted frame images, or empty list on failure.
    """
    print("=" * 60)
    print("3D SCANNER - Video Scan Session")
    print("=" * 60)

    # Create session directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_dir = Path(output_dir) if output_dir else Path(OUTPUT_DIR)
    session_dir = base_dir / timestamp
    session_dir.mkdir(parents=True, exist_ok=True)

    cam = Camera()
    try:
        if not cam.open():
            print("[Scan] Failed to open camera.")
            return []

        # Step 1: Preview (optional)
        if with_preview:
            print("\n[Step 1] Position your object in the camera view")
            print("Close the preview window (press 'q') when ready.")
            cam.show_preview()

        # Step 2: Video recording
        print("\n[Step 2] Recording video - rotate the object 360°...")
        video_basename = str(session_dir / "scan_video")
        recorded = cam.record_video(video_basename, duration_seconds=duration)

        if not recorded:
            print("[Scan] Video recording failed.")
            return []

        # Step 3: Frame extraction
        print("\n[Step 3] Extracting evenly-spaced frames...")
        extracted = cam.extract_frames_from_video(
            recorded,
            num_frames=num_angles,
            output_dir=str(session_dir)
        )

        if extracted:
            print(f"\nScan complete! {len(extracted)} images saved.")
            print(f"  Location: {session_dir}")
            print(f"\nNext step: Run 'python main.py --reconstruct' to create 3D model via Kiri Engine")
        else:
            print("\nFrame extraction failed.")

        return extracted

    finally:
        cam.release()


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


def run_live_tracking_mode():
    """
    Run the new LiveTracker3D for live ORB feature tracking + Open3D point cloud.

    This replaces the old Scanner3D class which no longer exists in scanner.py.
    LiveTracker3D provides real-time feature detection/tracking via Lucas-Kanade
    optical flow and renders a live, persistent point cloud in Open3D.
    """
    print("=" * 60)
    print("3D SCANNER - Live ORB Feature Tracking")
    print("=" * 60)
    print("This mode provides real-time feature tracking with Open3D point cloud visualization.")
    print("Press 'q' in the video window to quit.\n")

    LiveTracker3D = _get_live_tracker()
    if LiveTracker3D is None:
        print("Error: LiveTracker3D not available (open3d may not be installed).")
        print("  Install: pip install open3d")
        print("  Or use the standard scan workflow: python main.py")
        return

    tracker = LiveTracker3D()
    try:
        tracker.run()
    except Exception as e:
        print(f"\nError in live tracking: {e}")
        print()
        print("Troubleshooting:")
        print("1. Make sure camera is connected and working")
        print("2. Run: python main.py --list-cameras to find the correct camera index")
        print("3. Set DROIDCAM_INDEX in scanner3d/config.py")
    finally:
        if hasattr(tracker, '_shutdown'):
            tracker._shutdown()


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

    if args.live_tracking:
        run_live_tracking_mode()
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
            file_format=args.file_format,
            auto_view=not args.no_view,
            poll_interval=args.poll_interval
        )
        return

    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    if args.preview_only:
        run_preview()

    elif args.scan_only:
        run_scan(
            args.angles,
            args.duration,
            output_dir=args.output,
            with_preview=False
        )

    else:
        # Full scan with preview
        run_scan(
            args.angles,
            args.duration,
            output_dir=args.output,
            with_preview=True
        )


if __name__ == "__main__":
    main()