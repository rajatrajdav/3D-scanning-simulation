"""
Kiri Engine 3D Reconstruction Pipeline
========================================
Merges three previously separate scripts into one working pipeline:

  1. kiri_reconstructor.py  -> upload video, get a "serialize" task ID
  2. download.py            -> poll Kiri for the task, download + extract the model zip
  3. view.py                -> load the extracted OBJ/STL with trimesh and view it

No local GPU or COLMAP needed - reconstruction happens on Kiri's servers.

Why this merge was needed (not just a copy/paste):
  The upload endpoint (`/v1/open/photo/video`) does NOT return a direct model
  download URL. It only returns a task `serialize` ID:
      {"code": 0, "data": {"serialize": "...", "calculateType": 1}, "ok": true}
  The original kiri_reconstructor.py tried to find a download URL straight out
  of that response, which it never contains - that's why download.py existed
  as a separate manual step. This module wires the real flow together:

      upload  ->  getStatus (poll until done)  ->  getModelZip  ->  download
      ->  extract  ->  view

API endpoints used:
  POST https://api.kiriengine.app/api/v1/open/photo/video        (upload)
  GET  https://api.kiriengine.app/api/v1/open/model/getStatus    (poll)
  GET  https://api.kiriengine.app/api/v1/open/model/getModelZip  (get zip link)
Auth: Bearer token in Authorization header (loaded from scanner3d.config,
      NEVER hardcoded - see note below).

Usage:
    python -m scanner3d.kiri_pipeline --session 20260612_103734
    python -m scanner3d.kiri_pipeline --auto                      (latest session, full pipeline)
    python -m scanner3d.kiri_pipeline --serialize <id>             (resume: fetch+view an existing task)
    python -m scanner3d.kiri_pipeline --view-only                  (just view the latest local model)
    python -m scanner3d.kiri_pipeline --auto --no-view             (skip opening the viewer)
"""

import os
import sys
import time
import json
import zipfile
from pathlib import Path
from typing import Optional, Dict, Any, List

import requests

# Optional dependency - only needed for the "view" step
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NOTE: API key comes from scanner3d.config (KIRI_API_KEY), exactly like the
# original kiri_reconstructor.py. Never hardcode the key in this file - put
# it in config.py (or an environment variable that config.py reads) instead.
from scanner3d.config import OUTPUT_DIR, MODEL_OUTPUT_DIR, KIRI_API_KEY, KIRI_BASE_URL


KIRI_UPLOAD_URL = f"{KIRI_BASE_URL.rstrip('/')}/v1/open/photo/video"
KIRI_STATUS_URL = f"{KIRI_BASE_URL.rstrip('/')}/v1/open/model/getStatus"
KIRI_MODEL_ZIP_URL = f"{KIRI_BASE_URL.rstrip('/')}/v1/open/model/getModelZip"

# Status codes returned by getStatus (per Kiri Engine docs)
STATUS_MESSAGES = {
    -1: "Uploading",
    0: "Processing",
    1: "Failed",
    2: "Successful",
    3: "Queuing",
    4: "Expired",
}
STATUS_SUCCESS = 2
STATUS_TERMINAL_FAILURES = {1, 4}  # Failed, Expired - no point polling further

VIEWABLE_EXTENSIONS = (".obj", ".stl", ".glb", ".fbx", ".ply")


class KiriReconstructor:
    """
    End-to-end Kiri Engine client:
      - uploads a recorded video and starts cloud reconstruction
      - polls task status until the model is ready
      - downloads + extracts the resulting zipped model
      - optionally opens it in a trimesh viewer
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or KIRI_API_KEY
        self.session_name: str = ""
        self.serialize_id: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Step 0: find the recorded video (from kiri_reconstructor.py)
    # ------------------------------------------------------------------ #
    def find_video_file(self, session_dir: Path) -> Optional[str]:
        """
        Find the recorded video file in the session directory.

        Args:
            session_dir: Path to the session directory

        Returns:
            Path to the video file, or None if not found
        """
        video_files = list(session_dir.glob("*.avi"))
        if video_files:
            return str(video_files[0])

        for ext in ['.mp4', '.mov', '.mkv']:
            video_files = list(session_dir.glob(f"*{ext}"))
            if video_files:
                return str(video_files[0])

        return None

    # ------------------------------------------------------------------ #
    # Step 1: upload video, get back a "serialize" task ID
    # ------------------------------------------------------------------ #
    def upload_and_reconstruct(self, video_path: str,
                                model_quality: str = "1",
                                texture_quality: str = "1",
                                file_format: str = "OBJ",
                                is_mask: str = "1",
                                texture_smoothing: str = "1") -> Dict[str, Any]:
        """
        Upload a video to Kiri Engine to kick off cloud reconstruction.
        This call returns a task `serialize` ID immediately - the model
        itself is produced asynchronously (see wait_for_completion()).

        Args:
            video_path: Path to the recorded video file
            model_quality: Model quality ("1", "2", "3" where 1=high)
            texture_quality: Texture quality ("1", "2", "3" where 1=high)
            file_format: Output format ("OBJ", "STL", "GLB", "FBX", "PLY")
            is_mask: Whether to use background mask ("1"=yes, "0"=no)
            texture_smoothing: Texture smoothing ("1"=yes, "0"=no)

        Returns:
            Parsed JSON response, e.g. {"code":0,"data":{"serialize":"..."}}
        """
        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.session_name = video_path_obj.parent.name

        file_size_mb = video_path_obj.stat().st_size / (1024 * 1024)
        print(f"\n[Uploading Video to Kiri Engine]")
        print(f"  File: {video_path_obj.name}")
        print(f"  Size: {file_size_mb:.1f} MB")
        print(f"  Format: {file_format}")
        print(f"  Model Quality: {model_quality}")
        print(f"  Texture Quality: {texture_quality}")
        print()

        with open(video_path, 'rb') as f:
            files = {
                'videoFile': (video_path_obj.name, f, 'video/avi')
            }
            data = {
                'modelQuality': model_quality,
                'textureQuality': texture_quality,
                'fileFormat': file_format,
                'isMask': is_mask,
                'textureSmoothing': texture_smoothing,
            }
            headers = {
                'Authorization': f'Bearer {self.api_key}',
            }

            print("  Uploading... Kiri will queue the video for processing.")
            start_time = time.time()

            response = requests.post(
                KIRI_UPLOAD_URL,
                headers=headers,
                files=files,
                data=data,
                timeout=1800  # 30 minute timeout for large uploads
            )

        elapsed = time.time() - start_time
        print(f"  Upload response time: {elapsed:.0f}s")

        if response.status_code >= 400:
            error_msg = f"Kiri API Error {response.status_code}: "
            try:
                error_msg += json.dumps(response.json(), indent=2)
            except Exception:
                error_msg += response.text[:1000]
            raise RuntimeError(error_msg)

        result = response.json()
        self.serialize_id = result.get("data", {}).get("serialize")

        if self.serialize_id:
            print(f"  ✓ Upload accepted. Task serialize ID: {self.serialize_id}")
        else:
            print(f"  ⚠ No serialize ID found in response: {json.dumps(result, indent=2)[:1000]}")

        return result

    # ------------------------------------------------------------------ #
    # Step 2: poll task status until the model is ready
    # ------------------------------------------------------------------ #
    def get_status(self, serialize_id: str = None) -> Dict[str, Any]:
        """Query the current processing status for a task."""
        serialize_id = serialize_id or self.serialize_id
        if not serialize_id:
            raise ValueError("No serialize ID available - upload a video first or pass one explicitly.")

        headers = {'Authorization': f'Bearer {self.api_key}'}
        params = {'serialize': serialize_id}

        response = requests.get(KIRI_STATUS_URL, headers=headers, params=params, timeout=60)
        if response.status_code >= 400:
            raise RuntimeError(f"Kiri API Error {response.status_code}: {response.text[:500]}")

        return response.json()

    def wait_for_completion(self, serialize_id: str = None,
                             poll_interval: int = 15,
                             max_wait: int = 1800) -> bool:
        """
        Poll getStatus until the model finishes processing (status 2), or
        a terminal failure/expiry occurs, or max_wait seconds elapse.

        Returns:
            True if the model finished successfully, False otherwise.
        """
        serialize_id = serialize_id or self.serialize_id
        print(f"\n[Waiting for Kiri Engine to finish processing: {serialize_id}]")

        start = time.time()
        while True:
            try:
                status_resp = self.get_status(serialize_id)
                status = status_resp.get("data", {}).get("status")
            except Exception as e:
                print(f"  ⚠ Status check failed: {e} (retrying)")
                status = None

            label = STATUS_MESSAGES.get(status, f"Unknown ({status})")
            elapsed = time.time() - start
            print(f"  [{elapsed:5.0f}s] Status: {label}")

            if status == STATUS_SUCCESS:
                print("  ✓ Model is ready.")
                return True
            if status in STATUS_TERMINAL_FAILURES:
                print(f"  ✗ Reconstruction did not complete ({label}).")
                return False

            if elapsed > max_wait:
                print(f"  ✗ Timed out after {max_wait}s while waiting for completion.")
                return False

            time.sleep(poll_interval)

    # ------------------------------------------------------------------ #
    # Step 3: fetch the download link + download/extract the zip
    # (merged from download.py)
    # ------------------------------------------------------------------ #
    def fetch_and_extract_model(self, serialize_id: str = None,
                                 output_dir: str = None,
                                 max_retries: int = 10,
                                 retry_interval: int = 15) -> Dict[str, str]:
        """
        Fetch the zipped-model download link from getModelZip, download it,
        and extract it. The link is only valid once Kiri reports success,
        so this retries a few times in case modelUrl == "xxx" (not ready yet).

        Args:
            serialize_id: Task ID. Defaults to self.serialize_id.
            output_dir: Where to save/extract the model. Defaults to
                        MODEL_OUTPUT_DIR from config.
            max_retries: How many times to retry if the link isn't ready yet.
            retry_interval: Seconds to wait between retries.

        Returns:
            Dict of {extension: path} for extracted model files, e.g.
            {"obj": "/path/to/model_xxx/model.obj"}. Empty dict on failure.
        """
        serialize_id = serialize_id or self.serialize_id
        if not serialize_id:
            raise ValueError("No serialize ID available - upload a video first or pass one explicitly.")

        output_path = Path(output_dir or MODEL_OUTPUT_DIR)
        output_path.mkdir(parents=True, exist_ok=True)

        headers = {'Authorization': f'Bearer {self.api_key}'}
        params = {'serialize': serialize_id}

        print(f"\n[Downloading Model]")
        print(f"  Requesting download URL for task: {serialize_id}...")

        model_url = None
        for attempt in range(1, max_retries + 1):
            response = requests.get(KIRI_MODEL_ZIP_URL, headers=headers, params=params, timeout=60)
            if response.status_code >= 400:
                raise RuntimeError(f"Kiri API Error {response.status_code}: {response.text[:500]}")

            res_data = response.json()
            candidate = res_data.get("data", {}).get("modelUrl")

            if candidate and candidate != "xxx" and candidate.startswith("http"):
                model_url = candidate
                break

            print(f"  [Status] Model still generating on Kiri's cloud "
                  f"(attempt {attempt}/{max_retries}). Retrying in {retry_interval}s...")
            time.sleep(retry_interval)

        if not model_url:
            print("\n  ⚠ Could not obtain a model download link after retries.")
            print("  Please wait a bit longer and try fetch_and_extract_model() again.")
            return {}

        zip_filename = output_path / f"{serialize_id}.zip"
        extract_folder = output_path / f"model_{serialize_id}"

        print(f"  Link obtained! Downloading archive to {zip_filename}...")
        with requests.get(model_url, stream=True, timeout=600) as file_stream:
            file_stream.raise_for_status()
            with open(zip_filename, 'wb') as f:
                for chunk in file_stream.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        print(f"  Download complete. Extracting files to {extract_folder}...")
        extract_folder.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)

        # Clean up the zip file after extraction
        zip_filename.unlink()

        print(f"\n[Success] Your 3D files are ready in: {extract_folder.resolve()}")

        extracted = {}
        for f in extract_folder.iterdir():
            if f.suffix.lower() in VIEWABLE_EXTENSIONS:
                extracted[f.suffix.lower().lstrip('.')] = str(f)
                print(f"  [{f.suffix.lower().lstrip('.')}] {f}")

        return extracted

    # ------------------------------------------------------------------ #
    # Step 4: view the result (merged from view.py)
    # ------------------------------------------------------------------ #
    def _find_model_dirs(self, search_dirs: List[Path]) -> List[Path]:
        """Find all 'model_*' directories across the given search paths."""
        model_dirs = []
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for item in search_dir.iterdir():
                if item.is_dir() and item.name.startswith("model_"):
                    model_dirs.append(item)
        return model_dirs

    def view_model(self, model_path: str) -> bool:
        """
        Load and display a single model file (or a directory containing
        one) with trimesh.

        Args:
            model_path: Path to an .obj/.stl file, or a directory containing one.

        Returns:
            True if the viewer was opened successfully.
        """
        if not TRIMESH_AVAILABLE:
            print("  ⚠ trimesh is not installed - run `pip install trimesh` to enable viewing.")
            return False

        path = Path(model_path)
        if path.is_dir():
            candidates = [f for f in path.iterdir() if f.suffix.lower() in (".obj", ".stl")]
            if not candidates:
                print(f"  No OBJ/STL files found in {path}")
                return False
            path = candidates[0]

        print(f"  Loading: {path}")
        try:
            mesh = trimesh.load(str(path))
            print(f"  Vertices: {len(mesh.vertices)}")
            print(f"  Faces: {len(mesh.faces)}")
            print("\n  Opening 3D viewer...")
            print("  Controls:")
            print("    - Left mouse: Rotate")
            print("    - Right mouse: Pan")
            print("    - Scroll: Zoom")
            print("    - Close window to exit")
            mesh.show()
            return True
        except Exception as e:
            print(f"  Error loading model: {e}")
            return False

    def view_latest_model(self, extra_dirs: List[str] = None) -> bool:
        """
        Find and view the most recently modified downloaded model,
        searching the script's root dir, a local 'models/' folder, and
        MODEL_OUTPUT_DIR (where fetch_and_extract_model() saves results).
        """
        base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        search_dirs = [base_dir, base_dir / "models", Path(MODEL_OUTPUT_DIR)]
        if extra_dirs:
            search_dirs.extend(Path(d) for d in extra_dirs)

        model_dirs = self._find_model_dirs(search_dirs)
        if not model_dirs:
            print("No models found. Complete a scan and reconstruction first.")
            return False

        model_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        latest_dir = model_dirs[0]
        print(f"Latest model: {latest_dir}")
        return self.view_model(str(latest_dir))

    # ------------------------------------------------------------------ #
    # Full pipeline: upload -> wait -> download -> view
    # ------------------------------------------------------------------ #
    def run(self, session_dir: str,
            output_name: str = None,
            file_format: str = "OBJ",
            model_quality: str = "1",
            texture_quality: str = "1",
            poll_interval: int = 15,
            max_wait: int = 1800,
            auto_view: bool = True) -> bool:
        """
        Run the full Kiri Engine pipeline end-to-end: upload video, wait
        for cloud reconstruction, download + extract the model, then
        (optionally) open it in the trimesh viewer.

        Args:
            session_dir: Directory containing the captured video
            output_name: Custom name for the output model (kept for compat, unused)
            file_format: Output format ("OBJ", "STL", "GLB", "FBX", "PLY")
            model_quality: Model quality ("1", "2", "3")
            texture_quality: Texture quality ("1", "2", "3")
            poll_interval: Seconds between status checks
            max_wait: Max seconds to wait for processing before giving up
            auto_view: If True, opens the trimesh viewer once the model downloads

        Returns:
            True if reconstruction + download succeeded.
        """
        print("=" * 60)
        print("KIRI ENGINE 3D RECONSTRUCTION PIPELINE")
        print("=" * 60)
        print(f"  Upload endpoint: {KIRI_UPLOAD_URL}")
        print(f"  Session: {session_dir}")
        print("=" * 60)

        session_path = Path(session_dir)
        video_path = self.find_video_file(session_path)

        if not video_path:
            print(f"\n✗ No video file found in {session_dir}")
            print("  Run a scan first (python main.py) to capture video.")
            return False

        print(f"\n[1/4] Uploading video to Kiri Engine...")
        try:
            self.upload_and_reconstruct(
                video_path=video_path,
                model_quality=model_quality,
                texture_quality=texture_quality,
                file_format=file_format,
                is_mask="1",
                texture_smoothing="1"
            )
        except Exception as e:
            print(f"✗ Upload failed: {e}")
            return False

        if not self.serialize_id:
            print("✗ No task ID returned by Kiri Engine - cannot continue.")
            return False

        print(f"\n[2/4] Waiting for cloud reconstruction to finish...")
        if not self.wait_for_completion(poll_interval=poll_interval, max_wait=max_wait):
            print(f"\n  You can resume later with: --serialize {self.serialize_id}")
            return False

        print(f"\n[3/4] Downloading reconstructed 3D model...")
        downloaded = self.fetch_and_extract_model()

        if not downloaded:
            print("\n✗ Model download failed.")
            print(f"  Resume later with: --serialize {self.serialize_id}")
            return False

        print("\n" + "=" * 60)
        print("RECONSTRUCTION COMPLETE!")
        print("=" * 60)
        for fmt, path in downloaded.items():
            print(f"  [{fmt}] {path}")

        if auto_view:
            print(f"\n[4/4] Opening viewer...")
            self.view_model(str(Path(next(iter(downloaded.values()))).parent))
        else:
            print("\n  Open in any 3D viewer or slicer (or run with --view-only).")

        return True


# -------------------------------------------------------------------------- #
# Module-level helpers (kept from the original kiri_reconstructor.py)
# -------------------------------------------------------------------------- #
def find_latest_session() -> Optional[str]:
    """Find the most recent capture session."""
    captures_dir = Path(OUTPUT_DIR)
    if not captures_dir.exists():
        return None

    sessions = sorted([
        d for d in captures_dir.iterdir()
        if d.is_dir() and d.name.startswith("20")
    ], reverse=True)

    return str(sessions[0]) if sessions else None


def reconstruct(session_id: str = None, output_name: str = None,
                 quality: str = "high",
                 file_format: str = "OBJ",
                 auto_view: bool = True,
                 progress_callback=None):
    """
    Run the full Kiri Engine pipeline on a captured session's video.

    Args:
        session_id: Session folder name (e.g., "20260612_103734").
                    If None, uses latest session.
        output_name: Custom name for the output model.
        quality: Reconstruction quality ("draft", "medium", "high", "ultra")
        file_format: Output format ("OBJ", "STL", "GLB", "FBX", "PLY")
        auto_view: Whether to open the trimesh viewer once done.
        progress_callback: Optional callback function(progress_percent) for UI updates
    """
    if session_id:
        session_dir = Path(OUTPUT_DIR) / session_id
    else:
        latest = find_latest_session()
        if not latest:
            print("✗ No capture sessions found.")
            print("  Run `python main.py` first to capture images.")
            return False
        session_dir = Path(latest)
        session_id = session_dir.name
        print(f"  Using latest session: {session_id}")

    if not session_dir.exists():
        print(f"✗ Session not found: {session_dir}")
        return False

    quality_map = {"draft": "3", "medium": "2", "high": "1", "ultra": "1"}
    model_quality = quality_map.get(quality, "1")
    texture_quality = model_quality

    recon = KiriReconstructor()
    if progress_callback:
        progress_callback(10)

    result = recon.run(
        str(session_dir),
        output_name=output_name,
        file_format=file_format,
        model_quality=model_quality,
        texture_quality=texture_quality,
        auto_view=auto_view,
    )

    if progress_callback:
        progress_callback(100)

    return result


def main():
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Kiri Engine 3D Reconstruction Pipeline - upload, poll, download, and view a 3D model"
    )
    parser.add_argument("--session", type=str, default=None,
                         help="Session folder name (uses latest if not specified)")
    parser.add_argument("--auto", action="store_true",
                         help="Shortcut for using the latest session")
    parser.add_argument("--name", type=str, default=None,
                         help="Custom output model name")
    parser.add_argument("--quality", type=str, default="high",
                         choices=["draft", "medium", "high", "ultra"],
                         help="Reconstruction quality (default: high)")
    parser.add_argument("--format", type=str, default="OBJ",
                         choices=["OBJ", "STL", "GLB", "FBX", "PLY"],
                         help="Output 3D format (default: OBJ)")
    parser.add_argument("--serialize", type=str, default=None,
                         help="Resume an existing task: fetch + extract + view by serialize ID "
                              "(skips upload, equivalent to the old standalone download.py)")
    parser.add_argument("--view-only", action="store_true",
                         help="Just open the viewer on the latest local model "
                              "(equivalent to the old standalone view.py)")
    parser.add_argument("--no-view", action="store_true",
                         help="Don't open the trimesh viewer automatically")
    parser.add_argument("--list-sessions", action="store_true",
                         help="List all available sessions")

    args = parser.parse_args()

    if args.list_sessions:
        print("Available sessions:")
        captures_dir = Path(OUTPUT_DIR)
        if captures_dir.exists():
            sessions = sorted([d for d in captures_dir.iterdir()
                                if d.is_dir() and d.name.startswith("20")], reverse=True)
            for s in sessions:
                images = list(s.glob("*.jpg")) + list(s.glob("*.png"))
                videos = list(s.glob("*.avi")) + list(s.glob("*.mp4"))
                print(f"  {s.name} ({len(images)} images, {len(videos)} videos)")
        else:
            print("  No sessions found.")
        return

    if args.view_only:
        recon = KiriReconstructor()
        success = recon.view_latest_model()
        sys.exit(0 if success else 1)

    if args.serialize:
        recon = KiriReconstructor()
        recon.serialize_id = args.serialize
        downloaded = recon.fetch_and_extract_model()
        if downloaded and not args.no_view:
            recon.view_model(str(Path(next(iter(downloaded.values()))).parent))
        sys.exit(0 if downloaded else 1)

    session_id = None if args.auto else args.session
    success = reconstruct(session_id, args.name, args.quality,
                           file_format=args.format,
                           auto_view=not args.no_view)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()