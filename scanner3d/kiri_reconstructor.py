"""
Kiri Engine 3D Reconstruction Module
======================================
Uploads recorded video directly to Kiri Engine API for cloud-based 3D reconstruction.
No local GPU or COLMAP needed - reconstruction happens on Kiri's servers.

API Endpoint: POST https://api.kiriengine.app/api/v1/open/photo/video
Auth: Bearer token in Authorization header

Pipeline:
  1. Find the recorded video (.avi) from a capture session
  2. Upload it directly to Kiri Engine (multipart form)
  3. Kiri processes and returns download URL for the 3D model
  4. Download and save the resulting .obj/.stl/.glb model

Usage:
    python -m scanner3d.kiri_reconstructor --session 20260612_103734
    python -m scanner3d.kiri_reconstructor --auto       (latest session)
"""

import os
import sys
import time
import json
import requests
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scanner3d.config import OUTPUT_DIR, MODEL_OUTPUT_DIR, KIRI_API_KEY, KIRI_BASE_URL


KIRI_RECONSTRUCT_URL = f"{KIRI_BASE_URL.rstrip('/')}/v1/open/photo/video"


class KiriReconstructor:
    """
    Uploads recorded video to Kiri Engine API and downloads the 3D model result.
    Uses the /v1/open/photo/video endpoint which accepts a video file directly.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or KIRI_API_KEY
        self.session_name: str = ""

    def find_video_file(self, session_dir: Path) -> Optional[str]:
        """
        Find the recorded video file (.avi) in the session directory.
        
        Args:
            session_dir: Path to the session directory
            
        Returns:
            Path to the video file, or None if not found
        """
        # Look for .avi files
        video_files = list(session_dir.glob("*.avi"))
        if video_files:
            return str(video_files[0])
        
        # Also check for .mp4 or .mov
        for ext in ['.mp4', '.mov', '.mkv']:
            video_files = list(session_dir.glob(f"*{ext}"))
            if video_files:
                return str(video_files[0])
        
        return None

    def upload_and_reconstruct(self, video_path: str,
                               model_quality: str = "1",
                               texture_quality: str = "1",
                               file_format: str = "OBJ",
                               is_mask: str = "1",
                               texture_smoothing: str = "1") -> Dict[str, Any]:
        """
        Upload a video to Kiri Engine and trigger 3D reconstruction.
        This is a single request that uploads + processes in one call.
        
        Args:
            video_path: Path to the .avi video file
            model_quality: Model quality ("1", "2", "3" where 1=high)
            texture_quality: Texture quality ("1", "2", "3" where 1=high)
            file_format: Output format ("OBJ", "STL", "GLB", "FBX", "PLY")
            is_mask: Whether to use background mask ("1"=yes, "0"=no)
            texture_smoothing: Texture smoothing ("1"=yes, "0"=no)
            
        Returns:
            API response containing model download URL(s)
        """
        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Store session name from the parent directory of the video
        self.session_name = video_path_obj.parent.name

        file_size_mb = video_path_obj.stat().st_size / (1024 * 1024)
        print(f"\n[Uploading Video to Kiri Engine]")
        print(f"  File: {video_path_obj.name}")
        print(f"  Size: {file_size_mb:.1f} MB")
        print(f"  Format: {file_format}")
        print(f"  Model Quality: {model_quality}")
        print(f"  Texture Quality: {texture_quality}")
        print()

        # Prepare multipart form data matching the curl command format
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

            print("  Uploading and processing... (this may take several minutes)")
            print("  Kiri Engine will process the video and generate a 3D model.")
            print()
            
            start_time = time.time()
            
            response = requests.post(
                KIRI_RECONSTRUCT_URL,
                headers=headers,
                files=files,
                data=data,
                timeout=1800  # 30 minute timeout for large uploads+processing
            )

        elapsed = time.time() - start_time
        print(f"  Response time: {elapsed:.0f}s")

        if response.status_code >= 400:
            error_msg = f"Kiri API Error {response.status_code}: "
            try:
                error_data = response.json()
                error_msg += json.dumps(error_data, indent=2)
            except:
                error_msg += response.text[:1000]
            raise RuntimeError(error_msg)

        try:
            result = response.json()
        except:
            result = {"raw": response.text}

        print(f"  Response received successfully")
        return result

    def download_model(self, api_response: Dict[str, Any],
                       output_dir: str = None) -> Dict[str, str]:
        """
        Download the reconstructed 3D model from the Kiri Engine response.
        
        Args:
            api_response: The JSON response from the upload endpoint
            output_dir: Directory to save the model files
            
        Returns:
            Dictionary of format -> file path for downloaded models
        """
        output_path = Path(output_dir or MODEL_OUTPUT_DIR)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n[Downloading Model]")
        print(f"  Output: {output_path}")

        # Find model download URL(s) in the response
        model_urls = {}

        # Check various possible response formats
        if isinstance(api_response, dict):
            # Direct URL
            for key in ['url', 'modelUrl', 'model_url', 'downloadUrl', 'download_url', 'fileUrl', 'file_url']:
                if key in api_response and api_response[key]:
                    val = api_response[key]
                    if isinstance(val, str) and val.startswith('http'):
                        model_urls['model'] = val

            # Nested in 'data' field
            if 'data' in api_response and isinstance(api_response['data'], dict):
                data = api_response['data']
                for key in ['url', 'modelUrl', 'model_url', 'downloadUrl', 'download_url']:
                    if key in data and data[key]:
                        val = data[key]
                        if isinstance(val, str) and val.startswith('http'):
                            model_urls['model'] = val

            # List of URLs
            if 'urls' in api_response and isinstance(api_response['urls'], list):
                for url in api_response['urls']:
                    if isinstance(url, str) and url.startswith('http'):
                        # Determine format from extension
                        ext = Path(url).suffix.lower().lstrip('.') or 'model'
                        model_urls[ext] = url

            # Multiple format URLs in result
            if 'result' in api_response and isinstance(api_response['result'], dict):
                for fmt_key, url in api_response['result'].items():
                    if isinstance(url, str) and url.startswith('http'):
                        model_urls[fmt_key] = url

        if not model_urls:
            # Print the full response for debugging
            print("  ⚠ Could not find model download URL in response.")
            print(f"  Response: {json.dumps(api_response, indent=2)[:2000]}")
            print("\n  You may be able to download the model manually.")
            return {}

        downloaded = {}
        for fmt, url in model_urls.items():
            print(f"  Downloading ({fmt})...", end=" ", flush=True)
            try:
                resp = requests.get(url, stream=True, timeout=600)
                resp.raise_for_status()

                # Determine filename
                content_disp = resp.headers.get('Content-Disposition', '')
                if 'filename=' in content_disp:
                    filename = content_disp.split('filename=')[-1].strip('"\'')
                else:
                    ext = Path(url).suffix or '.obj'
                    filename = f"{self.session_name or 'scan'}_3d_model{ext}"

                filepath = output_path / filename

                with open(filepath, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                size_mb = filepath.stat().st_size / (1024 * 1024)
                print(f"✓ ({size_mb:.1f} MB)")
                downloaded[fmt] = str(filepath)

            except Exception as e:
                print(f"✗ Failed: {e}")

        if downloaded:
            print(f"\n✓ Model downloaded successfully!")
            for fmt, path in downloaded.items():
                print(f"  [{fmt}] {path}")

        return downloaded

    def run(self, session_dir: str, output_name: str = None,
            file_format: str = "OBJ",
            model_quality: str = "1",
            texture_quality: str = "1") -> bool:
        """
        Run the full Kiri Engine reconstruction pipeline.
        Uploads video, gets 3D model back.

        Args:
            session_dir: Directory containing captured video
            output_name: Custom name for the output model (unused, kept for compat)
            file_format: Output format ("OBJ", "STL", "GLB", "FBX", "PLY")
            model_quality: Model quality ("1", "2", "3")
            texture_quality: Texture quality ("1", "2", "3")

        Returns:
            True if reconstruction succeeded
        """
        print("=" * 60)
        print("KIRI ENGINE 3D RECONSTRUCTION")
        print("=" * 60)
        print(f"  Endpoint: {KIRI_RECONSTRUCT_URL}")
        print(f"  Session: {session_dir}")
        print("=" * 60)

        session_path = Path(session_dir)
        video_path = self.find_video_file(session_path)

        if not video_path:
            print(f"\n✗ No video file found in {session_dir}")
            print("  Run a scan first (python main.py) to capture video.")
            return False

        print(f"\n[1/2] Uploading video to Kiri Engine...")
        try:
            api_response = self.upload_and_reconstruct(
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

        print(f"\n[2/2] Downloading reconstructed 3D model...")
        downloaded = self.download_model(api_response)

        if downloaded:
            print("\n" + "=" * 60)
            print("RECONSTRUCTION COMPLETE!")
            print("=" * 60)
            for fmt, path in downloaded.items():
                print(f"  [{fmt}] {path}")
            print("\n  Open in any 3D viewer or slicer.")
            return True
        else:
            print("\n✗ Model download failed.")
            print(f"  Check Kiri Engine dashboard for job status.")
            return False


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
                poll_interval: int = 15):
    """
    Run Kiri Engine reconstruction on a captured session's video.

    Args:
        session_id: Session folder name (e.g., "20260612_103734").
                    If None, uses latest session.
        output_name: Custom name for the output model.
        quality: Reconstruction quality ("draft", "medium", "high", "ultra")
        poll_interval: Kept for backward compatibility (unused in new API)
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

    # Map quality string to Kiri quality value
    quality_map = {
        "draft": "3",
        "medium": "2",
        "high": "1",
        "ultra": "1",
    }
    model_quality = quality_map.get(quality, "1")
    texture_quality = model_quality

    # Run reconstruction via Kiri Engine
    recon = KiriReconstructor()
    return recon.run(
        str(session_dir),
        output_name=output_name,
        file_format="OBJ",
        model_quality=model_quality,
        texture_quality=texture_quality
    )


def main():
    """Command-line entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Kiri Engine 3D Reconstruction - Generate 3D models from video via cloud API"
    )
    parser.add_argument("--session", type=str, default=None,
                        help="Session folder name (uses latest if not specified)")
    parser.add_argument("--name", type=str, default=None,
                        help="Custom output model name")
    parser.add_argument("--quality", type=str, default="high",
                        choices=["draft", "medium", "high", "ultra"],
                        help="Reconstruction quality (default: high)")
    parser.add_argument("--format", type=str, default="OBJ",
                        choices=["OBJ", "STL", "GLB", "FBX", "PLY"],
                        help="Output 3D format (default: OBJ)")
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
    
    success = reconstruct(args.session, args.name, args.quality)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()