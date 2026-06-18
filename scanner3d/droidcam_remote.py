"""
IP Webcam / DroidCam Remote Control Module.

Controls your phone camera over WiFi using the HTTP API.
Supports both IP Webcam and DroidCam apps.

IP Webcam API (default):
  - GET /photo.jpg - Snapshot
  - GET /video - MJPEG stream
  - GET /focus - Focus control
  - GET /torch - Torch/LED control
  - GET /settings - Camera settings
  - GET /info - Device info
  - GET /status - Status info

Usage:
    from scanner3d.droidcam_remote import PhoneRemote
    cam = PhoneRemote("10.138.159.186", port=8080)
    info = cam.get_info()
"""

import urllib.request
import json
import time
from typing import Optional, Dict, Any


class PhoneRemoteError(Exception):
    """Custom exception for phone camera API errors."""
    pass


class PhoneRemote:
    """Remote control for phone camera app (IP Webcam / DroidCam) over WiFi."""

    def __init__(self, host: str = "10.138.159.186", port: int = 8080,
                 timeout: float = 3.0, app: str = "ipwebcam"):
        """
        Initialize phone camera remote control.

        Args:
            host: IP address of the phone
            port: HTTP port (IP Webcam default: 8080, DroidCam: 4747)
            timeout: HTTP request timeout
            app: "ipwebcam" or "droidcam"
        """
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.app = app

    def _get(self, endpoint: str) -> Any:
        """Send a GET request to the phone API."""
        url = self.base_url + endpoint
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                if body.strip():
                    try:
                        return json.loads(body)
                    except json.JSONDecodeError:
                        return body
                return None
        except urllib.request.HTTPError as e:
            raise PhoneRemoteError(f"HTTP {e.code}: {e.reason}")
        except urllib.request.URLError as e:
            raise PhoneRemoteError(f"Connection failed: {e.reason}")

    def ping(self) -> bool:
        """Check if the phone camera server is reachable."""
        try:
            req = urllib.request.Request(self.base_url + "/", method="GET")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.status == 200
        except Exception:
            return False

    def get_info(self) -> Optional[Dict[str, Any]]:
        """Get camera/device information."""
        try:
            return self._get("/info")
        except PhoneRemoteError:
            return None

    def get_status(self) -> Optional[Dict[str, Any]]:
        """Get camera status."""
        try:
            return self._get("/status")
        except PhoneRemoteError:
            return None

    def get_settings(self) -> Optional[Dict[str, Any]]:
        """Get camera settings."""
        try:
            return self._get("/settings")
        except PhoneRemoteError:
            return None

    def get_phone_name(self) -> str:
        """Get phone device name."""
        try:
            info = self.get_info()
            if info and isinstance(info, dict):
                return info.get("device", info.get("model", "Unknown"))
            return str(info) if info else "Unknown"
        except Exception:
            return "Unknown"

    def get_battery_info(self) -> Optional[Dict[str, Any]]:
        """Get battery information."""
        try:
            info = self.get_info()
            if info and isinstance(info, dict):
                battery = info.get("battery", {})
                if isinstance(battery, dict):
                    return battery
                return {"level": battery}
            return None
        except Exception:
            return None

    def get_status_text(self) -> str:
        """Get a human-readable status summary."""
        parts = []
        try:
            name = self.get_phone_name()
            parts.append(f"Phone: {name}")
        except Exception:
            parts.append("Phone: Unknown")

        try:
            info = self.get_info()
            if info and isinstance(info, dict):
                res = info.get("resolution", info.get("preview_size", "?"))
                parts.append(f"Res: {res}")
        except Exception:
            pass

        try:
            bat = self.get_battery_info()
            if bat and isinstance(bat, dict):
                parts.append(f"Battery: {bat.get('level', '?')}%")
        except Exception:
            pass

        return " | ".join(parts)

    def snapshot_url(self) -> str:
        """Get URL for a high-res snapshot."""
        return f"{self.base_url}/photo.jpg"

    def stream_url(self) -> str:
        """Get URL for the MJPEG video stream."""
        return f"{self.base_url}/video"

    def __repr__(self) -> str:
        return f"<PhoneRemote {self.base_url}>"


# Quick test
if __name__ == "__main__":
    import sys
    host = sys.argv[1] if len(sys.argv) > 1 else "10.138.159.186"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080

    cam = PhoneRemote(host, port)
    print("=" * 60)
    print(f"Phone Remote Control - {cam.base_url}")
    print("=" * 60)

    if cam.ping():
        print("[✓] Phone camera is reachable!\n")
    else:
        print("[✗] Cannot reach phone camera.")
        sys.exit(1)

    print(f"  Status: {cam.get_status_text()}")
    print(f"  Stream: {cam.stream_url()}")
    print(f"  Snapshot: {cam.snapshot_url()}")
    print("\nDone.")