"""
3D Scanner Pro v3.0 - Professional GUI Application
===================================================
A modern, professional desktop interface integrating:
  - Live camera preview (IP Webcam / DroidCam)
  - ArUco marker-based dimension tracking
  - LiveTracker3D ORB feature + point cloud visualization
  - Background removal (CPU-optimized)
  - Video recording + frame extraction
  - Kiri Engine cloud 3D reconstruction (with file format selection)
  - trimesh-based 3D model viewer
"""

import cv2
import numpy as np
import os
import sys
import time
import json
import threading
import queue
import urllib.request
from pathlib import Path
from tkinter import (
    Tk, Toplevel, Frame, Label, Button, Canvas, Entry,
    StringVar, IntVar, BooleanVar, ttk, messagebox, OptionMenu
)
from PIL import Image, ImageTk
from typing import Optional, List, Tuple, Callable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scanner3d.config import (
    OUTPUT_DIR, MODEL_OUTPUT_DIR, NUM_ANGLES,
    KIRI_API_KEY, KIRI_BASE_URL,
    IP_WEBCAM_URL, IP_WEBCAM_STREAM_PATH
)
from scanner3d.kiri_reconstructor import reconstruct as run_reconstruction
from scanner3d.kiri_reconstructor import KiriReconstructor, find_latest_session
from scanner3d.segmentation import BackgroundRemover
from scanner3d.aruco_tracker import ArUcoDimensionTracker

# ─── Configuration ───────────────────────────────────────────────────────────
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui_config.json")

DEFAULT_CONFIG = {
    "webcam_url": IP_WEBCAM_URL,
    "kiri_api_key": KIRI_API_KEY,
    "num_angles": 70,
    "recording_duration": 30,
    "theme": "dark",
    "bg_removal": False,
    "output_format": "OBJ",
    "recon_quality": "ultra",
    "marker_size_cm": 5.0,
    "aruco_enabled": True,
    "video_quality": 100,
}


def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        except (json.JSONDecodeError, IOError):
            pass
    return dict(DEFAULT_CONFIG)


def save_config(config: dict):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
    except IOError as e:
        print(f"[Config] Save error: {e}")


# ─── Professional Theme ──────────────────────────────────────────────────────
class ProfessionalTheme:
    """Modern professional color scheme."""

    DARK = {
        "bg_primary": "#0f1419",
        "bg_secondary": "#1a1f26",
        "bg_tertiary": "#242b33",
        "bg_hover": "#2d353f",
        "accent_primary": "#00b4d8",
        "accent_secondary": "#7c3aed",
        "accent_success": "#10b981",
        "accent_warning": "#f59e0b",
        "accent_danger": "#ef4444",
        "text_primary": "#f1f5f9",
        "text_secondary": "#94a3b8",
        "text_tertiary": "#64748b",
        "border_light": "#334155",
        "border_medium": "#475569",
    }

    LIGHT = {
        "bg_primary": "#f8fafc",
        "bg_secondary": "#ffffff",
        "bg_tertiary": "#f1f5f9",
        "bg_hover": "#e2e8f0",
        "accent_primary": "#0284c7",
        "accent_secondary": "#7c3aed",
        "accent_success": "#059669",
        "accent_warning": "#d97706",
        "accent_danger": "#dc2626",
        "text_primary": "#0f172a",
        "text_secondary": "#475569",
        "text_tertiary": "#64748b",
        "border_light": "#e2e8f0",
        "border_medium": "#cbd5e1",
    }


# ─── Camera Stream ──────────────────────────────────────────────────────────
class CameraStream:
    """Background thread for IP Webcam MJPEG stream or local camera."""

    def __init__(self, source: str = None):
        """
        Args:
            source: URL string (e.g. http://10.x.x.x:8080/video)
                     or camera index for local camera.
        """
        self.source = source
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.thread: Optional[threading.Thread] = None
        self.source_name = "Disconnected"
        self._resolution = ""
        self._lock = threading.Lock()

    def open(self) -> bool:
        print(f"[Camera] Connecting: {self.source}")
        try:
            if self.source and self.source.startswith("http"):
                self.cap = cv2.VideoCapture(self.source)
            else:
                # Try local camera
                idx = int(self.source) if self.source else 0
                self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(idx)
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    self._resolution = f"{w}x{h}"
                    if self.source and self.source.startswith("http"):
                        self.source_name = f"IP Webcam ({self._resolution})"
                    else:
                        self.source_name = f"Camera {self.source or '0'} ({self._resolution})"
                    return True
                self.cap.release()
            self.cap = None
        except Exception as e:
            print(f"[Camera] Error: {e}")
        return False

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        with self._lock:
            if self.cap:
                self.cap.release()
                self.cap = None
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

    def read(self) -> Optional[np.ndarray]:
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def _capture_loop(self):
        while self.running:
            with self._lock:
                if self.cap:
                    ret, frame = self.cap.read()
                else:
                    ret, frame = False, None
            if ret and frame is not None:
                # Keep full resolution frame for recording quality
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(frame)
            else:
                time.sleep(0.02)

    def release(self):
        self.stop()

    @property
    def resolution(self):
        return self._resolution


# ─── Professional GUI ────────────────────────────────────────────────────────
class ScannerGUI:
    """Professional 3D Scanner Desktop Application."""

    def __init__(self):
        self.config = load_config()
        self.theme = ProfessionalTheme.DARK if self.config.get("theme", "dark") == "dark" else ProfessionalTheme.LIGHT

        # State
        self.camera_stream: Optional[CameraStream] = None
        self.camera_open = False
        self.scanning = False
        self._auto_connected = False
        self._recording = False
        self._record_start = 0.0
        self._record_frames: List[np.ndarray] = []
        self._session_dir: Optional[str] = None
        self._scan_duration = 30
        self._capture_mask: Optional[np.ndarray] = None
        self._capture_percentage = 0.0
        self._prev_features: Optional[np.ndarray] = None
        self._scan_complete_threshold = 85.0

        # Modules
        self.bg_remover = BackgroundRemover(enable=self.config.get("bg_removal", False))
        self.aruco_tracker = ArUcoDimensionTracker(
            marker_size_cm=self.config.get("marker_size_cm", 5.0),
            enable=self.config.get("aruco_enabled", True),
        )

        # Build UI
        self.root = Tk()
        self.root.title("3D Scanner Pro v3.0")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        self.root.configure(bg=self.theme["bg_primary"])

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(500, self._auto_connect)
        # Bind keyboard shortcuts
        self.root.bind("<Control-s>", lambda e: self._start_scan())
        self.root.bind("<Control-r>", lambda e: self._direct_reconstruct())
        self.root.bind("<Control-l>", lambda e: self._launch_live_tracker())
        self.root.bind("<Escape>", lambda e: self._on_close())

    def _build_ui(self):
        """Build professional UI layout."""

        # ── Top Navigation Bar ─────────────────────────────────────────
        nav = Frame(self.root, bg=self.theme["bg_secondary"], height=60)
        nav.pack(fill="x", side="top")
        nav.pack_propagate(False)

        logo_frame = Frame(nav, bg=self.theme["bg_secondary"])
        logo_frame.pack(side="left", padx=24, pady=0)

        Label(logo_frame, text="🎯", bg=self.theme["bg_secondary"],
              font=("Segoe UI", 20)).pack(side="left", padx=(0, 10))
        Label(logo_frame, text="3D Scanner Pro", bg=self.theme["bg_secondary"],
              fg=self.theme["text_primary"], font=("Segoe UI", 16, "bold")).pack(side="left")
        Label(logo_frame, text="v3.0", bg=self.theme["bg_secondary"],
              fg=self.theme["text_tertiary"], font=("Segoe UI", 9)).pack(side="left", padx=(4, 0))

        # Right side buttons
        nav_right = Frame(nav, bg=self.theme["bg_secondary"])
        nav_right.pack(side="right", padx=24)

        self.cam_status = Label(nav_right, text="● Offline", bg=self.theme["bg_secondary"],
                                fg=self.theme["accent_danger"], font=("Segoe UI", 10, "bold"))
        self.cam_status.pack(side="left", padx=(0, 16))

        self.theme_btn = Button(nav_right,
                                text="☀" if self.config.get("theme") == "dark" else "🌙",
                                command=self._toggle_theme,
                                bg=self.theme["bg_tertiary"], fg=self.theme["text_primary"],
                                relief="flat", bd=0, font=("Segoe UI", 12),
                                padx=10, pady=6, cursor="hand2")
        self.theme_btn.pack(side="left", padx=(0, 8))

        btn_settings = Button(nav_right, text="⚙ Settings", command=self._open_settings,
                              bg=self.theme["bg_tertiary"], fg=self.theme["text_primary"],
                              relief="flat", bd=0, font=("Segoe UI", 10),
                              padx=12, pady=6, cursor="hand2")
        btn_settings.pack(side="left", padx=(0, 8))

        self.recon_btn = Button(nav_right, text="🔧 Reconstruct",
                                command=self._direct_reconstruct,
                                bg=self.theme["accent_secondary"], fg="white",
                                relief="flat", bd=0, font=("Segoe UI", 10, "bold"),
                                padx=14, pady=6, cursor="hand2")
        self.recon_btn.pack(side="left")

        # ── Main Content Area ──────────────────────────────────────────
        content = Frame(self.root, bg=self.theme["bg_primary"])
        content.pack(fill="both", expand=True, padx=20, pady=20)

        # Left: Preview (70% width)
        preview_panel = Frame(content, bg=self.theme["bg_secondary"],
                              relief="flat", bd=0)
        preview_panel.pack(side="left", fill="both", expand=True, padx=(0, 16))

        preview_header = Frame(preview_panel, bg=self.theme["bg_secondary"], height=48)
        preview_header.pack(fill="x", padx=20, pady=(16, 0))
        preview_header.pack_propagate(False)

        Label(preview_header, text="📷 Live Preview", bg=self.theme["bg_secondary"],
              fg=self.theme["text_primary"], font=("Segoe UI", 13, "bold")).pack(side="left")

        self.res_label = Label(preview_header, text="—", bg=self.theme["bg_tertiary"],
                               fg=self.theme["text_secondary"], font=("Segoe UI", 9),
                               padx=8, pady=3)
        self.res_label.pack(side="right")

        canvas_container = Frame(preview_panel, bg="#000000",
                                 relief="flat", bd=1,
                                 highlightbackground=self.theme["border_light"],
                                 highlightthickness=1)
        canvas_container.pack(fill="both", expand=True, padx=16, pady=12)

        self.canvas = Canvas(canvas_container, bg="#000000", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        preview_footer = Frame(preview_panel, bg=self.theme["bg_secondary"], height=40)
        preview_footer.pack(fill="x", padx=20, pady=(0, 16))
        preview_footer.pack_propagate(False)

        self.status_label = Label(preview_footer, text="Ready", bg=self.theme["bg_secondary"],
                                  fg=self.theme["text_secondary"], font=("Segoe UI", 9))
        self.status_label.pack(side="left")

        self.server_label = Label(preview_footer,
                                  text=f"Server: {self.config.get('webcam_url', IP_WEBCAM_URL)}",
                                  bg=self.theme["bg_secondary"], fg=self.theme["text_tertiary"],
                                  font=("Segoe UI", 8))
        self.server_label.pack(side="right")

        # Right: Control Panel (30% width)
        control_panel = Frame(content, bg=self.theme["bg_secondary"],
                              width=320, relief="flat", bd=0)
        control_panel.pack(side="right", fill="y")
        control_panel.pack_propagate(False)

        ctrl_header = Frame(control_panel, bg=self.theme["bg_secondary"], height=48)
        ctrl_header.pack(fill="x", padx=20, pady=(16, 0))
        ctrl_header.pack_propagate(False)

        Label(ctrl_header, text="🎛 Control Panel", bg=self.theme["bg_secondary"],
              fg=self.theme["text_primary"], font=("Segoe UI", 13, "bold")).pack(side="left")

        # Scrollable control content
        ctrl_canvas = Canvas(control_panel, bg=self.theme["bg_secondary"], highlightthickness=0)
        ctrl_scrollbar = ttk.Scrollbar(control_panel, orient="vertical", command=ctrl_canvas.yview)
        ctrl_content = Frame(ctrl_canvas, bg=self.theme["bg_secondary"])

        ctrl_content.bind(
            "<Configure>",
            lambda e, c=ctrl_canvas: c.configure(scrollregion=c.bbox("all"))
        )

        ctrl_canvas.create_window((0, 0), window=ctrl_content, anchor="nw", width=300)
        ctrl_canvas.configure(yscrollcommand=ctrl_scrollbar.set)

        ctrl_canvas.pack(side="left", fill="both", expand=True, padx=(20, 0), pady=12)
        ctrl_scrollbar.pack(side="right", fill="y", pady=12)

        # Mouse wheel for control panel
        def _on_ctrl_mousewheel(event):
            ctrl_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        ctrl_canvas.bind("<Enter>", lambda e: ctrl_canvas.bind_all("<MouseWheel>", _on_ctrl_mousewheel))
        ctrl_canvas.bind("<Leave>", lambda e: ctrl_canvas.unbind_all("<MouseWheel>"))

        # ── Device Section ──
        self._section_label(ctrl_content, "📱 Device")
        device_frame = Frame(ctrl_content, bg=self.theme["bg_tertiary"],
                             relief="flat", bd=0, padx=12, pady=10)
        device_frame.pack(fill="x", pady=(0, 12))

        Label(device_frame, text="Connection Status", bg=self.theme["bg_tertiary"],
              fg=self.theme["text_secondary"], font=("Segoe UI", 8, "bold")).pack(anchor="w")
        self.conn_status = Label(device_frame, text="Disconnected", bg=self.theme["bg_tertiary"],
                                 fg=self.theme["accent_danger"], font=("Segoe UI", 9))
        self.conn_status.pack(anchor="w", pady=(2, 4))

        Label(device_frame, text="Resolution", bg=self.theme["bg_tertiary"],
              fg=self.theme["text_secondary"], font=("Segoe UI", 8, "bold")).pack(anchor="w")
        self.resolution_value = Label(device_frame, text="—", bg=self.theme["bg_tertiary"],
                                      fg=self.theme["text_primary"], font=("Segoe UI", 9))
        self.resolution_value.pack(anchor="w", pady=(2, 0))

        # ── Scan Settings ──
        self._section_label(ctrl_content, "⚙ Scan Settings")
        settings_frame = Frame(ctrl_content, bg=self.theme["bg_tertiary"],
                               relief="flat", bd=0, padx=12, pady=10)
        settings_frame.pack(fill="x", pady=(0, 12))

        Label(settings_frame, text="Number of Angles", bg=self.theme["bg_tertiary"],
              fg=self.theme["text_secondary"], font=("Segoe UI", 9)).pack(anchor="w", pady=(0, 4))
        self.angles_var = StringVar(value=str(self.config.get("num_angles", 70)))
        Entry(settings_frame, textvariable=self.angles_var,
              bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
              relief="flat", font=("Segoe UI", 10), justify="center").pack(fill="x", pady=(0, 8))

        Label(settings_frame, text="Recording Duration (seconds)", bg=self.theme["bg_tertiary"],
              fg=self.theme["text_secondary"], font=("Segoe UI", 9)).pack(anchor="w", pady=(0, 4))
        self.dur_var = StringVar(value=str(self.config.get("recording_duration", 30)))
        Entry(settings_frame, textvariable=self.dur_var,
              bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
              relief="flat", font=("Segoe UI", 10), justify="center").pack(fill="x")

        # ── ArUco Dimensions ──
        self._section_label(ctrl_content, "📐 ArUco Dimensions")
        dim_frame = Frame(ctrl_content, bg=self.theme["bg_tertiary"],
                          relief="flat", bd=0, padx=12, pady=10)
        dim_frame.pack(fill="x", pady=(0, 12))

        self.dim_label = Label(dim_frame, text="No marker detected", bg=self.theme["bg_tertiary"],
                               fg=self.theme["accent_warning"], font=("Segoe UI", 10, "bold"))
        self.dim_label.pack(anchor="w", pady=(0, 4))

        self.dim_detail = Label(dim_frame, text="Place ArUco marker near object",
                                bg=self.theme["bg_tertiary"], fg=self.theme["text_secondary"],
                                font=("Segoe UI", 9))
        self.dim_detail.pack(anchor="w")

        # ── Actions ──
        self._section_label(ctrl_content, "▶ Actions")

        self.scan_btn = Button(ctrl_content, text="▶  START 3D SCAN",
                               command=self._start_scan,
                               bg=self.theme["accent_primary"], fg="white",
                               relief="flat", bd=0, font=("Segoe UI", 12, "bold"),
                               padx=20, pady=12, cursor="hand2")
        self.scan_btn.pack(fill="x", pady=(0, 6))

        self.live_tracker_btn = Button(ctrl_content, text="🎯 LIVE TRACKING (ORB + Point Cloud)",
                                       command=self._launch_live_tracker,
                                       bg=self.theme["accent_success"], fg="white",
                                       relief="flat", bd=0, font=("Segoe UI", 10, "bold"),
                                       padx=16, pady=10, cursor="hand2")
        self.live_tracker_btn.pack(fill="x", pady=(0, 6))

        self.stop_btn = Button(ctrl_content, text="⏹  STOP SCAN",
                               command=self._stop_scan,
                               bg=self.theme["accent_danger"], fg="white",
                               relief="flat", bd=0, font=("Segoe UI", 11, "bold"),
                               padx=16, pady=10, cursor="hand2", state="disabled")
        self.stop_btn.pack(fill="x", pady=(0, 8))

        # ── Progress ──
        self._section_label(ctrl_content, "📊 Progress")
        progress_frame = Frame(ctrl_content, bg=self.theme["bg_tertiary"],
                               relief="flat", bd=0, padx=12, pady=12)
        progress_frame.pack(fill="x", pady=(0, 12))

        self.progress_var = IntVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                            length=100, mode="determinate")
        self.progress_bar.pack(fill="x", pady=(0, 6))

        self.progress_label = Label(progress_frame, text="0%", bg=self.theme["bg_tertiary"],
                                    fg=self.theme["text_secondary"], font=("Segoe UI", 9))
        self.progress_label.pack()

        # ── Quick Actions ──
        self._section_label(ctrl_content, "⚡ Quick Actions")
        actions_frame = Frame(ctrl_content, bg=self.theme["bg_tertiary"],
                              relief="flat", bd=0, padx=12, pady=10)
        actions_frame.pack(fill="x", pady=(0, 12))

        Button(actions_frame, text="📁 Open Captures Folder",
               command=self._open_captures,
               bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
               relief="flat", bd=0, font=("Segoe UI", 9),
               padx=10, pady=6, cursor="hand2").pack(fill="x", pady=(0, 6))

        Button(actions_frame, text="🔄 Reconstruct Latest",
               command=self._direct_reconstruct,
               bg=self.theme["accent_secondary"], fg="white",
               relief="flat", bd=0, font=("Segoe UI", 9, "bold"),
               padx=10, pady=6, cursor="hand2").pack(fill="x", pady=(0, 6))

        Button(actions_frame, text="👁 View 3D Model",
               command=self._view_model,
               bg=self.theme["accent_primary"], fg="white",
               relief="flat", bd=0, font=("Segoe UI", 9, "bold"),
               padx=10, pady=6, cursor="hand2").pack(fill="x")

        # ── Bottom Status Bar ─────────────────────────────────────────
        status_bar = Frame(self.root, bg=self.theme["bg_secondary"], height=32)
        status_bar.pack(fill="x", side="bottom")
        status_bar.pack_propagate(False)

        self.status_icon = Label(status_bar, text="●", bg=self.theme["bg_secondary"],
                                 fg=self.theme["accent_success"], font=("Segoe UI", 10))
        self.status_icon.pack(side="left", padx=(20, 6), pady=6)

        self.status_text = StringVar(value="Ready to scan")
        Label(status_bar, textvariable=self.status_text, bg=self.theme["bg_secondary"],
              fg=self.theme["text_secondary"], font=("Segoe UI", 9),
              anchor="w").pack(side="left", fill="x", expand=True, pady=6)

        Label(status_bar, text="Ctrl+S: Scan | Ctrl+R: Reconstruct | Ctrl+L: LiveTrack | Esc: Exit",
              bg=self.theme["bg_secondary"], fg=self.theme["text_tertiary"],
              font=("Segoe UI", 8)).pack(side="right", padx=20, pady=6)

    def _section_label(self, parent, text):
        Label(parent, text=text, bg=self.theme["bg_primary"],
              fg=self.theme["text_secondary"], font=("Segoe UI", 9, "bold")).pack(
                  anchor="w", pady=(0, 8))

    # ─── Auto Connect ────────────────────────────────────────────────────
    def _auto_connect(self):
        if self._auto_connected:
            return
        self._auto_connected = True
        self._set_status("Detecting phone camera...", "busy")
        self._show_placeholder("Connecting to IP Webcam...\nMake sure the app is running on your phone")

        def connect():
            url = self.config.get("webcam_url", IP_WEBCAM_URL) + IP_WEBCAM_STREAM_PATH
            stream = CameraStream(source=url)
            if stream.open():
                self.root.after(0, lambda: self._on_connect(stream))
            else:
                self.root.after(0, self._on_connect_failed)

        threading.Thread(target=connect, daemon=True).start()

    def _on_connect(self, stream):
        self.camera_stream = stream
        self.camera_stream.start()
        self.camera_open = True
        self.cam_status.config(text=f"● {stream.source_name}", fg=self.theme["accent_success"])
        self.conn_status.config(text="Connected", fg=self.theme["accent_success"])
        self.res_label.config(text=stream.resolution)
        self.resolution_value.config(text=stream.resolution)
        self._set_status("Connected to phone camera", "success")
        self._update_preview()

    def _on_connect_failed(self):
        self.cam_status.config(text="● Offline", fg=self.theme["accent_danger"])
        self.conn_status.config(text="Disconnected", fg=self.theme["accent_danger"])
        self._show_placeholder(
            "📱 Camera Not Found\n\n"
            "1. Install IP Webcam from Play Store\n"
            "2. Open the app \u2192 tap Start Server\n"
            "3. Phone and PC on same WiFi\n\n"
            f"Expected: {self.config.get('webcam_url', IP_WEBCAM_URL)}/video\n\n"
            "Click Settings to configure IP"
        )
        self._set_status("Camera not found. Open IP Webcam on your phone.", "error")

    # ─── Preview Pipeline ───────────────────────────────────────────────
    def _update_preview(self):
        if not self.camera_open or not self.camera_stream:
            self.root.after(30, self._update_preview)
            return

        frame = self.camera_stream.read()
        if frame is not None:
            # Record frames during scan
            if self._recording:
                self._record_frames.append(frame.copy())

            # Apply background removal if enabled
            if self.bg_remover.enabled:
                processed = self.bg_remover.process(frame)
                mask = self.bg_remover.get_mask()
            else:
                processed = frame.copy()
                mask = None

            # Update capture tracking (for auto-stop)
            if self._recording:
                self._update_capture_tracking(frame)

            # Apply ArUco dimension tracking
            processed = self.aruco_tracker.process(processed)

            # GREEN CAPTURE OVERLAY - Show scanned areas in green
            if self._recording and self._capture_mask is not None:
                try:
                    mask_uint8 = (self._capture_mask * 255).astype(np.uint8)
                    green_overlay = np.zeros_like(processed)
                    green_overlay[:, :] = (0, 255, 0)
                    green_overlay = cv2.bitwise_and(green_overlay, green_overlay, mask=mask_uint8)
                    cv2.addWeighted(processed, 0.7, green_overlay, 0.3, 0, processed)
                except Exception:
                    pass

            # Update dimension display
            dims = self.aruco_tracker.get_dimensions()
            if dims[0] > 0 and dims[1] > 0:
                self.dim_label.config(
                    text=f"{dims[0]} x {dims[1]} x {dims[2]} cm",
                    fg=self.theme["accent_success"]
                )
                self.dim_detail.config(
                    text=f"W: {dims[0]} cm | H: {dims[1]} cm | D: {dims[2]} cm"
                )
            else:
                self.dim_label.config(
                    text="No marker detected",
                    fg=self.theme["accent_warning"]
                )
                self.dim_detail.config(text="Place ArUco marker near object")

            # Display
            rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            if cw > 10 and ch > 10:
                iw, ih = img.size
                scale = min(cw / iw, ch / ih)
                img = img.resize((int(iw * scale), int(ih * scale)), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(cw // 2, ch // 2, image=photo, anchor="center")
            self.canvas.image = photo

        self.root.after(30, self._update_preview)

    # ─── Live Tracker (Launches external process) ───────────────────────
    def _launch_live_tracker(self):
        """Launch the LiveTracker3D in a separate thread/process."""
        self._set_status("🎯 Starting LiveTracker3D...", "busy")

        def run():
            try:
                # Use the same camera source as the GUI
                from scanner3d.scanner import LiveTracker3D
                tracker = LiveTracker3D()
                tracker.run()
                self.root.after(0, lambda: self._set_status("Live tracking ended.", "success"))
            except ImportError as e:
                error_msg = "LiveTracker3D requires open3d. Install: pip install open3d"
                self.root.after(0, lambda: messagebox.showerror("Live Tracker", error_msg))
                self.root.after(0, lambda: self._set_status("✗ Live tracking unavailable", "error"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Live Tracker Error", str(e)))
                self.root.after(0, lambda: self._set_status("✗ Live tracking error", "error"))

        threading.Thread(target=run, daemon=True).start()

    # ─── Scan ────────────────────────────────────────────────────────────
    def _start_scan(self):
        if not self.camera_open:
            messagebox.showwarning("Camera Required",
                                   "Camera is not connected.\nMake sure IP Webcam is running on your phone.")
            return
        if self.scanning:
            return

        self.scanning = True
        self._recording = True
        self._record_start = time.time()
        self._record_frames = []
        self._capture_mask = None
        self._capture_percentage = 0.0
        self._prev_features = None
        self.scan_btn.config(text="⏳ Scanning...", state="disabled")
        self.stop_btn.config(state="normal")
        self.live_tracker_btn.config(state="disabled")
        self._set_status("SCANNING \u2014 rotate the object 360\u00b0", "busy")

        # Create session
        sid = time.strftime("%Y%m%d_%H%M%S")
        self._session_dir = Path(OUTPUT_DIR) / sid
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self.config["last_session"] = sid
        save_config(self.config)

        try:
            self._scan_duration = int(self.dur_var.get())
        except ValueError:
            self._scan_duration = 30

        self._update_recording_status()

    def _update_capture_tracking(self, frame):
        """Update capture mask to track which areas have been scanned (visual overlay only)."""
        try:
            if frame is None:
                return
            h, w = frame.shape[:2]
            if self._capture_mask is None or self._capture_mask.shape[:2] != (h, w):
                self._capture_mask = np.zeros((h, w), dtype=np.float32)

            if not hasattr(self, '_frame_counter'):
                self._frame_counter = 0
            self._frame_counter += 1
            if self._frame_counter % 2 != 0:
                return

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            features = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=20)

            if features is not None:
                features = np.int0(features)
                for feature in features:
                    x, y = feature.ravel()
                    cv2.circle(self._capture_mask, (x, y), 25, 1.0, -1)
                self._capture_mask *= 0.998
                captured_pixels = np.sum(self._capture_mask > 0.1)
                total_pixels = h * w
                self._capture_percentage = min(100.0, (captured_pixels / total_pixels) * 100.0 * 4.0)
                # NOTE: Recording duration is now controlled ONLY by the timer in _update_recording_status
        except Exception as e:
            print(f"[CaptureTracking] Error: {e}")

    def _update_recording_status(self):
        if not self._recording:
            return
        elapsed = time.time() - self._record_start
        remaining = max(0, self._scan_duration - elapsed)

        # Use elapsed time for progress, not smart tracking
        progress_pct = int((elapsed / self._scan_duration) * 100) if self._scan_duration > 0 else 0
        self.progress_var.set(progress_pct)
        self.progress_label.config(text=f"{progress_pct}% ({remaining:.0f}s left)")
        self._set_status(f"🎥 Recording {elapsed:.0f}s / {self._scan_duration}s \u2014 {len(self._record_frames)} frames", "busy")

        if elapsed >= self._scan_duration:
            self._finish_recording()
        else:
            self.root.after(500, self._update_recording_status)

    def _finish_recording(self):
        if not self._recording:
            return
        self._recording = False
        self._set_status("Saving video...", "busy")
        self.root.update()

        def worker():
            try:
                sdir = str(self._session_dir)
                vpath = os.path.join(sdir, "scan_video.mp4")

                if self._record_frames:
                    h, w = self._record_frames[0].shape[:2]
                    # Try H264 codec first for best quality, fall back to MJPG
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')
                    fps = max(15, len(self._record_frames) // max(1, self._scan_duration))
                    out = cv2.VideoWriter(vpath, fourcc, fps, (w, h))
                    if not out.isOpened():
                        fourcc = cv2.VideoWriter_fourcc(*'H264')
                        out = cv2.VideoWriter(vpath, fourcc, fps, (w, h))
                    if not out.isOpened():
                        fourcc = cv2.VideoWriter_fourcc(*'X264')
                        out = cv2.VideoWriter(vpath, fourcc, fps, (w, h))
                    if not out.isOpened():
                        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                        vpath = os.path.join(sdir, "scan_video.avi")
                        out = cv2.VideoWriter(vpath, fourcc, fps, (w, h))

                    for i, frame in enumerate(self._record_frames):
                        out.write(frame)
                        if i % 30 == 0:
                            pct = int((i / len(self._record_frames)) * 30) + 30
                            self.root.after(0, lambda p=pct: self.progress_var.set(p))
                    out.release()
                    self.root.after(0, lambda: self.progress_var.set(40))

                    # Extract frames using Camera method
                    self.root.after(0, lambda: self._set_status("Extracting frames...", "busy"))
                    try:
                        num_angles = int(self.angles_var.get())
                    except ValueError:
                        num_angles = 70

                    from scanner3d.camera import Camera
                    temp_cam = Camera()
                    extracted = temp_cam.extract_frames_from_video(
                        vpath, num_frames=num_angles, output_dir=sdir
                    )

                    self.root.after(0, lambda: self.progress_var.set(80))

                    if extracted:
                        self.root.after(0, lambda: self._on_scan_done(
                            len(extracted), self._session_dir.name, sdir))
                    else:
                        raise RuntimeError("No frames extracted")
                else:
                    raise RuntimeError("No frames recorded")
            except Exception as e:
                self.root.after(0, lambda err=e: self._on_scan_error(err))

        threading.Thread(target=worker, daemon=True).start()

    def _on_scan_done(self, count, sid, sdir):
        self.progress_var.set(100)
        self.progress_label.config(text="100%")
        self._recording = False
        self._set_status(f"\u2713 Complete! {count} images captured", "success")
        self.scan_btn.config(text="\u25b6  START 3D SCAN", state="normal")
        self.stop_btn.config(state="disabled")
        self.live_tracker_btn.config(state="normal")
        self.scanning = False
        self._record_frames = []

        if messagebox.askyesno("Scan Complete",
                               f"\u2713 {count} images captured!\n\n"
                               f"Session: {sid}\n\n"
                               "Reconstruct to 3D model now?"):
            self._direct_reconstruct()
        elif messagebox.askyesno("Scan Complete",
                                 f"\u2713 {count} images captured!\n\n"
                                 f"Session: {sid}\n\nOpen folder?"):
            os.startfile(sdir)

    def _on_scan_error(self, error):
        self._recording = False
        self.scan_btn.config(text="\u25b6  START 3D SCAN", state="normal")
        self.stop_btn.config(state="disabled")
        self.live_tracker_btn.config(state="normal")
        self.scanning = False
        self._record_frames = []
        self.progress_var.set(0)
        self.progress_label.config(text="0%")
        self._set_status(f"\u2717 Scan failed: {error}", "error")
        messagebox.showerror("Scan Error", f"Scan failed:\n{error}")

    def _stop_scan(self):
        if self.scanning:
            self._set_status("Stopping scan...", "busy")
            self._recording = False
            self.scanning = False
            self._finish_recording()

    # ─── Reconstruct (uses the full KiriReconstructor pipeline) ────────
    def _direct_reconstruct(self, session_id: str = None):
        """Send latest video to Kiri Engine and get 3D model."""
        if session_id is None:
            session_id = self.config.get("last_session", "")
        if not session_id:
            # Try to find latest session
            latest = find_latest_session()
            if latest:
                session_id = Path(latest).name
            else:
                messagebox.showwarning("No Scan", "Complete a scan first.")
                return

        session_dir = Path(OUTPUT_DIR) / session_id
        if not session_dir.exists():
            messagebox.showerror("Not Found", f"Session {session_id} not found.\nRun a scan first.")
            return

        video_files = list(session_dir.glob("*.avi")) + list(session_dir.glob("*.mp4"))
        if not video_files:
            messagebox.showerror("No Video", f"No video found in session {session_id}.\nRun a scan first.")
            return

        output_format = self.config.get("output_format", "OBJ")
        recon_quality = self.config.get("recon_quality", "ultra")

        self._set_status(f"Reconstructing {session_id} \u2192 {output_format} (ultra quality)...", "busy")
        self.progress_var.set(5)
        self.root.update()

        def worker():
            try:
                def progress_cb(p):
                    self.root.after(0, lambda: self.progress_var.set(p))
                    self.root.after(0, lambda: self.progress_label.config(text=f"{p}%"))

                result = run_reconstruction(
                    session_id=session_id,
                    output_name=None,
                    quality=recon_quality,
                    file_format=output_format,
                    auto_view=True,
                    progress_callback=progress_cb,
                )
                self.root.after(0, lambda: self.progress_var.set(100))
                self.root.after(0, lambda: self.progress_label.config(text="100%"))

                if result:
                    self.root.after(0, lambda: self._set_status(f"\u2713 {output_format} created!", "success"))
                    self.root.after(0, lambda: messagebox.showinfo(
                        "Reconstruction Complete!",
                        f"3D model reconstructed successfully!\n\n"
                        f"Session: {session_id}\n"
                        f"Format: {output_format}\n\n"
                        f"Opening 3D viewer..."
                    ))
                else:
                    raise RuntimeError("Reconstruction did not complete successfully.")
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: self._set_status("\u2717 Reconstruction failed", "error"))
                self.root.after(0, lambda: messagebox.showerror(
                    "Reconstruction Failed", error_msg))

        threading.Thread(target=worker, daemon=True).start()

    # ─── View Model (using trimesh) ─────────────────────────────────────
    def _view_model(self):
        """Open the latest downloaded model in 3D viewer."""
        model_dirs = sorted(MODEL_OUTPUT_DIR.glob("model_*"), key=os.path.getmtime, reverse=True)

        if not model_dirs:
            messagebox.showwarning("No Model", "No model found.\nComplete a scan and reconstruction first.")
            return

        latest_dir = model_dirs[0]
        obj_files = list(latest_dir.glob("*.obj")) + list(latest_dir.glob("*.stl")) \
                     + list(latest_dir.glob("*.glb")) + list(latest_dir.glob("*.ply"))

        if not obj_files:
            messagebox.showwarning("No Model File", f"No supported 3D file found in:\n{latest_dir}")
            return

        model_path = str(obj_files[0])
        self._set_status(f"Opening model: {obj_files[0].name}", "busy")

        def viewer():
            try:
                import trimesh
                mesh = trimesh.load(model_path)
                print(f"Viewing: {model_path}")
                print(f"Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
                mesh.show()
                self.root.after(0, lambda: self._set_status("Ready to scan", "success"))
            except Exception as e:
                self.root.after(0, lambda: self._set_status("\u2717 Viewer error", "error"))
                self.root.after(0, lambda: messagebox.showerror("Viewer Error", str(e)))

        threading.Thread(target=viewer, daemon=True).start()

    # ─── Quick Actions ──────────────────────────────────────────────────
    def _open_captures(self):
        p = Path(OUTPUT_DIR)
        if p.exists():
            os.startfile(str(p))
        else:
            messagebox.showwarning("Not Found", "No captures yet. Complete a scan first.")

    # ─── Settings ────────────────────────────────────────────────────────
    def _open_settings(self):
        dlg = Toplevel(self.root)
        dlg.title("Settings")
        dlg.geometry("580x620")
        dlg.configure(bg=self.theme["bg_primary"])
        dlg.transient(self.root)
        dlg.grab_set()

        # ── Scrollable canvas ──────────────────────────────────────
        canvas = Canvas(dlg, bg=self.theme["bg_primary"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(dlg, orient="vertical", command=canvas.yview)
        scrollable_frame = Frame(canvas, bg=self.theme["bg_primary"], padx=24, pady=24)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=540)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Clean up binding on close
        def _on_dlg_close():
            canvas.unbind_all("<MouseWheel>")
            dlg.destroy()
        dlg.protocol("WM_DELETE_WINDOW", _on_dlg_close)

        mf = scrollable_frame

        Label(mf, text="\u2699 Settings", bg=self.theme["bg_primary"],
              fg=self.theme["text_primary"], font=("Segoe UI", 18, "bold")).pack(pady=(0, 20))

        # ── Camera ──
        self._settings_section(mf, "\U0001f4f1 Camera")

        cam_frame = Frame(mf, bg=self.theme["bg_tertiary"], padx=12, pady=10)
        cam_frame.pack(fill="x", pady=(0, 16))

        Label(cam_frame, text="IP Webcam URL:", bg=self.theme["bg_tertiary"],
              fg=self.theme["text_secondary"], font=("Segoe UI", 10)).pack(anchor="w")
        url_var = StringVar(value=self.config.get("webcam_url", IP_WEBCAM_URL))
        Entry(cam_frame, textvariable=url_var, bg=self.theme["bg_secondary"],
              fg=self.theme["text_primary"], relief="flat", font=("Segoe UI", 11)).pack(fill="x", pady=(4, 8))

        def test_connection():
            self._set_status("Testing connection...", "busy")
            url = url_var.get().strip()
            try:
                r = urllib.request.urlopen(url + "/", timeout=5)
                if r.status == 200:
                    messagebox.showinfo("Success", f"\u2713 IP Webcam reachable at {url}")
                    self._set_status("Connection successful", "success")
                r.close()
            except Exception as e:
                messagebox.showerror("Failed", f"\u2717 Cannot reach {url}\n\n{e}")
                self._set_status("Connection test failed", "error")

        Button(cam_frame, text="\U0001f4e1 Test Connection", command=test_connection,
               bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
               relief="flat", bd=0, font=("Segoe UI", 9), cursor="hand2",
               padx=10, pady=4).pack(pady=(0, 4))

        # ── Scanning ──
        self._settings_section(mf, "\u2699 Scanning")

        scan_frame = Frame(mf, bg=self.theme["bg_tertiary"], padx=12, pady=10)
        scan_frame.pack(fill="x", pady=(0, 16))

        row1 = Frame(scan_frame, bg=self.theme["bg_tertiary"])
        row1.pack(fill="x", pady=(0, 8))

        Label(row1, text="Default Angles:", bg=self.theme["bg_tertiary"],
              fg=self.theme["text_secondary"], font=("Segoe UI", 10)).pack(side="left")
        angles_var = StringVar(value=str(self.config.get("num_angles", 70)))
        Entry(row1, textvariable=angles_var, width=8,
              bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
              relief="flat", font=("Segoe UI", 11)).pack(side="right")

        row2 = Frame(scan_frame, bg=self.theme["bg_tertiary"])
        row2.pack(fill="x", pady=(0, 8))

        Label(row2, text="Duration (s):", bg=self.theme["bg_tertiary"],
              fg=self.theme["text_secondary"], font=("Segoe UI", 10)).pack(side="left")
        dur_var_s = StringVar(value=str(self.config.get("recording_duration", 30)))
        Entry(row2, textvariable=dur_var_s, width=8,
              bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
              relief="flat", font=("Segoe UI", 11)).pack(side="right")

        # ── Features ──
        self._settings_section(mf, "\U0001f9e9 Features")

        feat_frame = Frame(mf, bg=self.theme["bg_tertiary"], padx=12, pady=10)
        feat_frame.pack(fill="x", pady=(0, 16))

        bg_var = BooleanVar(value=self.config.get("bg_removal", False))
        ttk.Checkbutton(feat_frame, text="Enable Background Removal (CPU)",
                        variable=bg_var).pack(anchor="w", pady=(0, 4))

        aruco_var = BooleanVar(value=self.config.get("aruco_enabled", True))
        ttk.Checkbutton(feat_frame, text="Enable ArUco Dimension Tracking",
                        variable=aruco_var).pack(anchor="w", pady=(0, 4))

        row_marker = Frame(feat_frame, bg=self.theme["bg_tertiary"])
        row_marker.pack(fill="x", pady=(4, 0))

        Label(row_marker, text="ArUco Marker Size (cm):", bg=self.theme["bg_tertiary"],
              fg=self.theme["text_secondary"], font=("Segoe UI", 10)).pack(side="left")
        marker_var = StringVar(value=str(self.config.get("marker_size_cm", 5.0)))
        Entry(row_marker, textvariable=marker_var, width=8,
              bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
              relief="flat", font=("Segoe UI", 11)).pack(side="right")

        # ── Reconstruction ──
        self._settings_section(mf, "\u2601 Reconstruction")

        recon_frame = Frame(mf, bg=self.theme["bg_tertiary"], padx=12, pady=10)
        recon_frame.pack(fill="x", pady=(0, 16))

        Label(recon_frame, text="Output Format:", bg=self.theme["bg_tertiary"],
              fg=self.theme["text_secondary"], font=("Segoe UI", 10)).pack(anchor="w")
        fmt_var = StringVar(value=self.config.get("output_format", "OBJ"))
        format_menu = OptionMenu(recon_frame, fmt_var, "OBJ", "STL", "GLB", "FBX", "PLY")
        format_menu.config(bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
                           relief="flat", font=("Segoe UI", 11))
        format_menu.pack(fill="x", pady=(4, 8))

        Label(recon_frame, text="Quality:", bg=self.theme["bg_tertiary"],
              fg=self.theme["text_secondary"], font=("Segoe UI", 10)).pack(anchor="w")
        qual_var = StringVar(value=self.config.get("recon_quality", "ultra"))
        qual_menu = OptionMenu(recon_frame, qual_var, "draft", "medium", "high", "ultra")
        qual_menu.config(bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
                         relief="flat", font=("Segoe UI", 11))
        qual_menu.pack(fill="x", pady=(4, 8))

        # API Key
        Label(recon_frame, text="Kiri Engine API Key:", bg=self.theme["bg_tertiary"],
              fg=self.theme["text_secondary"], font=("Segoe UI", 10)).pack(anchor="w")
        api_var = StringVar(value=self.config.get("kiri_api_key", ""))
        Entry(recon_frame, textvariable=api_var, bg=self.theme["bg_secondary"],
              fg=self.theme["text_primary"], relief="flat", font=("Segoe UI", 11),
              show="*").pack(fill="x", pady=(4, 0))

        # Save / Cancel
        btn_frame = Frame(mf, bg=self.theme["bg_primary"])
        btn_frame.pack(fill="x", pady=(16, 0))

        def save():
            self.config["webcam_url"] = url_var.get().strip()
            self.config["kiri_api_key"] = api_var.get().strip()
            self.config["num_angles"] = int(angles_var.get())
            self.config["recording_duration"] = int(dur_var_s.get())
            self.config["bg_removal"] = bg_var.get()
            self.config["aruco_enabled"] = aruco_var.get()
            self.config["marker_size_cm"] = float(marker_var.get())
            self.config["output_format"] = fmt_var.get()
            self.config["recon_quality"] = qual_var.get()

            # Apply immediately
            self.bg_remover.toggle(self.config["bg_removal"])
            self.aruco_tracker.toggle(self.config["aruco_enabled"])
            self.aruco_tracker.marker_size_cm = self.config["marker_size_cm"]

            save_config(self.config)
            self.server_label.config(text=f"Server: {self.config['webcam_url']}")
            self._set_status("Settings saved", "success")
            dlg.destroy()

        Button(btn_frame, text="\U0001f4be Save Settings", command=save,
               bg=self.theme["accent_primary"], fg="white",
               relief="flat", bd=0, font=("Segoe UI", 11, "bold"),
               cursor="hand2", padx=16, pady=10).pack(fill="x", pady=(0, 8))
        Button(btn_frame, text="Cancel", command=_on_dlg_close,
               bg=self.theme["bg_tertiary"], fg=self.theme["text_primary"],
               relief="flat", bd=0, font=("Segoe UI", 10),
               cursor="hand2", padx=10, pady=6).pack()

    def _settings_section(self, parent, text):
        Label(parent, text=text, bg=self.theme["bg_primary"],
              fg=self.theme["text_secondary"], font=("Segoe UI", 9, "bold")).pack(
                  anchor="w", pady=(0, 6))

    # ─── Theme ───────────────────────────────────────────────────────────
    def _toggle_theme(self):
        new = "light" if self.config.get("theme") == "dark" else "dark"
        self.config["theme"] = new
        save_config(self.config)
        self.theme = ProfessionalTheme.LIGHT if new == "light" else ProfessionalTheme.DARK
        if self.camera_open:
            self._close_camera()
        self.root.destroy()
        self.__init__()
        self.run()

    # ─── Camera ──────────────────────────────────────────────────────────
    def _close_camera(self):
        if self.camera_stream:
            self.camera_stream.release()
            self.camera_stream = None
        self.camera_open = False
        self.cam_status.config(text="\u25cf Offline", fg=self.theme["accent_danger"])

    # ─── Helpers ─────────────────────────────────────────────────────────
    def _show_placeholder(self, text):
        self.canvas.delete("all")
        self.canvas.create_text(20, 20, text=text, fill="#64748b",
                                font=("Segoe UI", 12), anchor="nw")

    def _set_status(self, text, level="info"):
        self.status_text.set(text)
        if level == "error":
            self.status_icon.config(fg=self.theme["accent_danger"])
        elif level == "success":
            self.status_icon.config(fg=self.theme["accent_success"])
        elif level == "busy":
            self.status_icon.config(fg=self.theme["accent_warning"])
        else:
            self.status_icon.config(fg=self.theme["accent_primary"])
        self.root.update_idletasks()

    def _on_close(self):
        if self.scanning:
            if not messagebox.askokcancel("Exit", "Scan in progress. Exit?"):
                return
        self.bg_remover.release()
        self._close_camera()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ─── Entry ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = ScannerGUI()
    app.run()