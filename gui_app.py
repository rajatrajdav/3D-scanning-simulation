"""
3D Scanner Pro - Professional GUI Application
=============================================
A modern, professional desktop interface for 3D scanning.
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
import subprocess
from pathlib import Path
from tkinter import (
    Tk, Toplevel, Frame, Label, Button, Canvas, Entry,
    StringVar, IntVar, ttk, messagebox
)
from PIL import Image, ImageTk
from typing import Optional, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scanner3d.droidcam_remote import PhoneRemote
from scanner3d.config import (
    OUTPUT_DIR, MODEL_OUTPUT_DIR, NUM_ANGLES,
    KIRI_API_KEY, KIRI_BASE_URL,
    IP_WEBCAM_URL, IP_WEBCAM_STREAM_PATH
)
from scanner3d.kiri_reconstructor import reconstruct as run_reconstruction
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
    "resolution": "1280x720",
    "bg_removal": False,
    "scanning_grid": True,
    "output_format": "STL"
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
        # Backgrounds
        "bg_primary": "#0f1419",      # Main background
        "bg_secondary": "#1a1f26",    # Cards, panels
        "bg_tertiary": "#242b33",     # Inputs, elevated
        "bg_hover": "#2d353f",        # Hover states
        
        # Accents
        "accent_primary": "#00b4d8",  # Cyan - main accent
        "accent_secondary": "#7c3aed", # Purple - secondary
        "accent_success": "#10b981",  # Green - success
        "accent_warning": "#f59e0b",  # Orange - warning
        "accent_danger": "#ef4444",   # Red - danger
        
        # Text
        "text_primary": "#f1f5f9",    # Main text
        "text_secondary": "#94a3b8",  # Secondary text
        "text_tertiary": "#64748b",   # Tertiary text
        
        # Borders
        "border_light": "#334155",
        "border_medium": "#475569",
        
        # Shadows
        "shadow": "rgba(0, 0, 0, 0.3)",
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
        "shadow": "rgba(0, 0, 0, 0.1)",
    }


# ─── Camera Stream ──────────────────────────────────────────────────────────
class CameraStream:
    """Background thread for IP Webcam MJPEG stream."""
    
    def __init__(self, stream_url: str = None):
        self.stream_url = stream_url or (IP_WEBCAM_URL + IP_WEBCAM_STREAM_PATH)
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.thread: Optional[threading.Thread] = None
        self.source_name = ""
        self._resolution = ""
        self._lock = threading.Lock()
    
    def open(self) -> bool:
        print(f"[Camera] Connecting: {self.stream_url}")
        try:
            self.cap = cv2.VideoCapture(self.stream_url)
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    self._resolution = f"{w}x{h}"
                    self.source_name = f"IP Webcam ({self._resolution})"
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
                h, w = frame.shape[:2]
                if w > 640:
                    scale = 640 / w
                    frame = cv2.resize(frame, (int(w*scale), int(h*scale)), cv2.INTER_AREA)
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


# ─── Scanning Grid Overlay ──────────────────────────────────────────────────
class ScanningGridOverlay:
    """Scanning grid overlay - currently disabled (no grid, no scanline)."""
    
    def __init__(self, enable: bool = False):
        self.enabled = enable
    
    def process(self, frame: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        # No grid, no scanline - return frame as-is
        return frame


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
        
        # Smart scan tracking
        self._capture_mask: Optional[np.ndarray] = None
        self._capture_percentage = 0.0
        self._prev_features: Optional[np.ndarray] = None
        self._scan_complete_threshold = 85.0  # Auto-stop at 85% captured
        
        # Modules
        self.bg_remover = BackgroundRemover(enable=self.config.get("bg_removal", False))
        self.scanning_grid = ScanningGridOverlay(enable=False)
        self.aruco_tracker = ArUcoDimensionTracker(
            marker_size_cm=self.config.get("marker_size_cm", 5.0),
            enable=True
        )
        
        # Build UI
        self.root = Tk()
        self.root.title("3D Scanner Pro")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        self.root.configure(bg=self.theme["bg_primary"])
        
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(500, self._auto_connect)
    
    def _build_ui(self):
        """Build professional UI layout."""
        
        # ── Top Navigation Bar ─────────────────────────────────────────
        nav = Frame(self.root, bg=self.theme["bg_secondary"], height=60)
        nav.pack(fill="x", side="top")
        nav.pack_propagate(False)
        
        # Logo & Title
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
        
        # Camera status
        self.cam_status = Label(nav_right, text="● Offline", bg=self.theme["bg_secondary"],
                               fg=self.theme["accent_danger"], font=("Segoe UI", 10, "bold"))
        self.cam_status.pack(side="left", padx=(0, 16))
        
        # Settings button
        btn_settings = Button(nav_right, text="⚙ Settings", command=self._open_settings,
                             bg=self.theme["bg_tertiary"], fg=self.theme["text_primary"],
                             relief="flat", bd=0, font=("Segoe UI", 10),
                             padx=12, pady=6, cursor="hand2")
        btn_settings.pack(side="left", padx=(0, 8))
        
        # Theme toggle
        self.theme_btn = Button(nav_right, 
                               text="☀" if self.config.get("theme") == "dark" else "🌙",
                               command=self._toggle_theme,
                               bg=self.theme["bg_tertiary"], fg=self.theme["text_primary"],
                               relief="flat", bd=0, font=("Segoe UI", 12),
                               padx=10, pady=6, cursor="hand2")
        self.theme_btn.pack(side="left", padx=(0, 8))
        
        # Reconstruct button
        self.recon_btn = Button(nav_right, text="🔧 Reconstruct → STL",
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
        
        # Preview header
        preview_header = Frame(preview_panel, bg=self.theme["bg_secondary"], height=48)
        preview_header.pack(fill="x", padx=20, pady=(16, 0))
        preview_header.pack_propagate(False)
        
        Label(preview_header, text="📷 Live Preview", bg=self.theme["bg_secondary"],
              fg=self.theme["text_primary"], font=("Segoe UI", 13, "bold")).pack(side="left")
        
        self.res_label = Label(preview_header, text="—", bg=self.theme["bg_tertiary"],
                              fg=self.theme["text_secondary"], font=("Segoe UI", 9),
                              padx=8, pady=3)
        self.res_label.pack(side="right")
        
        # Canvas container
        canvas_container = Frame(preview_panel, bg="#000000",
                                relief="flat", bd=1,
                                highlightbackground=self.theme["border_light"],
                                highlightthickness=1)
        canvas_container.pack(fill="both", expand=True, padx=16, pady=12)
        
        self.canvas = Canvas(canvas_container, bg="#000000", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        
        # Preview footer
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
        
        # Control header
        ctrl_header = Frame(control_panel, bg=self.theme["bg_secondary"], height=48)
        ctrl_header.pack(fill="x", padx=20, pady=(16, 0))
        ctrl_header.pack_propagate(False)
        
        Label(ctrl_header, text="🎛 Control Panel", bg=self.theme["bg_secondary"],
              fg=self.theme["text_primary"], font=("Segoe UI", 13, "bold")).pack(side="left")
        
        # Scrollable content
        ctrl_content = Frame(control_panel, bg=self.theme["bg_secondary"])
        ctrl_content.pack(fill="both", expand=True, padx=20, pady=12)
        
        # ── Device Section ──
        self._section_label(ctrl_content, "📱 Device")
        
        device_frame = Frame(ctrl_content, bg=self.theme["bg_tertiary"],
                            relief="flat", bd=0, padx=12, pady=10)
        device_frame.pack(fill="x", pady=(0, 16))
        
        Label(device_frame, text="Connection Status", bg=self.theme["bg_tertiary"],
              fg=self.theme["text_secondary"], font=("Segoe UI", 8, "bold")).pack(anchor="w")
        self.conn_status = Label(device_frame, text="Disconnected", bg=self.theme["bg_tertiary"],
                                fg=self.theme["accent_danger"], font=("Segoe UI", 9))
        self.conn_status.pack(anchor="w", pady=(2, 8))
        
        Label(device_frame, text="Resolution", bg=self.theme["bg_tertiary"],
              fg=self.theme["text_secondary"], font=("Segoe UI", 8, "bold")).pack(anchor="w")
        self.resolution_value = Label(device_frame, text="—", bg=self.theme["bg_tertiary"],
                                     fg=self.theme["text_primary"], font=("Segoe UI", 9))
        self.resolution_value.pack(anchor="w", pady=(2, 0))
        
        # ── Scan Settings ──
        self._section_label(ctrl_content, "⚙ Scan Settings")
        
        settings_frame = Frame(ctrl_content, bg=self.theme["bg_tertiary"],
                              relief="flat", bd=0, padx=12, pady=10)
        settings_frame.pack(fill="x", pady=(0, 16))
        
        # Angles
        Label(settings_frame, text="Number of Angles", bg=self.theme["bg_tertiary"],
              fg=self.theme["text_secondary"], font=("Segoe UI", 9)).pack(anchor="w", pady=(0, 4))
        self.angles_var = StringVar(value=str(self.config.get("num_angles", 70)))
        angles_entry = Entry(settings_frame, textvariable=self.angles_var,
                            bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
                            relief="flat", font=("Segoe UI", 10), justify="center")
        angles_entry.pack(fill="x", pady=(0, 10))
        
        # Duration
        Label(settings_frame, text="Recording Duration (seconds)", bg=self.theme["bg_tertiary"],
              fg=self.theme["text_secondary"], font=("Segoe UI", 9)).pack(anchor="w", pady=(0, 4))
        self.dur_var = StringVar(value=str(self.config.get("recording_duration", 30)))
        dur_entry = Entry(settings_frame, textvariable=self.dur_var,
                         bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
                         relief="flat", font=("Segoe UI", 10), justify="center")
        dur_entry.pack(fill="x")
        
        # ── Actions ──
        self._section_label(ctrl_content, "▶ Actions")
        
        # Start Scan Button (Primary)
        self.scan_btn = Button(ctrl_content, text="▶  START 3D SCAN",
                              command=self._start_scan,
                              bg=self.theme["accent_primary"], fg="white",
                              relief="flat", bd=0, font=("Segoe UI", 12, "bold"),
                              padx=20, pady=14, cursor="hand2")
        self.scan_btn.pack(fill="x", pady=(0, 8))
        
        # Stop Button
        self.stop_btn = Button(ctrl_content, text="⏹  STOP SCAN",
                              command=self._stop_scan,
                              bg=self.theme["accent_danger"], fg="white",
                              relief="flat", bd=0, font=("Segoe UI", 11, "bold"),
                              padx=16, pady=10, cursor="hand2", state="disabled")
        self.stop_btn.pack(fill="x", pady=(0, 12))
        
        # ── Progress ──
        self._section_label(ctrl_content, "📊 Progress")
        
        progress_frame = Frame(ctrl_content, bg=self.theme["bg_tertiary"],
                              relief="flat", bd=0, padx=12, pady=12)
        progress_frame.pack(fill="x", pady=(0, 16))
        
        self.progress_var = IntVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                           length=100, mode="determinate")
        self.progress_bar.pack(fill="x", pady=(0, 6))
        
        self.progress_label = Label(progress_frame, text="0%", bg=self.theme["bg_tertiary"],
                                   fg=self.theme["text_secondary"], font=("Segoe UI", 9))
        self.progress_label.pack()
        
        # ── Live Dimensions ──
        self._section_label(ctrl_content, "📐 Live Dimensions")
        
        dim_frame = Frame(ctrl_content, bg=self.theme["bg_tertiary"],
                         relief="flat", bd=0, padx=12, pady=10)
        dim_frame.pack(fill="x", pady=(0, 16))
        
        self.dim_label = Label(dim_frame, text="No marker detected", bg=self.theme["bg_tertiary"],
                              fg=self.theme["accent_warning"], font=("Segoe UI", 10, "bold"))
        self.dim_label.pack(anchor="w", pady=(0, 4))
        
        self.dim_detail = Label(dim_frame, text="Place ArUco marker near object", 
                               bg=self.theme["bg_tertiary"], fg=self.theme["text_secondary"],
                               font=("Segoe UI", 9))
        self.dim_detail.pack(anchor="w")
        
        # ── Quick Actions ──
        self._section_label(ctrl_content, "⚡ Quick Actions")
        
        actions_frame = Frame(ctrl_content, bg=self.theme["bg_tertiary"],
                             relief="flat", bd=0, padx=12, pady=10)
        actions_frame.pack(fill="x", pady=(0, 16))
        
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
        
        # Keyboard shortcuts hint
        Label(status_bar, text="Ctrl+S: Start | Ctrl+R: Reconstruct | Esc: Exit",
              bg=self.theme["bg_secondary"], fg=self.theme["text_tertiary"],
              font=("Segoe UI", 8)).pack(side="right", padx=20, pady=6)
    
    def _section_label(self, parent, text):
        """Create a section label."""
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
            stream = CameraStream(stream_url=url)
            if stream.open():
                try:
                    remote = PhoneRemote(
                        host=self.config.get("webcam_url", IP_WEBCAM_URL)
                            .replace("http://", "").split(":")[0],
                        port=int(self.config.get("webcam_url", IP_WEBCAM_URL)
                                 .split(":")[-1])
                    )
                    if remote.ping():
                        name = remote.get_phone_name()
                        self.root.after(0, lambda: self.conn_status.config(text=f"Connected: {name}"))
                except Exception:
                    pass
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
            "2. Open the app → tap Start Server\n"
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
            
            # No background removal - show raw feed
            processed = frame
            mask = None
            
            # Update capture tracking (for auto-stop)
            if self._recording:
                self._update_capture_tracking(frame)
            
            # Apply scanning grid
            processed = self.scanning_grid.process(processed, mask)
            
            # Apply ArUco dimension tracking
            processed = self.aruco_tracker.process(processed)
            
            # GREEN CAPTURE OVERLAY - Show scanned areas in green
            if self._recording and self._capture_mask is not None:
                try:
                    # Create green overlay from capture mask
                    mask_uint8 = (self._capture_mask * 255).astype(np.uint8)
                    green_overlay = np.zeros_like(processed)
                    green_overlay[:, :] = (0, 255, 0)  # Green color (BGR)
                    # Apply mask to overlay
                    green_overlay = cv2.bitwise_and(green_overlay, green_overlay, mask=mask_uint8)
                    # Blend with processed frame
                    cv2.addWeighted(processed, 0.7, green_overlay, 0.3, 0, processed)
                except Exception as e:
                    pass  # Silently skip overlay if it fails
            
            # Update dimension display
            dims = self.aruco_tracker.get_dimensions()
            if dims[0] > 0 and dims[1] > 0:
                self.dim_label.config(
                    text=f"{dims[0]} x {dims[1]} x {dims[2]} cm",
                    fg=self.theme["accent_success"]
                )
                self.dim_detail.config(
                    text=f"Width: {dims[0]} cm | Height: {dims[1]} cm | Depth: {dims[2]} cm"
                )
            else:
                self.dim_label.config(
                    text="No marker detected",
                    fg=self.theme["accent_warning"]
                )
                self.dim_detail.config(
                    text="Place ArUco marker near object"
                )
            
            # Display
            rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            if cw > 10 and ch > 10:
                iw, ih = img.size
                scale = min(cw / iw, ch / ih)
                img = img.resize((int(iw*scale), int(ih*scale)), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(cw//2, ch//2, image=photo, anchor="center")
            self.canvas.image = photo
        
        self.root.after(30, self._update_preview)
    
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
        self._set_status("SCANNING — rotate the object 360°", "busy")
        
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
        """Update capture mask to track which areas have been scanned."""
        try:
            if frame is None:
                return
            
            h, w = frame.shape[:2]
            
            # Initialize capture mask if needed
            if self._capture_mask is None or self._capture_mask.shape[:2] != (h, w):
                self._capture_mask = np.zeros((h, w), dtype=np.float32)
            
            # Detect features (fast, every 2nd frame)
            if not hasattr(self, '_frame_counter'):
                self._frame_counter = 0
            self._frame_counter += 1
            if self._frame_counter % 2 != 0:
                return
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            features = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=20)
            
            if features is not None:
                features = np.int0(features)
                
                # Mark features as captured in mask
                for feature in features:
                    x, y = feature.ravel()
                    # Draw green circle on mask
                    cv2.circle(self._capture_mask, (x, y), 25, 1.0, -1)
                
                # Decay mask slightly over time (old captures fade)
                self._capture_mask *= 0.998
                
                # Calculate capture percentage
                captured_pixels = np.sum(self._capture_mask > 0.1)
                total_pixels = h * w
                self._capture_percentage = min(100.0, (captured_pixels / total_pixels) * 100.0 * 4.0)
                
                # Check if scan is complete
                if self._capture_percentage >= self._scan_complete_threshold:
                    print(f"[SmartScan] Scan complete! {self._capture_percentage:.1f}% captured")
                    self.root.after(0, self._finish_recording)
        except Exception as e:
            print(f"[CaptureTracking] Error: {e}")
            pass
    
    def _update_recording_status(self):
        if not self._recording:
            return
        elapsed = time.time() - self._record_start
        remaining = max(0, self._scan_duration - elapsed)
        
        # Use capture percentage instead of timer
        capture_progress = int(self._capture_percentage)
        self.progress_var.set(capture_progress)
        self.progress_label.config(text=f"{capture_progress}% captured")
        self._set_status(f"🎥 Scanning — {self._capture_percentage:.1f}% captured — {len(self._record_frames)} frames", "busy")
        
        # Auto-stop when scan is complete
        if self._capture_percentage >= self._scan_complete_threshold:
            self._finish_recording()
        elif elapsed >= self._scan_duration:
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
                vpath = os.path.join(sdir, "scan_video.avi")
                
                if self._record_frames:
                    h, w = self._record_frames[0].shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    fps = max(15, len(self._record_frames) // self._scan_duration)
                    out = cv2.VideoWriter(vpath, fourcc, fps, (w, h))
                    
                    for i, frame in enumerate(self._record_frames):
                        out.write(frame)
                        if i % 30 == 0:
                            pct = int((i / len(self._record_frames)) * 30) + 30
                            self.root.after(0, lambda p=pct: self.progress_var.set(p))
                    out.release()
                    self.root.after(0, lambda: self.progress_var.set(40))
                    
                    # Extract frames
                    self.root.after(0, lambda: self._set_status("Extracting frames...", "busy"))
                    try:
                        num_angles = int(self.angles_var.get())
                    except ValueError:
                        num_angles = 70
                    
                    from scanner3d.phone_camera import PhoneCamera
                    pcam = PhoneCamera()
                    extracted = pcam.extract_frames_from_video(
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
        self._set_status(f"✓ Complete! {count} images captured", "success")
        self.scan_btn.config(text="▶  START 3D SCAN", state="normal")
        self.stop_btn.config(state="disabled")
        self.scanning = False
        self._record_frames = []
        
        if messagebox.askyesno("Scan Complete",
                               f"✓ {count} images captured!\n\n"
                               f"Session: {sid}\n\n"
                               "Reconstruct to STL now?"):
            self._direct_reconstruct()
        elif messagebox.askyesno("Scan Complete",
                                 f"✓ {count} images captured!\n\n"
                                 f"Session: {sid}\n\nOpen folder?"):
            os.startfile(sdir)
    
    def _on_scan_error(self, error):
        self._recording = False
        self.scan_btn.config(text="▶  START 3D SCAN", state="normal")
        self.stop_btn.config(state="disabled")
        self.scanning = False
        self._record_frames = []
        self.progress_var.set(0)
        self.progress_label.config(text="0%")
        self._set_status(f"✗ Scan failed: {error}", "error")
        messagebox.showerror("Scan Error", f"Scan failed:\n{error}")
    
    def _stop_scan(self):
        if self.scanning:
            self._set_status("Stopping scan...", "busy")
            self._recording = False
            self.scanning = False
            self._finish_recording()
    
    # ─── Reconstruct ────────────────────────────────────────────────────
    def _direct_reconstruct(self):
        """Send latest video to Kiri Engine and get STL."""
        last = self.config.get("last_session", "")
        if not last:
            messagebox.showwarning("No Scan", "Complete a scan first.")
            return
        
        session_dir = Path(OUTPUT_DIR) / last
        if not session_dir.exists():
            messagebox.showerror("Not Found", f"Session {last} not found.\nRun a scan first.")
            return
        
        video_files = list(session_dir.glob("*.avi")) + list(session_dir.glob("*.mp4"))
        if not video_files:
            messagebox.showerror("No Video", f"No video found in session {last}.\nRun a scan first.")
            return
        
        self._set_status(f"🚀 Reconstructing {last} to STL...", "busy")
        self.progress_var.set(5)
        self.root.update()
        
        def worker():
            try:
                def progress_cb(p):
                    self.root.after(0, lambda: self.progress_var.set(p))
                    self.root.after(0, lambda: self.progress_label.config(text=f"{p}%"))
                
                result = run_reconstruction(
                    session_id=last,
                    output_name=None,
                    quality="high",
                    progress_callback=progress_cb
                )
                self.root.after(0, lambda: self.progress_var.set(100))
                self.root.after(0, lambda: self.progress_label.config(text="100%"))
                
                if result:
                    self.root.after(0, lambda: self._set_status("✓ STL created!", "success"))
                    self.root.after(0, lambda: messagebox.showinfo(
                        "🎉 STL Created!",
                        f"3D model reconstructed successfully!\n\n"
                        f"Session: {last}\n\n"
                        f"Downloading 3D model automatically..."
                    ))
                    # Auto-download the model
                    self.root.after(0, lambda sid=last: self._download_model(sid))
                else:
                    raise RuntimeError("No result returned from Kiri Engine")
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: self._set_status("✗ Reconstruction failed", "error"))
                self.root.after(0, lambda: messagebox.showerror(
                    "Reconstruction Failed", error_msg))
        
        threading.Thread(target=worker, daemon=True).start()
    
    # ─── Download & View Model ─────────────────────────────────────────
    def _download_model(self, serialize_id):
        """Download the reconstructed model using standalone download.py script."""
        self._set_status("📥 Downloading model...", "busy")
        
        def worker():
            try:
                # Use standalone download.py script
                script_path = os.path.join(os.path.dirname(__file__), "download.py")
                result = subprocess.run(
                    [sys.executable, script_path, serialize_id],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                if result.returncode == 0:
                    self.root.after(0, lambda: self._set_status("✓ Model downloaded!", "success"))
                    self.root.after(0, lambda: messagebox.showinfo(
                        "🎉 Model Ready!",
                        f"3D model downloaded successfully!\n\n"
                        f"Session: {serialize_id}\n\n"
                        f"Opening 3D viewer automatically..."
                    ))
                    # Auto-open the model viewer
                    self.root.after(1000, lambda sid=serialize_id: self._view_model_by_id(sid))
                else:
                    error_msg = result.stderr or "Download failed"
                    raise RuntimeError(error_msg)
                    
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: self._set_status("✗ Download failed", "error"))
                self.root.after(0, lambda: messagebox.showerror(
                    "Download Failed", error_msg))
        
        threading.Thread(target=worker, daemon=True).start()

    def _view_model_by_id(self, serialize_id):
        """Open a specific downloaded model using standalone view.py script."""
        model_dir = MODEL_OUTPUT_DIR / f"model_{serialize_id}"
        
        if not model_dir.exists():
            messagebox.showwarning("No Model", f"Model not found:\n{model_dir}")
            return
        
        obj_files = list(model_dir.glob("*.obj")) + list(model_dir.glob("*.stl"))
        
        if not obj_files:
            messagebox.showwarning("No Model File", f"No OBJ/STL file found in:\n{model_dir}")
            return
        
        self._set_status(f"👁 Opening model viewer...", "busy")
        
        def viewer():
            try:
                # Use standalone view.py script
                script_path = os.path.join(os.path.dirname(__file__), "view.py")
                result = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    self.root.after(0, lambda: self._set_status("Ready to scan", "success"))
                else:
                    error_msg = result.stderr or "Viewer failed"
                    raise RuntimeError(error_msg)
                    
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: self._set_status("✗ Viewer error", "error"))
                self.root.after(0, lambda: messagebox.showerror(
                    "Viewer Error", error_msg))
        
        threading.Thread(target=viewer, daemon=True).start()

    def _view_model(self):
        """Open the latest downloaded model in 3D viewer."""
        # Find latest model folder
        model_dirs = sorted(MODEL_OUTPUT_DIR.glob("model_*"), key=os.path.getmtime, reverse=True)
        
        if not model_dirs:
            messagebox.showwarning("No Model", "No model found.\nComplete a scan and reconstruction first.")
            return
        
        latest_dir = model_dirs[0]
        # Find OBJ or STL file
        obj_files = list(latest_dir.glob("*.obj")) + list(latest_dir.glob("*.stl"))
        
        if not obj_files:
            messagebox.showwarning("No Model File", f"No OBJ/STL file found in:\n{latest_dir}")
            return
        
        model_path = str(obj_files[0])
        self._set_status(f"👁 Opening model: {obj_files[0].name}", "busy")
        
        def viewer():
            try:
                import trimesh
                mesh = trimesh.load(model_path)
                print(f"Viewing: {model_path}")
                print(f"Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
                mesh.show()
                self.root.after(0, lambda: self._set_status("Ready to scan", "success"))
            except Exception as e:
                self.root.after(0, lambda: self._set_status("✗ Viewer error", "error"))
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
        dlg.geometry("520x420")
        dlg.configure(bg=self.theme["bg_primary"])
        dlg.transient(self.root)
        dlg.grab_set()
        
        mf = Frame(dlg, bg=self.theme["bg_primary"], padx=24, pady=24)
        mf.pack(fill="both", expand=True)
        
        Label(mf, text="⚙ Settings", bg=self.theme["bg_primary"],
              fg=self.theme["text_primary"], font=("Segoe UI", 18, "bold")).pack(pady=(0, 20))
        
        # IP Webcam URL
        Label(mf, text="IP Webcam URL:", bg=self.theme["bg_primary"],
              fg=self.theme["text_secondary"], font=("Segoe UI", 10)).pack(anchor="w")
        url_var = StringVar(value=self.config.get("webcam_url", IP_WEBCAM_URL))
        Entry(mf, textvariable=url_var, bg=self.theme["bg_secondary"],
              fg=self.theme["text_primary"], relief="flat", font=("Segoe UI", 11)).pack(fill="x", pady=(4, 12))
        
        # API Key
        Label(mf, text="Kiri Engine API Key:", bg=self.theme["bg_primary"],
              fg=self.theme["text_secondary"], font=("Segoe UI", 10)).pack(anchor="w")
        api_var = StringVar(value=self.config.get("kiri_api_key", ""))
        Entry(mf, textvariable=api_var, bg=self.theme["bg_secondary"],
              fg=self.theme["text_primary"], relief="flat", font=("Segoe UI", 11)).pack(fill="x", pady=(4, 16))
        
        # Test connection
        def test():
            url = url_var.get().strip()
            self._set_status(f"Testing: {url}...", "busy")
            try:
                r = urllib.request.urlopen(url + "/", timeout=5)
                if r.status == 200:
                    messagebox.showinfo("Success", f"✓ IP Webcam reachable at {url}")
                    self._set_status("Connection successful", "success")
                r.close()
            except Exception as e:
                messagebox.showerror("Failed", f"✗ Cannot reach {url}\n\n{e}")
        
        Button(mf, text="📡 Test Connection", command=test,
               bg=self.theme["bg_tertiary"], fg=self.theme["text_primary"],
               relief="flat", bd=0, font=("Segoe UI", 10), cursor="hand2",
               padx=12, pady=6).pack(pady=(0, 16))
        
        # Save
        def save():
            self.config["webcam_url"] = url_var.get().strip()
            self.config["kiri_api_key"] = api_var.get().strip()
            save_config(self.config)
            self.server_label.config(text=f"Server: {self.config['webcam_url']}")
            self._set_status("Settings saved", "success")
            dlg.destroy()
        
        Button(mf, text="💾 Save Settings", command=save,
               bg=self.theme["accent_primary"], fg="white",
               relief="flat", bd=0, font=("Segoe UI", 11, "bold"),
               cursor="hand2", padx=16, pady=10).pack(fill="x", pady=(8, 8))
        Button(mf, text="Cancel", command=dlg.destroy,
               bg=self.theme["bg_tertiary"], fg=self.theme["text_primary"],
               relief="flat", bd=0, font=("Segoe UI", 10),
               cursor="hand2", padx=10, pady=6).pack()
    
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
        self.cam_status.config(text="● Offline", fg=self.theme["accent_danger"])
    
    # ─── Helpers ──────────────────────────────────────────────────────────
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