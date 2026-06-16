"""
3D Scanner - Professional GUI Application
===========================================
A professional desktop interface for 3D scanning with DroidCam,
simplified scan workflow, and settings management.

Usage:
    python gui_app.py

Prerequisites:
    - Install DroidCam PC Client from https://droidcam.app
    - Connect DroidCam phone app to PC Client over WiFi

Features:
    - Live camera preview (DroidCam virtual camera)
    - Camera index auto-detection with manual override
    - One-click scan (preview → record → extract frames)
    - Kiri Engine reconstruction with progress tracking
    - Settings page for API key management
    - Session history and model output management
"""

import cv2
import numpy as np
import os
import sys
import time
import json
import threading
import queue
from pathlib import Path
from tkinter import (
    Tk, Toplevel, Frame, Label, Button, Canvas, Entry,
    OptionMenu, StringVar, IntVar, BooleanVar, Text,
    ttk, messagebox, filedialog, PhotoImage
)
from PIL import Image, ImageTk
from typing import Optional, List, Tuple, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scanner3d.camera import Camera
from scanner3d.scanner import Scanner3D
from scanner3d.config import (
    OUTPUT_DIR, MODEL_OUTPUT_DIR, NUM_ANGLES,
    DROIDCAM_INDEX, KIRI_API_KEY,
    KIRI_BASE_URL
)
from scanner3d.kiri_reconstructor import reconstruct as run_reconstruction

# ─── Configuration File ───────────────────────────────────────────────────────
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui_config.json")

DEFAULT_CONFIG = {
    "camera_index": -1,  # -1 = auto-detect
    "kiri_api_key": KIRI_API_KEY,
    "num_angles": 36,
    "recording_duration": 30,
    "theme": "dark"
}


def load_config() -> dict:
    """Load GUI configuration from JSON file."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        except (json.JSONDecodeError, IOError):
            pass
    return dict(DEFAULT_CONFIG)


def save_config(config: dict):
    """Save GUI configuration to JSON file."""
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
    except IOError as e:
        print(f"[Config] Failed to save: {e}")


# ─── Color Palette & Themes ───────────────────────────────────────────────────
THEMES = {
    "dark": {
        "bg": "#1a1a2e",
        "bg2": "#16213e",
        "bg3": "#0f3460",
        "accent": "#e94560",
        "accent2": "#533483",
        "text": "#eaeaea",
        "text2": "#a0a0b0",
        "success": "#4ecca3",
        "warning": "#ffc857",
        "danger": "#e94560",
        "card": "#1e2a4a",
        "card_border": "#2a3a5a",
        "input_bg": "#0d1b3e",
        "input_border": "#2a3a5a",
        "button_primary": "#e94560",
        "button_primary_hover": "#ff6b81",
        "button_secondary": "#533483",
        "button_success": "#4ecca3",
        "progress_bg": "#2a3a5a",
        "progress_fill": "#e94560",
    },
    "light": {
        "bg": "#f0f2f5",
        "bg2": "#ffffff",
        "bg3": "#e8ecf1",
        "accent": "#e94560",
        "accent2": "#6c5ce7",
        "text": "#2d3436",
        "text2": "#636e72",
        "success": "#00b894",
        "warning": "#fdcb6e",
        "danger": "#e17055",
        "card": "#ffffff",
        "card_border": "#dfe6e9",
        "input_bg": "#ffffff",
        "input_border": "#b2bec3",
        "button_primary": "#e94560",
        "button_primary_hover": "#d63031",
        "button_secondary": "#6c5ce7",
        "button_success": "#00b894",
        "progress_bg": "#dfe6e9",
        "progress_fill": "#e94560",
    }
}


# ─── Camera Stream Thread ────────────────────────────────────────────────────
class CameraStream:
    """Background thread for DroidCam virtual camera frame capture."""

    def __init__(self, camera_index: int = -1):
        self.camera_index = camera_index  # -1 = auto-detect
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

    def open(self) -> bool:
        """Open the DroidCam virtual camera by index."""
        try:
            if self.camera_index >= 0:
                indices = [self.camera_index]
            else:
                indices = list(range(5))

            for idx in indices:
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    self.cap = cap
                    print(f"[CameraStream] Opened camera at index {idx}")
                    return True
                cap.release()
            return False
        except Exception as e:
            print(f"[CameraStream] Error opening camera: {e}")
            return False

    def start(self):
        """Start the capture thread."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the capture thread."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        with self.lock:
            if self.cap:
                self.cap.release()
                self.cap = None
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

    def read(self) -> Optional[np.ndarray]:
        """Get the latest frame from the queue (non-blocking)."""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def _capture_loop(self):
        """Continuously capture frames."""
        while self.running and self.cap:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # Resize for performance
                h, w = frame.shape[:2]
                if w > 640:
                    scale = 640 / w
                    new_w, new_h = int(w * scale), int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                # Replace old frame in queue
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(frame)
            else:
                time.sleep(0.01)

    def release(self):
        """Release resources."""
        self.stop()


# ─── Main Application ─────────────────────────────────────────────────────────
class ScannerGUI:
    """Professional 3D Scanner Desktop Application."""

    def __init__(self):
        self.config = load_config()
        self.theme = THEMES.get(self.config.get("theme", "dark"), THEMES["dark"])
        self.camera_stream: Optional[CameraStream] = None
        self.scanning = False
        self.recording = False
        self.preview_active = False
        self.camera_open = False

        # Build UI
        self.root = Tk()
        self.root.title("3D Scanner Pro - DroidCam")
        self.root.geometry("1100x750")
        self.root.minsize(900, 650)
        self.root.configure(bg=self.theme["bg"])

        # Set icon if available
        try:
            self.root.iconbitmap(default="")
        except:
            pass

        self._setup_styles()
        self._build_ui()
        self._bind_shortcuts()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ─── Style Setup ────────────────────────────────────────────────────────
    def _setup_styles(self):
        """Configure ttk styles for a modern look."""
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("TFrame", background=self.theme["bg"])
        style.configure("Card.TFrame", background=self.theme["card"],
                        relief="solid", borderwidth=1)
        style.configure("Header.TLabel", background=self.theme["bg"],
                        foreground=self.theme["text"],
                        font=("Segoe UI", 16, "bold"))
        style.configure("Title.TLabel", background=self.theme["bg"],
                        foreground=self.theme["text"],
                        font=("Segoe UI", 12, "bold"))
        style.configure("Body.TLabel", background=self.theme["bg"],
                        foreground=self.theme["text2"],
                        font=("Segoe UI", 10))
        style.configure("Status.TLabel", background=self.theme["bg"],
                        foreground=self.theme["accent"],
                        font=("Segoe UI", 10, "bold"))
        style.configure("Success.TLabel", background=self.theme["bg"],
                        foreground=self.theme["success"],
                        font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10))
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Small.TButton", font=("Segoe UI", 9))

        # Notebook styles
        style.configure("TNotebook", background=self.theme["bg"],
                        borderwidth=0)
        style.configure("TNotebook.Tab", background=self.theme["bg2"],
                        foreground=self.theme["text"],
                        padding=[15, 5],
                        font=("Segoe UI", 10))
        style.map("TNotebook.Tab",
                  background=[("selected", self.theme["accent"])],
                  foreground=[("selected", "#ffffff")])

        # Progress bar
        style.configure("TProgressbar", background=self.theme["progress_fill"],
                        troughcolor=self.theme["progress_bg"],
                        borderwidth=0, thickness=8)

        # Scrollbar
        style.configure("TScrollbar", background=self.theme["bg2"],
                        troughcolor=self.theme["bg"],
                        arrowsize=14)

    # ─── UI Construction ────────────────────────────────────────────────────
    def _build_ui(self):
        """Build the complete user interface."""
        # Main container
        self.main_container = Frame(self.root, bg=self.theme["bg"])
        self.main_container.pack(fill="both", expand=True, padx=15, pady=15)

        # ─── Top: Header ────────────────────────────────────────────────────
        self._build_header()

        # ─── Middle: Main Content ───────────────────────────────────────────
        self._build_content()

        # ─── Bottom: Status Bar ─────────────────────────────────────────────
        self._build_status_bar()

    def _build_header(self):
        """Build the header/toolbar area."""
        header_frame = Frame(self.main_container, bg=self.theme["bg"])
        header_frame.pack(fill="x", pady=(0, 10))

        # Logo / Title
        title_frame = Frame(header_frame, bg=self.theme["bg"])
        title_frame.pack(side="left")
        Label(title_frame, text="🔬 3D Scanner Pro",
              bg=self.theme["bg"], fg=self.theme["text"],
              font=("Segoe UI", 20, "bold")).pack(side="left")

        # Subtitle
        Label(title_frame, text=" | DroidCam Scan Suite",
              bg=self.theme["bg"], fg=self.theme["text2"],
              font=("Segoe UI", 11)).pack(side="left", padx=(5, 0))

        # Right side buttons
        btn_frame = Frame(header_frame, bg=self.theme["bg"])
        btn_frame.pack(side="right")

        self._create_button(btn_frame, "⚙ Settings", self._open_settings,
                            self.theme["button_secondary"], self.theme["text"])
        self._create_button(btn_frame, "🌙 Dark" if self.config.get("theme") == "light" else "☀ Light",
                            self._toggle_theme,
                            self.theme["bg3"], self.theme["text"], padx=(0, 8))

        separator = Frame(self.main_container, height=2, bg=self.theme["card_border"])
        separator.pack(fill="x", pady=(0, 10))

    def _build_content(self):
        """Build the main content area."""
        self.content_pane = Frame(self.main_container, bg=self.theme["bg"])
        self.content_pane.pack(fill="both", expand=True)

        # ─── Left: Camera Preview ──────────────────────────────────────────
        left_frame = Frame(self.content_pane, bg=self.theme["bg"])
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Camera Preview Card
        self.preview_card = Frame(left_frame, bg=self.theme["card"],
                                  highlightbackground=self.theme["card_border"],
                                  highlightthickness=1)
        self.preview_card.pack(fill="both", expand=True)

        # Preview header
        preview_header = Frame(self.preview_card, bg=self.theme["card"])
        preview_header.pack(fill="x", padx=12, pady=(10, 5))
        Label(preview_header, text="📷 DroidCam Preview",
              bg=self.theme["card"], fg=self.theme["text"],
              font=("Segoe UI", 12, "bold")).pack(side="left")

        # Camera source indicator
        self.cam_indicator = Label(preview_header, text="● Offline",
                                   bg=self.theme["card"],
                                   fg=self.theme["danger"],
                                   font=("Segoe UI", 9))
        self.cam_indicator.pack(side="right")

        # Camera preview canvas
        preview_container = Frame(self.preview_card, bg=self.theme["bg3"],
                                  highlightbackground=self.theme["card_border"],
                                  highlightthickness=1)
        preview_container.pack(fill="both", expand=True, padx=12, pady=(5, 12))

        self.preview_canvas = Canvas(preview_container, bg="#000000",
                                     highlightthickness=0,
                                     width=640, height=480)
        self.preview_canvas.pack(fill="both", expand=True)

        # Placeholder text on canvas
        self.preview_placeholder = self.preview_canvas.create_text(
            320, 240, text="Camera Offline\n1. Install DroidCam PC Client\n2. Connect to your phone\n3. Click 'Open Camera'",
            fill="#555555", font=("Segoe UI", 11), justify="center", anchor="center"
        )

        if not self.config.get("camera_index") or self.config.get("camera_index") == -1:
            self.preview_canvas.create_text(
                320, 300, text="(Auto-detect: scans indices 0-4)",
                fill="#444444", font=("Segoe UI", 9), justify="center", anchor="center"
            )

        # ─── Right: Controls Panel ────────────────────────────────────────
        right_frame = Frame(self.content_pane, bg=self.theme["bg"])
        right_frame.pack(side="right", fill="y", padx=(10, 0))

        # Controls Card
        controls_card = Frame(right_frame, bg=self.theme["card"],
                              highlightbackground=self.theme["card_border"],
                              highlightthickness=1)
        controls_card.pack(fill="both")

        Label(controls_card, text="🎛 Controls",
              bg=self.theme["card"], fg=self.theme["text"],
              font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=15, pady=(15, 5))

        # ─ Camera Index ──────────────────────────────────────────────────
        source_frame = Frame(controls_card, bg=self.theme["card"])
        source_frame.pack(fill="x", padx=15, pady=5)

        Label(source_frame, text="Camera Index (DroidCam):",
              bg=self.theme["card"], fg=self.theme["text2"],
              font=("Segoe UI", 9)).pack(anchor="w")

        index_frame = Frame(source_frame, bg=self.theme["card"])
        index_frame.pack(fill="x", pady=(2, 0))

        self.camera_index_var = StringVar(
            value=str(self.config.get("camera_index", -1))
        )
        self.camera_index_menu = ttk.Combobox(index_frame,
                                              textvariable=self.camera_index_var,
                                              values=["-1 (Auto)", "0", "1", "2", "3", "4"],
                                              state="normal", width=12)
        self.camera_index_menu.pack(side="left")

        # Detect button
        self.detect_cam_btn = self._create_button(
            index_frame, "🔍 Detect", self._detect_cameras,
            self.theme["button_secondary"], self.theme["text"], side="right"
        )

        # Info label
        Label(source_frame, text="Tip: Run 'Detect' to find which index has your DroidCam feed",
              bg=self.theme["card"], fg=self.theme["text2"],
              font=("Segoe UI", 8)).pack(anchor="w", pady=(4, 0))

        # ─ Open/Close Camera Buttons ──────────────────────────────────────
        cam_btn_frame = Frame(controls_card, bg=self.theme["card"])
        cam_btn_frame.pack(fill="x", padx=15, pady=8)

        self.open_cam_btn = self._create_button(
            cam_btn_frame, "▶ Open Camera", self._toggle_camera,
            self.theme["button_success"], "#ffffff", side="top", expand=True)

        # Separator
        separator = Frame(controls_card, height=1, bg=self.theme["card_border"])
        separator.pack(fill="x", padx=15, pady=8)

        # ─ Scan Controls ──────────────────────────────────────────────────
        scan_frame = Frame(controls_card, bg=self.theme["card"])
        scan_frame.pack(fill="x", padx=15, pady=5)

        Label(scan_frame, text="Scan Settings:",
              bg=self.theme["card"], fg=self.theme["text2"],
              font=("Segoe UI", 9)).pack(anchor="w")

        # Number of angles
        angle_row = Frame(scan_frame, bg=self.theme["card"])
        angle_row.pack(fill="x", pady=3)
        Label(angle_row, text="Angles:",
              bg=self.theme["card"], fg=self.theme["text"],
              font=("Segoe UI", 9)).pack(side="left")
        self.angles_var = StringVar(value=str(self.config.get("num_angles", 36)))
        angles_entry = Entry(angle_row, textvariable=self.angles_var,
                             bg=self.theme["input_bg"], fg=self.theme["text"],
                             relief="flat", width=8,
                             font=("Segoe UI", 9))
        angles_entry.pack(side="right")
        self._style_entry(angles_entry)

        # Duration
        dur_row = Frame(scan_frame, bg=self.theme["card"])
        dur_row.pack(fill="x", pady=3)
        Label(dur_row, text="Duration (s):",
              bg=self.theme["card"], fg=self.theme["text"],
              font=("Segoe UI", 9)).pack(side="left")
        self.duration_var = StringVar(value=str(self.config.get("recording_duration", 30)))
        dur_entry = Entry(dur_row, textvariable=self.duration_var,
                          bg=self.theme["input_bg"], fg=self.theme["text"],
                          relief="flat", width=8,
                          font=("Segoe UI", 9))
        dur_entry.pack(side="right")
        self._style_entry(dur_entry)

        # ─ Action Buttons ────────────────────────────────────────────────
        separator2 = Frame(controls_card, height=1, bg=self.theme["card_border"])
        separator2.pack(fill="x", padx=15, pady=8)

        action_frame = Frame(controls_card, bg=self.theme["card"])
        action_frame.pack(fill="x", padx=15, pady=(5, 15))

        self.start_scan_btn = self._create_button(
            action_frame, "▶ START SCAN", self._start_scan,
            self.theme["button_primary"], "#ffffff",
            height=40, font_size=12, expand=True, side="top")

        self.stop_scan_btn = self._create_button(
            action_frame, "⬛ STOP", self._stop_scan,
            self.theme["danger"], "#ffffff",
            height=35, expand=True, side="top", enabled=False)
        self.stop_scan_btn.pack(fill="x", pady=(5, 0))

        # ─ Quick Actions ─────────────────────────────────────────────────
        quick_frame = Frame(right_frame, bg=self.theme["card"],
                            highlightbackground=self.theme["card_border"],
                            highlightthickness=1)
        quick_frame.pack(fill="x", pady=(10, 0))

        Label(quick_frame, text="⚡ Quick Actions",
              bg=self.theme["card"], fg=self.theme["text"],
              font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=15, pady=(15, 5))

        self._create_button(quick_frame, "🔄 Reconstruct (Kiri Engine)",
                           self._run_reconstruction,
                           self.theme["accent2"], "#ffffff",
                           expand=True, side="top", padx=15)
        self._create_button(quick_frame, "📁 Open Captures Folder",
                           self._open_captures,
                           self.theme["bg3"], self.theme["text"],
                           expand=True, side="top", padx=15,
                           pady=(5, 15))

    def _build_status_bar(self):
        """Build the bottom status bar."""
        status_frame = Frame(self.main_container, bg=self.theme["bg2"],
                             highlightbackground=self.theme["card_border"],
                             highlightthickness=1)
        status_frame.pack(fill="x", pady=(10, 0))

        # Status text
        self.status_text = StringVar(value="🟢 Ready — Open DroidCam to begin")
        status_label = Label(status_frame, textvariable=self.status_text,
                             bg=self.theme["bg2"], fg=self.theme["text2"],
                             font=("Segoe UI", 9))
        status_label.pack(side="left", padx=12, pady=8)

        # Progress bar
        self.progress_var = IntVar(value=0)
        self.progress_bar = ttk.Progressbar(status_frame,
                                            variable=self.progress_var,
                                            length=200, mode="determinate")
        self.progress_bar.pack(side="right", padx=12, pady=8)

        # Version label
        Label(status_frame, text="v2.0",
              bg=self.theme["bg2"], fg=self.theme["text2"],
              font=("Segoe UI", 8)).pack(side="right", padx=(0, 12))

    # ─── Helper: Styled Button ─────────────────────────────────────────────
    def _create_button(self, parent, text, command, bg, fg,
                       side="left", padx=0, pady=0, height=None,
                       font_size=10, expand=False, enabled=True):
        """Create a custom styled button."""
        btn = Button(parent, text=text, command=command,
                     bg=bg, fg=fg,
                     activebackground=self._lighten(bg, 20),
                     activeforeground=fg,
                     relief="flat", bd=0,
                     font=("Segoe UI", font_size),
                     padx=15, pady=6 if height is None else 0,
                     cursor="hand2",
                     state="normal" if enabled else "disabled")
        btn.pack(side=side, padx=padx, pady=pady, fill="x" if expand else "none",
                 expand=expand)
        if height:
            btn.config(height=height // 6)
        self._style_button(btn, bg)
        return btn

    def _style_button(self, btn, bg):
        """Apply hover effects to a button."""

        def on_enter(e):
            if btn["state"] != "disabled":
                btn["bg"] = self._lighten(bg, 20)

        def on_leave(e):
            if btn["state"] != "disabled":
                btn["bg"] = bg

        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)

    def _style_entry(self, entry):
        """Apply styling to an entry widget."""
        entry.configure(insertbackground=self.theme["text"],
                        selectbackground=self.theme["accent"],
                        selectforeground="#ffffff")
        entry.configure(highlightbackground=self.theme["input_border"],
                        highlightcolor=self.theme["accent"],
                        highlightthickness=1)

    def _lighten(self, color, amount=30):
        """Lighten a hex color by amount (0-255)."""
        color = color.lstrip('#')
        if len(color) != 6:
            return color
        r = min(255, int(color[0:2], 16) + amount)
        g = min(255, int(color[2:4], 16) + amount)
        b = min(255, int(color[4:6], 16) + amount)
        return f"#{r:02x}{g:02x}{b:02x}"

    # ─── Event Handlers ────────────────────────────────────────────────────
    def _toggle_camera(self):
        """Open or close the camera."""
        if self.camera_open:
            self._close_camera()
        else:
            self._open_camera()

    def _open_camera(self):
        """Open the DroidCam virtual camera and start preview."""
        if self.camera_open:
            return

        # Parse camera index
        raw = self.camera_index_var.get().strip()
        try:
            camera_index = int(raw.split()[0])  # Handle "-1 (Auto)" format
        except (ValueError, IndexError):
            camera_index = -1

        self.set_status("Opening DroidCam...", "status")
        self.root.update()

        # Create and open camera stream
        self.camera_stream = CameraStream(camera_index=camera_index)

        if not self.camera_stream.open():
            messagebox.showerror("Camera Error",
                                 "Failed to open DroidCam virtual camera.\n\n"
                                 "Troubleshooting:\n"
                                 "1. Install DroidCam PC Client (https://droidcam.app)\n"
                                 "2. Connect PC Client to your phone (same WiFi)\n"
                                 "3. Try clicking 'Detect' to find the right camera index\n"
                                 "4. Make sure no other app is using the camera")
            self.camera_stream = None
            self.set_status("DroidCam connection failed", "error")
            return

        self.camera_stream.start()
        self.camera_open = True
        self.open_cam_btn.config(text="■ Close Camera",
                                 bg=self.theme["danger"])
        self.cam_indicator.config(text="● Live", fg=self.theme["success"])
        self.set_status("DroidCam connected — Position object and click START SCAN")

        # Start preview update loop
        self._update_preview()

    def _close_camera(self):
        """Close the camera and stop preview."""
        if self.camera_stream:
            self.camera_stream.release()
            self.camera_stream = None
        self.camera_open = False
        self.open_cam_btn.config(text="▶ Open Camera",
                                 bg=self.theme["button_success"])
        self.cam_indicator.config(text="● Offline", fg=self.theme["danger"])

        # Clear canvas
        self.preview_canvas.delete("all")
        self.preview_canvas.create_text(
            320, 240, text="Camera Offline\n1. Install DroidCam PC Client\n2. Connect to your phone\n3. Click 'Open Camera'",
            fill="#555555", font=("Segoe UI", 11), justify="center", anchor="center"
        )
        self.set_status("Camera closed")

    def _update_preview(self):
        """Update the camera preview in the canvas."""
        if not self.camera_open or not self.camera_stream:
            return

        frame = self.camera_stream.read()
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)

            canvas_w = self.preview_canvas.winfo_width()
            canvas_h = self.preview_canvas.winfo_height()
            if canvas_w > 10 and canvas_h > 10:
                img_w, img_h = img.size
                scale = min(canvas_w / img_w, canvas_h / img_h)
                new_w, new_h = int(img_w * scale), int(img_h * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)

            photo = ImageTk.PhotoImage(img)
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(
                canvas_w // 2, canvas_h // 2, image=photo, anchor="center"
            )
            self.preview_canvas.image = photo

        self.root.after(30, self._update_preview)

    def _detect_cameras(self):
        """Run camera detection in a thread and show results."""
        self.set_status("Scanning for cameras...")

        def detect():
            available = Camera.list_cameras(max_indices=10)
            self.root.after(0, lambda: self._show_detection_results(available))

        threading.Thread(target=detect, daemon=True).start()

    def _show_detection_results(self, cameras: list):
        """Show camera detection results in a dialog."""
        if not cameras:
            messagebox.showinfo("Camera Detection",
                                "No cameras found.\n\n"
                                "1. Install DroidCam PC Client\n"
                                "2. Connect it to your phone\n"
                                "3. Click 'Detect' again")
            self.set_status("No cameras detected")
            return

        msg = f"Found {len(cameras)} camera(s):\n\n"
        for idx, name in cameras:
            msg += f"  ✓ Index {idx}: {name}\n"
        msg += "\nEnter the index number in 'Camera Index' field above."
        msg += "\n(DroidCam typically appears at index 1 or 2)"

        messagebox.showinfo("Camera Detection Results", msg)
        self.set_status(f"Found {len(cameras)} camera(s)")

    def _start_scan(self):
        """Start the scanning process."""
        if not self.camera_open:
            messagebox.showwarning("Camera Required",
                                   "Please open the DroidCam camera first.")
            return

        if self.scanning:
            return

        self.scanning = True
        self.start_scan_btn.config(text="⏳ Scanning...", state="disabled")
        self.stop_scan_btn.config(state="normal")
        self.set_status("Starting scan session...")

        try:
            num_angles = int(self.angles_var.get())
            duration = int(self.duration_var.get())
        except ValueError:
            messagebox.showerror("Invalid Settings",
                                 "Please enter valid numbers for angles and duration.")
            self._reset_scan_buttons()
            return

        # Close preview camera so Scanner3D can use the camera
        self._close_camera()
        self.root.update()

        def scan_worker():
            try:
                self.root.after(0, lambda: self.set_status(
                    "Opening DroidCam for scan session...", "status"))
                self.root.after(0, lambda: self._update_progress(5))

                scanner = Scanner3D()

                self.root.after(0, lambda: self.set_status(
                    "Step 1/3: Position object in camera view — press 'q' when ready"))
                self.root.after(0, lambda: self._update_progress(15))
                scanner.run_preview()

                self.root.after(0, lambda: self.set_status(
                    "Step 2/3: Recording — rotate the object 360° slowly..."))
                self.root.after(0, lambda: self._update_progress(30))

                session_id = time.strftime("%Y%m%d_%H%M%S")
                session_dir = Path(OUTPUT_DIR) / session_id
                session_dir.mkdir(parents=True, exist_ok=True)

                video_path = str(session_dir / "scan_video")
                recorded = scanner.camera.record_video(
                    video_path, duration_seconds=duration
                )

                if not recorded:
                    raise RuntimeError("Video recording failed")

                self.root.after(0, lambda: self._update_progress(60))
                self.root.after(0, lambda: self.set_status(
                    "Step 3/3: Extracting frames from video..."))

                extracted = scanner.camera.extract_frames_from_video(
                    recorded, num_frames=num_angles,
                    output_dir=str(session_dir)
                )

                self.root.after(0, lambda: self._update_progress(90))
                scanner.camera.release()
                self.root.after(0, lambda: self._update_progress(100))

                if extracted:
                    self.root.after(0, lambda: self._on_scan_complete(
                        len(extracted), session_id, str(session_dir)))
                else:
                    raise RuntimeError("No frames extracted")

            except Exception as e:
                self.root.after(0, lambda err=e: self._on_scan_error(err))

        threading.Thread(target=scan_worker, daemon=True).start()

    def _on_scan_complete(self, num_images: int, session_id: str, session_dir: str):
        """Handle successful scan completion."""
        self._update_progress(100)
        self.set_status(f"✓ Scan complete! {num_images} images captured")
        self.start_scan_btn.config(text="▶ START SCAN", state="normal")
        self.stop_scan_btn.config(state="disabled")
        self.scanning = False

        self.config["last_session"] = session_id
        save_config(self.config)

        result = messagebox.askyesno(
            "Scan Complete",
            f"✓ Successfully captured {num_images} images!\n\n"
            f"Session: {session_id}\n"
            f"Location: {session_dir}\n\n"
            "Would you like to open the captures folder?"
        )
        if result:
            os.startfile(session_dir)

    def _on_scan_error(self, error: Exception):
        """Handle scan error."""
        self._reset_scan_buttons()
        self.set_status(f"✗ Scan failed: {str(error)}", "error")
        messagebox.showerror("Scan Error",
                             f"Scan failed:\n{str(error)}\n\n"
                             "Check DroidCam connection and try again.")

    def _reset_scan_buttons(self):
        """Reset scan buttons to idle state."""
        self.start_scan_btn.config(text="▶ START SCAN", state="normal")
        self.stop_scan_btn.config(state="disabled")
        self.scanning = False
        self._update_progress(0)

    def _stop_scan(self):
        """Stop an ongoing scan."""
        if self.scanning:
            self.set_status("Stopping scan... (will finish current step)")
            self.scanning = False
            self._reset_scan_buttons()

    def _run_reconstruction(self):
        """Run Kiri Engine reconstruction."""
        last_session = self.config.get("last_session", "")
        sessions = self._get_sessions()

        if not sessions:
            messagebox.showwarning("No Sessions",
                                   "No scan sessions found.\n\n"
                                   "Complete a scan first before reconstructing.")
            return

        session_id = last_session if last_session in sessions else sessions[-1]
        self._open_reconstruct_dialog(session_id, sessions)

    def _get_sessions(self) -> List[str]:
        """Get list of scan sessions."""
        output_dir = Path(OUTPUT_DIR)
        if not output_dir.exists():
            return []
        sessions = [d.name for d in output_dir.iterdir()
                    if d.is_dir() and d.name.startswith("20")]
        return sorted(sessions)

    def _open_reconstruct_dialog(self, default_session: str, sessions: List[str]):
        """Open reconstruction options dialog."""
        dialog = Toplevel(self.root)
        dialog.title("Kiri Engine Reconstruction")
        dialog.geometry("500x600")
        dialog.configure(bg=self.theme["bg"])
        dialog.transient(self.root)
        dialog.grab_set()

        main_frame = Frame(dialog, bg=self.theme["bg"], padx=20, pady=20)
        main_frame.pack(fill="both", expand=True)

        Label(main_frame, text="🔄 3D Reconstruction",
              bg=self.theme["bg"], fg=self.theme["text"],
              font=("Segoe UI", 16, "bold")).pack(pady=(0, 15))

        session_frame = Frame(main_frame, bg=self.theme["bg"])
        session_frame.pack(fill="x", pady=5)

        Label(session_frame, text="Session:",
              bg=self.theme["bg"], fg=self.theme["text2"],
              font=("Segoe UI", 9)).pack(anchor="w")

        self.recon_session_var = StringVar(value=default_session)
        session_menu = ttk.Combobox(session_frame,
                                     textvariable=self.recon_session_var,
                                     values=sessions,
                                     state="readonly", width=35)
        session_menu.pack(fill="x", pady=(2, 0))

        name_frame = Frame(main_frame, bg=self.theme["bg"])
        name_frame.pack(fill="x", pady=5)

        Label(name_frame, text="Model Name (optional):",
              bg=self.theme["bg"], fg=self.theme["text2"],
              font=("Segoe UI", 9)).pack(anchor="w")

        self.recon_name_var = StringVar(value="")
        name_entry = Entry(name_frame, textvariable=self.recon_name_var,
                           bg=self.theme["input_bg"], fg=self.theme["text"],
                           relief="flat", font=("Segoe UI", 10))
        name_entry.pack(fill="x", pady=(2, 0))
        self._style_entry(name_entry)

        qual_frame = Frame(main_frame, bg=self.theme["bg"])
        qual_frame.pack(fill="x", pady=5)

        Label(qual_frame, text="Reconstruction Quality:",
              bg=self.theme["bg"], fg=self.theme["text2"],
              font=("Segoe UI", 9)).pack(anchor="w")

        self.recon_quality_var = StringVar(value="high")
        quality_menu = ttk.Combobox(qual_frame,
                                     textvariable=self.recon_quality_var,
                                     values=["draft", "medium", "high", "ultra"],
                                     state="readonly", width=15)
        quality_menu.pack(fill="x", pady=(2, 0))

        info_text = (
            "The captured images will be uploaded to Kiri Engine API\n"
            "for cloud-based 3D reconstruction.\n\n"
            "• Draft: Fast (~2 min)\n"
            "• Medium: Balanced (~5 min)\n"
            "• High: Quality (~15 min)\n"
            "• Ultra: Best quality (~30 min)\n\n"
            "Times are approximate."
        )
        info_label = Label(main_frame, text=info_text,
                           bg=self.theme["bg"], fg=self.theme["text2"],
                           font=("Segoe UI", 9), justify="left")
        info_label.pack(pady=10)

        btn_frame = Frame(main_frame, bg=self.theme["bg"])
        btn_frame.pack(fill="x", pady=(10, 0))

        self._create_button(btn_frame, "Start Reconstruction",
                           lambda: self._start_reconstruction(dialog),
                           self.theme["button_primary"], "#ffffff",
                           expand=True, side="top", height=38, font_size=11)

        self._create_button(btn_frame, "Cancel",
                           dialog.destroy,
                           self.theme["bg3"], self.theme["text"],
                           expand=True, side="top", padx=0, pady=(5, 0))

    def _start_reconstruction(self, dialog):
        """Start the reconstruction process."""
        session_id = self.recon_session_var.get()
        output_name = self.recon_name_var.get().strip() or None
        quality = self.recon_quality_var.get()

        dialog.destroy()

        self.set_status(f"Starting Kiri Engine reconstruction ({quality})...")
        self._update_progress(5)
        self.root.update()

        def recon_worker():
            try:
                result = run_reconstruction(
                    session_id=session_id,
                    output_name=output_name,
                    quality=quality,
                    progress_callback=lambda p: self.root.after(
                        0, lambda: self._update_progress(p))
                )

                self.root.after(0, lambda: self._update_progress(100))

                if result:
                    self.root.after(0, lambda: self.set_status(
                        f"✓ Reconstruction complete! Model: {result}", "success"))
                    self.root.after(0, lambda: messagebox.showinfo(
                        "Reconstruction Complete",
                        f"3D model created successfully!\n\n{result}"))
                else:
                    raise RuntimeError("Reconstruction returned no result")

            except Exception as e:
                self.root.after(0, lambda: self.set_status(
                    f"✗ Reconstruction failed: {str(e)}", "error"))
                self.root.after(0, lambda: messagebox.showerror(
                    "Reconstruction Error",
                    f"Kiri Engine reconstruction failed:\n{str(e)}\n\n"
                    "Check your API key and internet connection."))

        threading.Thread(target=recon_worker, daemon=True).start()

    def _open_captures(self):
        """Open the captures folder in file explorer."""
        captures_path = Path(OUTPUT_DIR)
        if captures_path.exists():
            os.startfile(str(captures_path))
        else:
            messagebox.showwarning("Not Found",
                                   "No captures folder found yet. Complete a scan first.")

    def _open_settings(self):
        """Open settings dialog."""
        dialog = Toplevel(self.root)
        dialog.title("Settings")
        dialog.geometry("450x350")
        dialog.configure(bg=self.theme["bg"])
        dialog.transient(self.root)
        dialog.grab_set()

        main_frame = Frame(dialog, bg=self.theme["bg"], padx=20, pady=20)
        main_frame.pack(fill="both", expand=True)

        Label(main_frame, text="⚙ Settings",
              bg=self.theme["bg"], fg=self.theme["text"],
              font=("Segoe UI", 16, "bold")).pack(pady=(0, 20))

        # Kiri API Key
        api_frame = Frame(main_frame, bg=self.theme["bg"])
        api_frame.pack(fill="x", pady=5)

        Label(api_frame, text="Kiri Engine API Key:",
              bg=self.theme["bg"], fg=self.theme["text2"],
              font=("Segoe UI", 9)).pack(anchor="w")

        settings_api_var = StringVar(value=self.config.get("kiri_api_key", ""))
        api_entry = Entry(api_frame, textvariable=settings_api_var,
                          bg=self.theme["input_bg"], fg=self.theme["text"],
                          relief="flat", font=("Segoe UI", 10))
        api_entry.pack(fill="x", pady=(2, 0))
        self._style_entry(api_entry)

        # Save button
        def save_settings():
            # Save camera index
            raw = self.camera_index_var.get().strip()
            try:
                idx = int(raw.split()[0])
            except (ValueError, IndexError):
                idx = -1
            self.config["camera_index"] = idx
            self.config["kiri_api_key"] = settings_api_var.get().strip()
            save_config(self.config)
            self.set_status("Settings saved")
            dialog.destroy()

        self._create_button(main_frame, "Save Settings",
                           save_settings,
                           self.theme["button_success"], "#ffffff",
                           expand=True, side="top", height=38, font_size=11)
        self._create_button(main_frame, "Cancel",
                           dialog.destroy,
                           self.theme["bg3"], self.theme["text"],
                           expand=True, side="top", padx=0, pady=(5, 0))

    def _toggle_theme(self):
        """Toggle between dark and light theme."""
        new_theme = "light" if self.config.get("theme") == "dark" else "dark"
        self.config["theme"] = new_theme
        save_config(self.config)

        self.theme = THEMES[new_theme]
        self._close_camera()
        self.root.destroy()
        self.__init__()
        self.run()

    def _bind_shortcuts(self):
        """Bind keyboard shortcuts."""
        self.root.bind("<Escape>", lambda e: self._on_close())

    def _update_progress(self, value: int):
        """Update the progress bar."""
        self.progress_var.set(value)
        self.root.update_idletasks()

    def set_status(self, text: str, level: str = "info"):
        """Set the status bar text."""
        colors = {"info": self.theme["text2"], "status": self.theme["accent"],
                  "success": self.theme["success"], "error": self.theme["danger"]}
        self.status_text.set(text)
        self.root.update_idletasks()

    def _on_close(self):
        """Handle application close."""
        if self.scanning:
            if not messagebox.askokcancel("Exit", "A scan is in progress. Exit anyway?"):
                return
        self._close_camera()
        self.root.destroy()

    def run(self):
        """Start the application main loop."""
        self.root.mainloop()


# ─── Entry Point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = ScannerGUI()
    app.run()