"""
3D Scanner Pro - IP Webcam GUI Application
===========================================
A professional desktop interface for 3D scanning using your phone's camera.

Quick Start:
  1. Install "IP Webcam" from Play Store on your phone
  2. Open the app → tap "Start Server"
  3. Note the URL shown (e.g., http://10.138.159.186:8080)
  4. Run: python gui_app.py
  5. The app auto-connects to your phone camera!

Features:
    - Auto-connect to phone camera via WiFi
    - Live MJPEG video stream preview
    - One-click 3D scan (record → extract frames)
    - Kiri Engine cloud reconstruction
    - Session history management
    - Dark/Light theme
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
    StringVar, IntVar, BooleanVar, ttk, messagebox
)
from PIL import Image, ImageTk
from typing import Optional, List, Tuple, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scanner3d.camera import Camera
from scanner3d.phone_camera import PhoneCamera
from scanner3d.scanner import Scanner3D
from scanner3d.droidcam_remote import PhoneRemote
from scanner3d.config import (
    OUTPUT_DIR, MODEL_OUTPUT_DIR, NUM_ANGLES,
    KIRI_API_KEY, KIRI_BASE_URL,
    IP_WEBCAM_URL, IP_WEBCAM_STREAM_PATH
)
from scanner3d.kiri_reconstructor import reconstruct as run_reconstruction

# ─── Configuration ───────────────────────────────────────────────────────────
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui_config.json")

DEFAULT_CONFIG = {
    "webcam_url": IP_WEBCAM_URL,
    "kiri_api_key": KIRI_API_KEY,
    "num_angles": 70,
    "recording_duration": 30,
    "theme": "dark",
    "resolution": "1280x720"
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


# ─── Themes ──────────────────────────────────────────────────────────────────
THEMES = {
    "dark": {
        "bg": "#0d1117", "bg2": "#161b22", "bg3": "#21262d",
        "accent": "#58a6ff", "accent2": "#8b5cf6",
        "text": "#e6edf3", "text2": "#8b949e",
        "success": "#3fb950", "warning": "#d29922", "danger": "#f85149",
        "card": "#161b22", "card_border": "#30363d",
        "input_bg": "#0d1117", "input_border": "#30363d",
        "btn_primary": "#238636", "btn_primary_hover": "#2ea043",
        "btn_secondary": "#21262d",
    },
    "light": {
        "bg": "#ffffff", "bg2": "#f6f8fa", "bg3": "#e1e4e8",
        "accent": "#0969da", "accent2": "#8250df",
        "text": "#24292f", "text2": "#57606a",
        "success": "#1a7f37", "warning": "#9a6700", "danger": "#cf222e",
        "card": "#ffffff", "card_border": "#d0d7de",
        "input_bg": "#ffffff", "input_border": "#d0d7de",
        "btn_primary": "#2da44e", "btn_primary_hover": "#2c974b",
        "btn_secondary": "#f6f8fa",
    }
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

    def open(self) -> bool:
        """Open the IP Webcam MJPEG stream."""
        print(f"[Camera] Connecting to IP Webcam: {self.stream_url}")
        try:
            self.cap = cv2.VideoCapture(self.stream_url)
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    self._resolution = f"{w}x{h}"
                    self.source_name = f"IP Webcam ({self._resolution})"
                    print(f"[Camera] ✓ Connected! {self._resolution}")
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
        while self.running and self.cap:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                if w > 640:
                    scale = 640 / w
                    new_w, new_h = int(w * scale), int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
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


# ─── Main Application ──────────────────────────────────────────────────────
class ScannerGUI:
    """Professional 3D Scanner Desktop Application."""

    def __init__(self):
        self.config = load_config()
        self.theme = THEMES.get(self.config.get("theme", "dark"), THEMES["dark"])
        self.camera_stream: Optional[CameraStream] = None
        self.camera_open = False
        self.scanning = False
        self._auto_connected = False

        # Build UI
        self.root = Tk()
        self.root.title("3D Scanner Pro")
        self.root.geometry("1200x780")
        self.root.minsize(960, 650)
        self.root.configure(bg=self.theme["bg"])

        self._setup_styles()
        self._build_ui()
        self._bind_shortcuts()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Auto-connect after UI loads
        self.root.after(800, self._auto_connect)

    # ─── Styles ───────────────────────────────────────────────────────────
    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        bg = self.theme["bg"]
        
        style.configure("TFrame", background=bg)
        style.configure("Card.TFrame", background=self.theme["card"], relief="solid", borderwidth=1)
        style.configure("Header.TLabel", background=bg, foreground=self.theme["text"],
                        font=("Segoe UI", 18, "bold"))
        style.configure("Title.TLabel", background=bg, foreground=self.theme["text"],
                        font=("Segoe UI", 12, "bold"))
        style.configure("Body.TLabel", background=bg, foreground=self.theme["text2"],
                        font=("Segoe UI", 10))
        style.configure("Status.TLabel", background=bg, foreground=self.theme["accent"],
                        font=("Segoe UI", 10, "bold"))
        style.configure("Success.TLabel", background=bg, foreground=self.theme["success"],
                        font=("Segoe UI", 10))
        style.configure("TProgressbar", background=self.theme["accent"],
                        troughcolor=self.theme["bg3"], borderwidth=0, thickness=6)

    # ─── UI ───────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Main container with padding
        root_frame = Frame(self.root, bg=self.theme["bg"])
        root_frame.pack(fill="both", expand=True, padx=20, pady=15)

        # ─── Top Header ─────────────────────────────────────────────────
        header = Frame(root_frame, bg=self.theme["bg"])
        header.pack(fill="x", pady=(0, 12))

        # Title with icon
        title_frame = Frame(header, bg=self.theme["bg"])
        title_frame.pack(side="left")
        Label(title_frame, text="🔬", bg=self.theme["bg"], fg=self.theme["text"],
              font=("Segoe UI", 24)).pack(side="left")
        Label(title_frame, text="3D Scanner Pro", bg=self.theme["bg"],
              fg=self.theme["text"], font=("Segoe UI", 20, "bold")).pack(side="left", padx=(8, 4))
        Label(title_frame, text="| IP Webcam", bg=self.theme["bg"],
              fg=self.theme["text2"], font=("Segoe UI", 12)).pack(side="left")

        # Right buttons
        btn_frame = Frame(header, bg=self.theme["bg"])
        btn_frame.pack(side="right")
        self._btn(btn_frame, "⚙ Settings", self._open_settings, self.theme["btn_secondary"])
        self._btn(btn_frame, "🌙" if self.config.get("theme") == "light" else "☀",
                  self._toggle_theme, self.theme["btn_secondary"], padx=(0, 4))

        # Divider
        Frame(root_frame, height=2, bg=self.theme["card_border"]).pack(fill="x", pady=(0, 12))

        # ─── Main Content ───────────────────────────────────────────────
        content = Frame(root_frame, bg=self.theme["bg"])
        content.pack(fill="both", expand=True)

        # Left: Preview
        left = Frame(content, bg=self.theme["bg"])
        left.pack(side="left", fill="both", expand=True, padx=(0, 12))

        self._build_preview(left)

        # Right: Controls
        right = Frame(content, bg=self.theme["bg"])
        right.pack(side="right", fill="y")

        self._build_controls(right)

        # ─── Bottom Status ──────────────────────────────────────────────
        self._build_status(root_frame)

    def _build_preview(self, parent):
        """Build the camera preview panel."""
        card = Frame(parent, bg=self.theme["card"],
                     highlightbackground=self.theme["card_border"], highlightthickness=1)
        card.pack(fill="both", expand=True)

        # Header
        hdr = Frame(card, bg=self.theme["card"])
        hdr.pack(fill="x", padx=14, pady=(12, 6))
        Label(hdr, text="📷 Live Preview", bg=self.theme["card"],
              fg=self.theme["text"], font=("Segoe UI", 13, "bold")).pack(side="left")

        self.cam_badge = Label(hdr, text="● Offline", bg=self.theme["card"],
                               fg=self.theme["danger"], font=("Segoe UI", 9))
        self.cam_badge.pack(side="right")

        # Canvas
        container = Frame(card, bg=self.theme["bg3"],
                          highlightbackground=self.theme["card_border"], highlightthickness=1)
        container.pack(fill="both", expand=True, padx=14, pady=(4, 14))

        self.canvas = Canvas(container, bg="#000000", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # Connection status overlay
        self._show_placeholder("Initializing...\nConnecting to phone camera")

        # Connection info
        self.conn_info = Label(card, text=f"Server: {self.config.get('webcam_url', IP_WEBCAM_URL)}",
                               bg=self.theme["card"], fg=self.theme["text2"],
                               font=("Segoe UI", 8))
        self.conn_info.pack(anchor="w", padx=14, pady=(0, 10))

    def _build_controls(self, parent):
        """Build the controls panel."""
        card = Frame(parent, bg=self.theme["card"],
                     highlightbackground=self.theme["card_border"], highlightthickness=1)
        card.pack(fill="y")

        Label(card, text="🎛 Controls", bg=self.theme["card"],
              fg=self.theme["text"], font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=16, pady=(14, 8))

        # ── Status Info ─────────────────────────────────────────────────
        info_frame = Frame(card, bg=self.theme["card"])
        info_frame.pack(fill="x", padx=16, pady=4)

        self.status_rows = {}
        for label, key in [("📡 Server:", "server"), ("📱 Device:", "device"),
                           ("🔋 Battery:", "battery"), ("📐 Resolution:", "resolution")]:
            row = Frame(info_frame, bg=self.theme["card"])
            row.pack(fill="x", pady=1)
            Label(row, text=label, bg=self.theme["card"], fg=self.theme["text2"],
                  font=("Segoe UI", 9), width=11, anchor="w").pack(side="left")
            val = Label(row, text="—", bg=self.theme["card"], fg=self.theme["text"],
                        font=("Segoe UI", 9), anchor="w")
            val.pack(side="left", fill="x", expand=True)
            self.status_rows[key] = val

        # ── Separator ──────────────────────────────────────────────────
        Frame(card, height=1, bg=self.theme["card_border"]).pack(fill="x", padx=16, pady=8)

        # ── Scan Settings ──────────────────────────────────────────────
        Label(card, text="Scan Settings:", bg=self.theme["card"],
              fg=self.theme["text2"], font=("Segoe UI", 9)).pack(anchor="w", padx=16)

        # Angles
        arow = Frame(card, bg=self.theme["card"])
        arow.pack(fill="x", padx=16, pady=3)
        Label(arow, text="Angles:", bg=self.theme["card"], fg=self.theme["text"],
              font=("Segoe UI", 9)).pack(side="left")
        self.angles_var = StringVar(value=str(self.config.get("num_angles", 70)))
        e = Entry(arow, textvariable=self.angles_var, bg=self.theme["input_bg"],
                  fg=self.theme["text"], relief="flat", width=8, font=("Segoe UI", 9))
        e.pack(side="right")
        self._style_entry(e)

        # Duration
        drow = Frame(card, bg=self.theme["card"])
        drow.pack(fill="x", padx=16, pady=3)
        Label(drow, text="Duration (s):", bg=self.theme["card"], fg=self.theme["text"],
              font=("Segoe UI", 9)).pack(side="left")
        self.dur_var = StringVar(value=str(self.config.get("recording_duration", 30)))
        e2 = Entry(drow, textvariable=self.dur_var, bg=self.theme["input_bg"],
                   fg=self.theme["text"], relief="flat", width=8, font=("Segoe UI", 9))
        e2.pack(side="right")
        self._style_entry(e2)

        # ── Separator ──────────────────────────────────────────────────
        Frame(card, height=1, bg=self.theme["card_border"]).pack(fill="x", padx=16, pady=8)

        # ── Action Buttons ─────────────────────────────────────────────
        self.scan_btn = self._btn(card, "▶ START 3D SCAN", self._start_scan,
                                  self.theme["btn_primary"], fg="#ffffff",
                                  height=42, font_size=12, expand=True, side="top",
                                  padx=16, pady=(0, 4))
        self.stop_btn = self._btn(card, "⬛ STOP", self._stop_scan,
                                  self.theme["danger"], fg="#ffffff",
                                  height=36, expand=True, side="top",
                                  padx=16, enabled=False)

        # ── Quick Actions ──────────────────────────────────────────────
        Frame(card, height=1, bg=self.theme["card_border"]).pack(fill="x", padx=16, pady=8)

        self._btn(card, "🔄 Reconstruct (Kiri Engine)", self._run_reconstruction,
                  self.theme["accent2"], fg="#ffffff", expand=True, side="top", padx=16)
        self._btn(card, "📁 Open Captures", self._open_captures,
                  self.theme["btn_secondary"], expand=True, side="top", padx=16,
                  pady=(4, 14))

    def _build_status(self, parent):
        """Build the bottom status bar."""
        bar = Frame(parent, bg=self.theme["bg2"],
                    highlightbackground=self.theme["card_border"], highlightthickness=1)
        bar.pack(fill="x", pady=(12, 0))

        self.status_text = StringVar(value="🟢 Ready")
        Label(bar, textvariable=self.status_text, bg=self.theme["bg2"],
              fg=self.theme["text2"], font=("Segoe UI", 9)).pack(side="left", padx=14, pady=8)

        self.progress_var = IntVar(value=0)
        self.progress_bar = ttk.Progressbar(bar, variable=self.progress_var,
                                            length=180, mode="determinate")
        self.progress_bar.pack(side="right", padx=14, pady=8)

        Label(bar, text="v3.0", bg=self.theme["bg2"], fg=self.theme["text2"],
              font=("Segoe UI", 8)).pack(side="right", padx=(0, 14))

    # ─── Helpers ─────────────────────────────────────────────────────────
    def _btn(self, parent, text, cmd, bg, fg=None, side="left", padx=0, pady=0,
             height=None, font_size=10, expand=False, enabled=True):
        fg = fg or self.theme["text"]
        btn = Button(parent, text=text, command=cmd, bg=bg, fg=fg,
                     activebackground=self._lighten(bg), activeforeground=fg,
                     relief="flat", bd=0, font=("Segoe UI", font_size),
                     padx=14, pady=6 if height is None else 0,
                     cursor="hand2", state="normal" if enabled else "disabled")
        btn.pack(side=side, padx=padx, pady=pady, fill="x" if expand else "none", expand=expand)
        if height:
            btn.config(height=max(1, height // 6))

        def on_enter(e):
            if btn["state"] != "disabled":
                btn["bg"] = self._lighten(bg, 15)
        def on_leave(e):
            if btn["state"] != "disabled":
                btn["bg"] = bg
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        return btn

    def _style_entry(self, entry):
        entry.configure(insertbackground=self.theme["text"],
                        highlightbackground=self.theme["input_border"],
                        highlightcolor=self.theme["accent"],
                        highlightthickness=1)

    def _lighten(self, color, amount=18):
        color = color.lstrip('#')
        if len(color) != 6:
            return color
        r = min(255, int(color[0:2], 16) + amount)
        g = min(255, int(color[2:4], 16) + amount)
        b = min(255, int(color[4:6], 16) + amount)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _show_placeholder(self, text):
        self.canvas.delete("all")
        self.canvas.create_text(10, 10, text=text, fill="#555555",
                                font=("Segoe UI", 11), anchor="nw")

    # ─── Auto Connect ────────────────────────────────────────────────────
    def _auto_connect(self):
        """Automatically connect to IP Webcam on startup."""
        if self._auto_connected:
            return
        self._auto_connected = True
        
        self.set_status("🔍 Detecting phone camera...", "status")
        self._show_placeholder("Connecting to IP Webcam...\nMake sure the app is running on your phone")

        def connect():
            url = self.config.get("webcam_url", IP_WEBCAM_URL) + IP_WEBCAM_STREAM_PATH
            stream = CameraStream(stream_url=url)
            if stream.open():
                # Try to get device info
                info_text = ""
                try:
                    remote = PhoneRemote(
                        host=self.config.get("webcam_url", IP_WEBCAM_URL)
                            .replace("http://", "").split(":")[0],
                        port=int(self.config.get("webcam_url", IP_WEBCAM_URL)
                                 .split(":")[-1])
                    )
                    if remote.ping():
                        name = remote.get_phone_name()
                        bat = remote.get_battery_info()
                        info_text = f" | Battery: {bat.get('level', '?')}%" if bat else ""
                        self.root.after(0, lambda: self._update_device_info(remote, name, bat))
                except Exception:
                    pass

                self.root.after(0, lambda: self._on_connect(stream, info_text))
            else:
                self.root.after(0, self._on_connect_failed)

        threading.Thread(target=connect, daemon=True).start()

    def _update_device_info(self, remote, name, battery):
        self.status_rows["device"].config(text=str(name))
        if battery and isinstance(battery, dict):
            self.status_rows["battery"].config(text=f"{battery.get('level', '?')}%")
        self.status_rows["server"].config(text=remote.base_url)

    def _on_connect(self, stream, info_text=""):
        self.camera_stream = stream
        self.camera_stream.start()
        self.camera_open = True
        self.cam_badge.config(text=f"● {stream.source_name}", fg=self.theme["success"])
        self.set_status(f"✓ Connected to phone camera{info_text}")
        self._update_preview()

    def _on_connect_failed(self):
        self.cam_badge.config(text="● Offline", fg=self.theme["danger"])
        self._show_placeholder(
            "📱 Camera Not Found\n\n"
            "1. Install IP Webcam from Play Store\n"
            "2. Open the app → tap Start Server\n"
            "3. Phone must be on same WiFi network\n\n"
            f"Expected URL: {self.config.get('webcam_url', IP_WEBCAM_URL)}/video\n\n"
            "Click Settings to configure the IP address"
        )
        self.set_status("✗ Camera not found. Open IP Webcam on your phone.", "error")

    # ─── Preview ─────────────────────────────────────────────────────────
    def _update_preview(self):
        if not self.camera_open or not self.camera_stream:
            return
        frame = self.camera_stream.read()
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

    # ─── Scan ────────────────────────────────────────────────────────────
    def _start_scan(self):
        if not self.camera_open:
            messagebox.showwarning("Camera Required", "Camera is not connected.\nMake sure IP Webcam is running on your phone.")
            return
        if self.scanning:
            return

        self.scanning = True
        self.scan_btn.config(text="⏳ Scanning...", state="disabled")
        self.stop_btn.config(state="normal")
        self.set_status("Starting scan...")

        try:
            num_angles = int(self.angles_var.get())
            duration = int(self.dur_var.get())
        except ValueError:
            messagebox.showerror("Invalid", "Enter valid numbers for angles and duration.")
            self._reset_scan()
            return

        self._close_camera()
        self.root.update()

        def worker():
            try:
                self.root.after(0, lambda: self.set_status("Opening camera for scan...", "status"))
                self.root.after(0, lambda: self.progress_var.set(5))

                scanner = Scanner3D(use_phone=True)

                self.root.after(0, lambda: self.set_status("Position object — press 'q' when ready"))
                self.root.after(0, lambda: self.progress_var.set(15))
                scanner.run_preview()

                self.root.after(0, lambda: self.set_status("Recording — rotate object 360° slowly"))
                self.root.after(0, lambda: self.progress_var.set(30))

                sid = time.strftime("%Y%m%d_%H%M%S")
                sdir = Path(OUTPUT_DIR) / sid
                sdir.mkdir(parents=True, exist_ok=True)
                vpath = str(sdir / "scan_video")
                recorded = scanner.camera.record_video(vpath, duration_seconds=duration)
                if not recorded:
                    raise RuntimeError("Recording failed")

                self.root.after(0, lambda: self.progress_var.set(60))
                self.root.after(0, lambda: self.set_status("Extracting frames..."))
                extracted = scanner.camera.extract_frames_from_video(
                    recorded, num_frames=num_angles, output_dir=str(sdir)
                )
                self.root.after(0, lambda: self.progress_var.set(90))
                scanner.camera.release()
                self.root.after(0, lambda: self.progress_var.set(100))

                if extracted:
                    self.root.after(0, lambda: self._on_scan_done(len(extracted), sid, str(sdir)))
                else:
                    raise RuntimeError("No frames extracted")
            except Exception as e:
                self.root.after(0, lambda err=e: self._on_scan_error(err))

        threading.Thread(target=worker, daemon=True).start()

    def _on_scan_done(self, count, sid, sdir):
        self.progress_var.set(100)
        self.set_status(f"✓ Complete! {count} images captured")
        self.scan_btn.config(text="▶ START 3D SCAN", state="normal")
        self.stop_btn.config(state="disabled")
        self.scanning = False
        self.config["last_session"] = sid
        save_config(self.config)

        if messagebox.askyesno("Scan Complete",
                                f"✓ {count} images captured!\n\n"
                                f"Session: {sid}\nOpen folder?"):
            os.startfile(sdir)

    def _on_scan_error(self, error):
        self._reset_scan()
        self.set_status(f"✗ Scan failed: {error}", "error")
        messagebox.showerror("Scan Error", f"Scan failed:\n{error}")

    def _reset_scan(self):
        self.scan_btn.config(text="▶ START 3D SCAN", state="normal")
        self.stop_btn.config(state="disabled")
        self.scanning = False
        self.progress_var.set(0)

    def _stop_scan(self):
        if self.scanning:
            self.set_status("Stopping scan...")
            self.scanning = False
            self._reset_scan()

    # ─── Quick Actions ──────────────────────────────────────────────────
    def _run_reconstruction(self):
        last = self.config.get("last_session", "")
        sessions = self._get_sessions()
        if not sessions:
            messagebox.showwarning("No Sessions", "Complete a scan first.")
            return
        self._open_reconstruct_dialog(last if last in sessions else sessions[-1], sessions)

    def _get_sessions(self) -> List[str]:
        d = Path(OUTPUT_DIR)
        if not d.exists():
            return []
        return sorted([x.name for x in d.iterdir() if x.is_dir() and x.name.startswith("20")])

    def _open_reconstruct_dialog(self, default, sessions):
        dlg = Toplevel(self.root)
        dlg.title("3D Reconstruction")
        dlg.geometry("520x550")
        dlg.configure(bg=self.theme["bg"])
        dlg.transient(self.root)
        dlg.grab_set()

        mf = Frame(dlg, bg=self.theme["bg"], padx=20, pady=20)
        mf.pack(fill="both", expand=True)

        Label(mf, text="🔄 3D Reconstruction", bg=self.theme["bg"],
              fg=self.theme["text"], font=("Segoe UI", 16, "bold")).pack(pady=(0, 15))

        # Session
        Frame(mf, bg=self.theme["bg"]).pack(fill="x", pady=5)
        self.rs = StringVar(value=default)
        ttk.Combobox(mf, textvariable=self.rs, values=sessions, state="readonly", width=35).pack(fill="x", pady=2)

        # Name
        Frame(mf, bg=self.theme["bg"]).pack(fill="x", pady=5)
        self.rn = StringVar(value="")
        Entry(mf, textvariable=self.rn, bg=self.theme["input_bg"],
              fg=self.theme["text"], relief="flat", font=("Segoe UI", 10)).pack(fill="x", pady=2)

        # Quality
        Frame(mf, bg=self.theme["bg"]).pack(fill="x", pady=5)
        self.rq = StringVar(value="high")
        ttk.Combobox(mf, textvariable=self.rq, values=["draft", "medium", "high", "ultra"],
                     state="readonly", width=15).pack(fill="x", pady=2)

        info = ("Upload images to Kiri Engine API for cloud 3D reconstruction.\n"
                "• Draft: ~2 min  • Medium: ~5 min\n• High: ~15 min  • Ultra: ~30 min")
        Label(mf, text=info, bg=self.theme["bg"], fg=self.theme["text2"],
              font=("Segoe UI", 9), justify="left").pack(pady=12)

        self._btn(mf, "Start Reconstruction", lambda: self._start_recon(dlg),
                  self.theme["btn_primary"], fg="#ffffff", expand=True, side="top",
                  height=38, font_size=11)
        self._btn(mf, "Cancel", dlg.destroy, self.theme["btn_secondary"],
                  expand=True, side="top", pady=(4, 0))

    def _start_recon(self, dlg):
        sid = self.rs.get()
        name = self.rn.get().strip() or None
        qual = self.rq.get()
        dlg.destroy()
        self.set_status(f"Reconstruction ({qual})...")
        self.progress_var.set(5)
        self.root.update()

        def worker():
            try:
                result = run_reconstruction(
                    session_id=sid, output_name=name, quality=qual,
                    progress_callback=lambda p: self.root.after(0, lambda: self.progress_var.set(p))
                )
                self.root.after(0, lambda: self.progress_var.set(100))
                if result:
                    self.root.after(0, lambda: self.set_status(f"✓ Model: {result}", "success"))
                    self.root.after(0, lambda: messagebox.showinfo("Complete", f"3D model created!\n{result}"))
                else:
                    raise RuntimeError("No result")
            except Exception as e:
                self.root.after(0, lambda: self.set_status(f"✗ Failed: {e}", "error"))
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))

        threading.Thread(target=worker, daemon=True).start()

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
        dlg.geometry("520x380")
        dlg.configure(bg=self.theme["bg"])
        dlg.transient(self.root)
        dlg.grab_set()

        mf = Frame(dlg, bg=self.theme["bg"], padx=20, pady=20)
        mf.pack(fill="both", expand=True)

        Label(mf, text="⚙ Settings", bg=self.theme["bg"],
              fg=self.theme["text"], font=("Segoe UI", 16, "bold")).pack(pady=(0, 18))

        # IP Webcam URL
        Frame(mf, bg=self.theme["bg"]).pack(fill="x", pady=5)
        Label(mf, text="IP Webcam URL:", bg=self.theme["bg"],
              fg=self.theme["text2"], font=("Segoe UI", 9)).pack(anchor="w")
        url_var = StringVar(value=self.config.get("webcam_url", IP_WEBCAM_URL))
        Entry(mf, textvariable=url_var, bg=self.theme["input_bg"],
              fg=self.theme["text"], relief="flat", font=("Segoe UI", 10)).pack(fill="x", pady=2)

        # API Key
        Frame(mf, bg=self.theme["bg"]).pack(fill="x", pady=5)
        Label(mf, text="Kiri Engine API Key:", bg=self.theme["bg"],
              fg=self.theme["text2"], font=("Segoe UI", 9)).pack(anchor="w")
        api_var = StringVar(value=self.config.get("kiri_api_key", ""))
        Entry(mf, textvariable=api_var, bg=self.theme["input_bg"],
              fg=self.theme["text"], relief="flat", font=("Segoe UI", 10)).pack(fill="x", pady=2)

        # Test button
        def test():
            url = url_var.get().strip()
            self.set_status(f"Testing: {url}...")
            try:
                import urllib.request
                r = urllib.request.urlopen(url + "/", timeout=5)
                if r.status == 200:
                    messagebox.showinfo("Success", f"✓ IP Webcam reachable at {url}")
                    self.set_status("✓ Connection successful")
                r.close()
            except Exception as e:
                messagebox.showerror("Failed", f"✗ Cannot reach {url}\n\n{e}")
        
        self._btn(mf, "📡 Test Connection", test, self.theme["btn_secondary"],
                  expand=True, side="top", pady=4)

        # Save
        def save():
            self.config["webcam_url"] = url_var.get().strip()
            self.config["kiri_api_key"] = api_var.get().strip()
            save_config(self.config)
            self.conn_info.config(text=f"Server: {self.config['webcam_url']}")
            self.set_status("Settings saved")
            dlg.destroy()

        self._btn(mf, "Save Settings", save, self.theme["btn_primary"],
                  fg="#ffffff", expand=True, side="top", height=38, font_size=11)
        self._btn(mf, "Cancel", dlg.destroy, self.theme["btn_secondary"],
                  expand=True, side="top", pady=(4, 0))

    # ─── Theme ───────────────────────────────────────────────────────────
    def _toggle_theme(self):
        new = "light" if self.config.get("theme") == "dark" else "dark"
        self.config["theme"] = new
        save_config(self.config)
        self.theme = THEMES[new]
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
        self.cam_badge.config(text="● Offline", fg=self.theme["danger"])

    # ─── Misc ────────────────────────────────────────────────────────────
    def _bind_shortcuts(self):
        self.root.bind("<Escape>", lambda e: self._on_close())

    def set_status(self, text, level="info"):
        self.status_text.set(text)
        self.root.update_idletasks()

    def _on_close(self):
        if self.scanning:
            if not messagebox.askokcancel("Exit", "Scan in progress. Exit?"):
                return
        self._close_camera()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ─── Entry ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = ScannerGUI()
    app.run()