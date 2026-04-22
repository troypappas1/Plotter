import json
import os
import shutil
import time
from pathlib import Path

import cv2
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QSplitter,
    QPushButton, QToolBar, QLabel, QComboBox, QMessageBox,
    QInputDialog, QSystemTrayIcon, QMenu, QSizePolicy, QStyle,
)

from cv_controller.core.tracker import FaceTracker
from cv_controller.core.switches import MOUSE_SOURCES, SwitchDefinition, SwitchEngine
from cv_controller.core.emitter import ActionEmitter
from cv_controller.core.hotkey import HotkeyListener
from cv_controller.core.serial_reader import SerialThread
from cv_controller.ui.camera_widget import CameraWidget
from cv_controller.ui.switch_list import SwitchListWidget
from cv_controller.ui.switch_dialog import SwitchDialog
from cv_controller.ui.arduino_panel import ArduinoPanel
from cv_controller.ui.mouse_panel import MouseControlPanel

import platform as _platform
if _platform.system() == "Windows":
    _appdata = os.environ.get("APPDATA", str(Path.home()))
    PROFILES_DIR = Path(_appdata) / "CVController" / "profiles"
elif _platform.system() == "Darwin":
    PROFILES_DIR = Path.home() / "Library" / "Application Support" / "CVController" / "profiles"
else:
    PROFILES_DIR = Path.home() / ".config" / "CVController" / "profiles"
_RES         = Path(__file__).parent.parent.parent / "resources"
MODEL_PATH   = _RES / "face_landmarker.task"
GESTURE_PATH = _RES / "gesture_recognizer.task"

_MOUSE_CALIBRATION_WINDOW = 90
_MOUSE_CALIBRATION_PADDING = 0.02
_MOUSE_EDGE_REACH_BOOST = 1.15
_MOUSE_PAN_GAIN = 1.4
_MOUSE_PAN_DEADZONE = 0.003
_MOUSE_TOGGLE_SCORE_THRESHOLD = 0.65
_MOUSE_TOGGLE_HOLD_SECONDS = 0.15


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenCV")
        self.setMinimumSize(920, 620)
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #1a1a1a; }
            QToolBar { background: #222; border-bottom: 1px solid #333; spacing: 4px; padding: 4px 8px; }
            QPushButton {
                background: #2e2e2e; color: #ccc; border: 1px solid #444;
                border-radius: 5px; padding: 5px 12px; font-size: 12px;
            }
            QPushButton:hover  { background: #3a3a3a; color: #fff; }
            QPushButton:pressed { background: #252525; }
            QPushButton#startBtn { background: #1e4d35; border-color: #2d7a50; color: #6dffaa; font-weight: bold; }
            QPushButton#startBtn:hover { background: #255e40; }
            QPushButton#stopBtn  { background: #4d1e1e; border-color: #7a2d2d; color: #ffaaaa; font-weight: bold; }
            QPushButton#stopBtn:hover  { background: #5e2525; }
            QPushButton:checked { background: #1e3d5e; border-color: #2d6a9a; color: #88ccff; }
            QComboBox { background: #2e2e2e; color: #ccc; border: 1px solid #444; border-radius: 4px; padding: 3px 8px; }
            QScrollBar:vertical { background: #1a1a1a; width: 6px; border: none; }
            QScrollBar::handle:vertical { background: #444; border-radius: 3px; min-height: 20px; }
        """)

        self._tracker: FaceTracker | None = None
        self._engine  = SwitchEngine()
        self._emitter = ActionEmitter()
        self._hotkey  = HotkeyListener()
        self._current_profile_path: Path | None = None
        self._selected_camera_index = 0
        self._mouse_paused = False
        self._thumbs_up_started_at: float | None = None
        self._thumbs_up_triggered = False
        self._last_mouse_toggle = 0.0
        self._mouse_ranges: dict[str, dict[str, float]] = {}
        self._last_mouse_input_pos: dict[str, tuple[float, float]] = {}

        self._serial_thread: SerialThread | None = None
        self._prev_serial: dict = {k: 0 for k in ["b1", "b2", "j1", "j2", "j3", "j4"]}

        self._setup_ui()
        self._setup_tray()
        self._ensure_profiles_dir()
        self._load_default_profile()
        self._hotkey.triggered.connect(self._toggle_tracking)
        self._hotkey.start()

    def _starter_switches(self) -> list[SwitchDefinition]:
        return [
            SwitchDefinition(
                id="starter-open-palm-left-click",
                name="Open Palm -> Left Click",
                movement="gesture_Open_Palm",
                threshold=0.45,
                action_type="mouse_left",
                action_key="left_click",
                cooldown_ms=450,
                enabled=True,
            ),
            SwitchDefinition(
                id="starter-blink-right-click",
                name="Blink -> Right Click",
                movement="eyeBlinkBoth",
                threshold=0.32,
                action_type="mouse_right",
                action_key="right_click",
                cooldown_ms=700,
                enabled=True,
            ),
        ]

    def _normalize_switches_for_profile(self, path: Path, switches: list[SwitchDefinition]) -> list[SwitchDefinition]:
        if path.name.lower() == "default.json":
            return self._starter_switches()
        return switches or self._starter_switches()

    def _setup_ui(self):
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self.start_btn = QPushButton("▶  Start")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.clicked.connect(self.start_tracking)
        toolbar.addWidget(self.start_btn)

        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.clicked.connect(self.stop_tracking)
        self.stop_btn.setEnabled(False)
        toolbar.addWidget(self.stop_btn)

        toolbar.addSeparator()
        toolbar.addWidget(QLabel("  Camera: "))
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(170)
        self.camera_combo.currentIndexChanged.connect(self._on_camera_selected)
        toolbar.addWidget(self.camera_combo)

        refresh_cameras_btn = QPushButton("Refresh")
        refresh_cameras_btn.clicked.connect(self._refresh_camera_list)
        toolbar.addWidget(refresh_cameras_btn)

        toolbar.addSeparator()
        toolbar.addWidget(QLabel("  Profile: "))
        self.profile_combo = QComboBox()
        self.profile_combo.currentIndexChanged.connect(self._on_profile_selected)
        toolbar.addWidget(self.profile_combo)

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save_profile)
        toolbar.addWidget(save_btn)

        new_btn = QPushButton("New")
        new_btn.clicked.connect(self._new_profile)
        toolbar.addWidget(new_btn)

        toolbar.addSeparator()

        add_btn = QPushButton("＋ Add Switch")
        add_btn.clicked.connect(self._add_switch)
        toolbar.addWidget(add_btn)

        self._arduino_btn = QPushButton("🎮 Arduino")
        self._arduino_btn.setCheckable(True)
        self._arduino_btn.setChecked(False)
        self._arduino_btn.setToolTip("Show / hide the Arduino physical controller panel")
        self._arduino_btn.toggled.connect(self._toggle_arduino_panel)
        toolbar.addWidget(self._arduino_btn)

        self._mouse_btn = QPushButton("🖱 Mouse")
        self._mouse_btn.setCheckable(True)
        self._mouse_btn.setChecked(True)
        self._mouse_btn.setToolTip("Show / hide the Mouse Control panel")
        self._mouse_btn.toggled.connect(self._toggle_mouse_panel)
        toolbar.addWidget(self._mouse_btn)

        self.flip_btn = QPushButton("⇄ Flip")
        self.flip_btn.setCheckable(True)
        self.flip_btn.setChecked(True)
        self.flip_btn.setToolTip("Flip camera horizontally")
        self.flip_btn.toggled.connect(self._on_flip_toggled)
        toolbar.addWidget(self.flip_btn)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        hotkey_label = QLabel("  [8]  ")
        hotkey_label.setStyleSheet("color: #555; font-size: 11px;")
        hotkey_label.setToolTip("Global hotkey: press 8 from any app to toggle tracking")
        toolbar.addWidget(hotkey_label)

        about_btn = QPushButton("ℹ")
        about_btn.setFixedWidth(32)
        about_btn.clicked.connect(self._show_about)
        toolbar.addWidget(about_btn)

        self.status_label = QLabel("● Stopped")
        self.status_label.setStyleSheet("color: #555; font-size: 11px; padding: 0 8px;")
        toolbar.addWidget(self.status_label)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(self._splitter)

        self.camera_widget = CameraWidget()
        self._splitter.addWidget(self.camera_widget)

        self.switch_list = SwitchListWidget()
        self.switch_list.edit_requested.connect(self._edit_switch)
        self.switch_list.delete_requested.connect(self._delete_switch)
        self._splitter.addWidget(self.switch_list)

        self.arduino_panel = ArduinoPanel()
        self.arduino_panel.connect_requested.connect(self._connect_arduino)
        self.arduino_panel.disconnect_requested.connect(self._disconnect_arduino)
        self.arduino_panel.mappings_changed.connect(self._on_arduino_mappings_changed)
        self._splitter.addWidget(self.arduino_panel)

        self.mouse_panel = MouseControlPanel()
        self.mouse_panel.source_changed.connect(self._on_mouse_source_changed)
        self.mouse_panel.sensitivity_changed.connect(self._on_mouse_sensitivity_changed)
        self._splitter.addWidget(self.mouse_panel)

        self._splitter.setSizes([580, 220, 220, 200])
        self._splitter.setHandleWidth(1)
        self._toggle_arduino_panel(False)
        self._refresh_camera_list()

    def _setup_tray(self):
        icon = self.windowIcon()
        if icon.isNull():
            icon = self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
            self.setWindowIcon(icon)
        self.tray = QSystemTrayIcon(icon, self)

        menu = QMenu()
        self._tray_toggle = QAction("▶ Start Tracking", self)
        self._tray_toggle.triggered.connect(self._toggle_tracking)
        menu.addAction(self._tray_toggle)

        show_act = QAction("Show Window", self)
        show_act.triggered.connect(lambda: (self.show(), self.raise_(), self.activateWindow()))
        menu.addAction(show_act)
        menu.addSeparator()

        quit_act = QAction("Quit OpenCV", self)
        quit_act.triggered.connect(self._quit)
        menu.addAction(quit_act)

        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._on_tray_activated)
        self.tray.setToolTip("OpenCV — Stopped")
        self.tray.show()

    # ── Tracking ─────────────────────────────────────────────────────────────

    def _toggle_tracking(self):
        if self._tracker and self._tracker.isRunning():
            self.stop_tracking()
        else:
            self.start_tracking()

    def start_tracking(self):
        if self._tracker and self._tracker.isRunning():
            return
        if not MODEL_PATH.exists():
            import platform as _plt
            setup_cmd = "setup.bat" if _plt.system() == "Windows" else "setup.sh"
            QMessageBox.critical(self, "Model Missing",
                f"Face model not found:\n{MODEL_PATH}\n\nRun {setup_cmd} first.")
            return

        gesture_path = str(GESTURE_PATH) if GESTURE_PATH.exists() else None
        self._tracker = FaceTracker(
            model_path=str(MODEL_PATH),
            gesture_model_path=gesture_path,
            camera_index=self._selected_camera_index,
        )
        self._tracker.frame_ready.connect(self.camera_widget.update_frame)
        self._tracker.face_data.connect(self.camera_widget.update_face_data)
        self._tracker.face_data.connect(self._on_face_data)
        self._tracker.tracking_error.connect(self._on_tracking_error)
        self._tracker.start()
        self._mouse_paused = False
        self._thumbs_up_started_at = None
        self._thumbs_up_triggered = False
        self._reset_mouse_calibration()
        self._apply_mouse_settings()
        self.mouse_panel.set_mouse_enabled(True)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._set_status("● Tracking", "#3dff7a")
        self._tray_toggle.setText("■ Stop Tracking")
        self.tray.setToolTip("OpenCV — Tracking active")

    def stop_tracking(self):
        if self._tracker:
            self._emitter.release_all()
            self._tracker.stop()
            self._tracker = None
        self._mouse_paused = False
        self._thumbs_up_started_at = None
        self._thumbs_up_triggered = False
        self._reset_mouse_calibration()
        self.camera_widget.set_mouse_target(None, clear_trail=True)
        self.mouse_panel.set_active(False)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._set_status("● Stopped", "#555")
        self._tray_toggle.setText("▶ Start Tracking")
        self.tray.setToolTip("OpenCV — Stopped")

    def _set_status(self, text: str, color: str):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color}; font-size: 11px; padding: 0 8px;")

    @pyqtSlot(dict)
    def _on_face_data(self, data: dict):
        self._handle_mouse_toggle_gesture(data)
        self._update_mouse_from_face_data(data)
        events = self._engine.evaluate(data)
        self.switch_list.update_values({e.switch.id: e.current_value for e in events})
        for event in events:
            sw = event.switch
            if event.triggered:
                self.switch_list.flash(sw.id)
                if sw.action_type in ("tap", "mouse_left", "mouse_right", "scroll_up", "scroll_down"):
                    self._emitter.execute(sw.action_type, sw.action_key, sw.id, True)
            if sw.action_type == "hold":
                self._emitter.execute("hold", sw.action_key, sw.id, event.active)

    def _on_flip_toggled(self, checked: bool):
        if self._tracker:
            self._tracker.flip_horizontal = checked

    @pyqtSlot(str)
    def _on_tracking_error(self, msg: str):
        self.stop_tracking()
        QMessageBox.warning(self, "Tracking Error", msg)

    # ── Arduino panel ─────────────────────────────────────────────────────────

    def _toggle_arduino_panel(self, visible: bool):
        sizes = self._splitter.sizes()
        if visible:
            total = sum(sizes)
            self._splitter.setSizes([total - sizes[1] - 220 - sizes[3], sizes[1], 220, sizes[3]])
        else:
            self._splitter.setSizes([sizes[0] + sizes[2], sizes[1], 0, sizes[3]])

    def _toggle_mouse_panel(self, visible: bool):
        sizes = self._splitter.sizes()
        if visible:
            total = sum(sizes)
            self._splitter.setSizes([total - sizes[1] - sizes[2] - 200, sizes[1], sizes[2], 200])
        else:
            self._splitter.setSizes([sizes[0] + sizes[3], sizes[1], sizes[2], 0])

    def _on_mouse_source_changed(self, source: str):
        self._mouse_paused = False
        self._reset_mouse_calibration()
        self._apply_mouse_settings()
        if source == "none":
            self.camera_widget.set_mouse_target(None, clear_trail=True)
        self.mouse_panel.set_mouse_enabled(
            bool(self._tracker and self._tracker.isRunning() and not self._mouse_paused and source != "none")
        )
        self._save_profile_if_ready()

    def _refresh_camera_list(self):
        available = self._detect_cameras()
        current = self._selected_camera_index

        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        for index, label in available:
            self.camera_combo.addItem(label, index)

        selected_idx = self.camera_combo.findData(current)
        if selected_idx < 0 and self.camera_combo.count():
            selected_idx = 0
            self._selected_camera_index = self.camera_combo.itemData(0)
        if selected_idx >= 0:
            self.camera_combo.setCurrentIndex(selected_idx)
        self.camera_combo.blockSignals(False)

    def _detect_cameras(self) -> list[tuple[int, str]]:
        cameras: list[tuple[int, str]] = []
        for index in range(8):
            cap = cv2.VideoCapture(index)
            try:
                if cap.isOpened():
                    ok, _ = cap.read()
                    if ok:
                        cameras.append((index, f"Camera {index}"))
            finally:
                cap.release()

        if not cameras:
            cameras.append((0, "Camera 0"))
        return cameras

    def _on_camera_selected(self, index: int):
        camera_index = self.camera_combo.itemData(index)
        if camera_index is None:
            return
        self._selected_camera_index = int(camera_index)
        self._save_profile_if_ready()
        if self._tracker and self._tracker.isRunning():
            self.stop_tracking()
            self.start_tracking()

    def _apply_mouse_settings(self):
        source = self.mouse_panel.get_source()
        if source == "none" or self._mouse_paused:
            self._emitter.set_mouse_control(None)
        elif source in ("hand", "index_tip"):
            self._emitter.set_mouse_control(None)
        else:
            self._emitter.set_mouse_control(source)

    def _on_mouse_sensitivity_changed(self, sensitivity: float):
        self._save_profile_if_ready()

    def _update_mouse_from_face_data(self, data: dict):
        source = self.mouse_panel.get_source()
        if source == "none" or self._mouse_paused:
            self._clear_mouse_anchor(source)
            self.camera_widget.set_mouse_target(None)
            return
        sensitivity = self.mouse_panel.get_sensitivity()

        pos = None
        if source in ("nose", "forehead", "chin", "head"):
            if not data.get("face_detected"):
                return
            body_points = data.get("body_points", {})
            pos = body_points.get(source)
        elif source == "hand":
            hp = data.get("hand_position", {})
            if hp:
                pos = (hp.get("x", 0.5), hp.get("y", 0.5))
        elif source == "index_tip":
            it = data.get("index_tip", {})
            if it:
                pos = (it.get("x", 0.5), it.get("y", 0.5))

        if pos:
            if source in ("hand", "index_tip"):
                preview = self._pan_mouse_from_delta(source, pos, sensitivity)
                if preview:
                    self.camera_widget.set_mouse_target(preview, f"Cursor - {MOUSE_SOURCES.get(source, source)}")
                else:
                    self.camera_widget.set_mouse_target(None)
            else:
                x, y = self._expand_mouse_range(source, pos, sensitivity)
                self._emitter.update_mouse_position(x, y)
                self.camera_widget.set_mouse_target((x, y), f"Cursor - {MOUSE_SOURCES.get(source, source)}")
        else:
            self._clear_mouse_anchor(source)
            self.camera_widget.set_mouse_target(None)

    def _handle_mouse_toggle_gesture(self, data: dict):
        gestures = data.get("gestures", {})
        thumbs_up = gestures.get("Thumb_Up", 0.0) >= _MOUSE_TOGGLE_SCORE_THRESHOLD
        now = time.time()
        if thumbs_up:
            if self._thumbs_up_started_at is None:
                self._thumbs_up_started_at = now
        else:
            self._thumbs_up_started_at = None
            self._thumbs_up_triggered = False

        if (
            thumbs_up
            and not self._thumbs_up_triggered
            and self.mouse_panel.get_source() != "none"
            and self._thumbs_up_started_at is not None
            and now - self._thumbs_up_started_at >= _MOUSE_TOGGLE_HOLD_SECONDS
            and now - self._last_mouse_toggle >= 1.0
        ):
            self._mouse_paused = not self._mouse_paused
            self._last_mouse_toggle = now
            self._thumbs_up_triggered = True
            self._apply_mouse_settings()
            self.mouse_panel.set_mouse_enabled(not self._mouse_paused)
            if self._mouse_paused:
                self.camera_widget.set_mouse_target(None, clear_trail=True)

    def _expand_mouse_range(self, source: str, pos: tuple[float, float], sensitivity: float) -> tuple[float, float]:
        x = self._normalize_mouse_axis(source, "x", pos[0], sensitivity)
        y = self._normalize_mouse_axis(source, "y", pos[1], sensitivity)
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        return x, y

    def _normalize_mouse_axis(self, source: str, axis: str, value: float, sensitivity: float) -> float:
        ranges = self._mouse_ranges.setdefault(source, {})
        min_key = f"{axis}_min"
        max_key = f"{axis}_max"

        if min_key not in ranges:
            ranges[min_key] = value
            ranges[max_key] = value
        else:
            ranges[min_key] = min(ranges[min_key], value)
            ranges[max_key] = max(ranges[max_key], value)

        observed_min = ranges[min_key]
        observed_max = ranges[max_key]
        span = max(0.04, observed_max - observed_min)

        # Map the observed range directly to the screen so the furthest seen
        # left/right/up/down positions can actually reach the edges.
        normalized = (value - observed_min) / span
        return 0.5 + (normalized - 0.5) * max(_MOUSE_EDGE_REACH_BOOST, sensitivity)

    def _reset_mouse_calibration(self):
        self._mouse_ranges.clear()
        self._last_mouse_input_pos.clear()

    def _clear_mouse_anchor(self, source: str):
        if source in self._last_mouse_input_pos:
            self._last_mouse_input_pos.pop(source, None)

    def _pan_mouse_from_delta(self, source: str, pos: tuple[float, float], sensitivity: float) -> tuple[float, float] | None:
        previous = self._last_mouse_input_pos.get(source)
        self._last_mouse_input_pos[source] = pos
        if previous is None:
            return None

        dx = (pos[0] - previous[0]) * max(_MOUSE_PAN_GAIN, sensitivity)
        dy = (pos[1] - previous[1]) * max(_MOUSE_PAN_GAIN, sensitivity)
        if abs(dx) < _MOUSE_PAN_DEADZONE:
            dx = 0.0
        if abs(dy) < _MOUSE_PAN_DEADZONE:
            dy = 0.0
        if dx == 0.0 and dy == 0.0:
            return pos
        self._emitter.move_mouse_relative(dx, dy)
        return pos

    def _connect_arduino(self, port: str, baud: int):
        if self._serial_thread and self._serial_thread.isRunning():
            self._serial_thread.stop()
        self._serial_thread = SerialThread(port, baud)
        self._serial_thread.data_received.connect(self._on_serial_data)
        self._serial_thread.connection_changed.connect(self._on_serial_connected)
        self._serial_thread.error_occurred.connect(self._on_serial_error)
        self._serial_thread.start()

    def _disconnect_arduino(self):
        if self._serial_thread:
            self._serial_thread.stop()
            self._serial_thread = None
        self.arduino_panel.set_connected(False, "Disconnected")

    @pyqtSlot(dict)
    def _on_serial_data(self, data: dict):
        # Update controller visualizer
        self.arduino_panel.update_states(data)

        # Detect 0 → 1 (press) and 1 → 0 (release) transitions
        mappings = self.arduino_panel.get_mappings()
        for key in ["b1", "b2", "j1", "j2", "j3", "j4"]:
            curr = int(data.get(key, 0))
            prev = self._prev_serial.get(key, 0)

            if curr and not prev:
                # Rising edge — fire / start hold
                self.arduino_panel.flash_input(key)
                m = mappings.get(key, {})
                action_type = m.get("action_type", "tap")
                action_key  = m.get("action_key", "")
                self._emitter.execute(action_type, action_key, f"arduino_{key}", True)

            elif not curr and prev:
                # Falling edge — release hold (no-op for tap / mouse / scroll)
                m = mappings.get(key, {})
                if m.get("action_type") == "hold":
                    self._emitter.execute("hold", m.get("action_key", ""),
                                          f"arduino_{key}", False)

        self._prev_serial = {k: int(data.get(k, 0))
                             for k in ["b1", "b2", "j1", "j2", "j3", "j4"]}

    @pyqtSlot(bool, str)
    def _on_serial_connected(self, connected: bool, message: str):
        self.arduino_panel.set_connected(connected, message)
        if connected and not self._arduino_btn.isChecked():
            # Auto-show the panel when a device connects
            self._arduino_btn.setChecked(True)

    @pyqtSlot(str)
    def _on_serial_error(self, error: str):
        self.arduino_panel.set_connected(False, f"Error: {error}")

    def _on_arduino_mappings_changed(self):
        # Auto-save to current profile when user edits a mapping
        if self._current_profile_path:
            self._save_profile()

    # ── Switch management ─────────────────────────────────────────────────────

    def _add_switch(self):
        dlg = SwitchDialog(self)
        if dlg.exec():
            sw = dlg.get_switch()
            self._engine.switches.append(sw)
            self.switch_list.add_switch(sw)
            self._save_profile_if_ready()

    def _edit_switch(self, switch_id: str):
        sw = next((s for s in self._engine.switches if s.id == switch_id), None)
        if not sw:
            return
        dlg = SwitchDialog(self, sw)
        if dlg.exec():
            dlg.get_switch(existing=sw)
            self.switch_list.refresh_switch(sw)
            self._save_profile_if_ready()

    def _delete_switch(self, switch_id: str):
        self._engine.switches = [s for s in self._engine.switches if s.id != switch_id]
        self.switch_list.remove_switch(switch_id)
        self._save_profile_if_ready()

    # ── Profiles ─────────────────────────────────────────────────────────────

    def _ensure_profiles_dir(self):
        PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        default_src = Path(__file__).parent.parent.parent / "profiles" / "default.json"
        default_dst = PROFILES_DIR / "default.json"
        if not default_dst.exists() and default_src.exists():
            shutil.copy(default_src, default_dst)
        self._refresh_profile_list()

    def _refresh_profile_list(self):
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        for f in sorted(PROFILES_DIR.glob("*.json")):
            self.profile_combo.addItem(f.stem, str(f))
        self.profile_combo.blockSignals(False)

    def _load_default_profile(self):
        default = PROFILES_DIR / "default.json"
        if not default.exists():
            self._current_profile_path = default
            self._engine.switches = self._starter_switches()
            self.switch_list.clear()
            for sw in self._engine.switches:
                self.switch_list.add_switch(sw)
            self.mouse_panel.set_source("none")
            self.mouse_panel.set_sensitivity(2.5)
            self._save_profile()
        self._load_profile(default)

    def _load_profile(self, path: Path):
        try:
            with open(path) as f:
                data = json.load(f)
            self._current_profile_path = path
            raw_switches = [SwitchDefinition.from_dict(s) for s in data.get("switches", [])]
            self._engine.switches = self._normalize_switches_for_profile(path, raw_switches)
            self.switch_list.clear()
            for sw in self._engine.switches:
                self.switch_list.add_switch(sw)
            if "arduino" in data:
                self.arduino_panel.set_mappings(data["arduino"])
            mouse_settings = data.get("mouse", {})
            self._selected_camera_index = int(data.get("camera_index", 0))
            self._mouse_paused = False
            self._reset_mouse_calibration()
            self.camera_combo.blockSignals(True)
            self._refresh_camera_list()
            self.camera_combo.blockSignals(False)
            self.mouse_panel.source_combo.blockSignals(True)
            self.mouse_panel.sens_slider.blockSignals(True)
            self.mouse_panel.set_source(mouse_settings.get("source", "none"))
            self.mouse_panel.set_sensitivity(mouse_settings.get("sensitivity", 1.0))
            self.mouse_panel.source_combo.blockSignals(False)
            self.mouse_panel.sens_slider.blockSignals(False)
            self.mouse_panel.set_active(bool(self._tracker and self._tracker.isRunning()))
            if self._tracker and self._tracker.isRunning():
                self._apply_mouse_settings()
            if path.name.lower() == "default.json":
                self._save_profile()
            for i in range(self.profile_combo.count()):
                if self.profile_combo.itemData(i) == str(path):
                    self.profile_combo.blockSignals(True)
                    self.profile_combo.setCurrentIndex(i)
                    self.profile_combo.blockSignals(False)
                    break
        except Exception as e:
            QMessageBox.warning(self, "Load Error", str(e))

    def _save_profile(self):
        path = self._current_profile_path
        if not path:
            name, ok = QInputDialog.getText(self, "Save Profile", "Profile name:")
            if not ok or not name.strip():
                return
            path = PROFILES_DIR / f"{name.strip()}.json"
            self._current_profile_path = path
        with open(path, "w") as f:
            json.dump({
                "name":    path.stem,
                "version": 1,
                "camera_index": self._selected_camera_index,
                "switches": [s.to_dict() for s in self._engine.switches],
                "arduino":  self.arduino_panel.get_mappings(),
                "mouse": {
                    "source": self.mouse_panel.get_source(),
                    "sensitivity": self.mouse_panel.get_sensitivity(),
                },
            }, f, indent=2)
        self._refresh_profile_list()

    def _save_profile_if_ready(self):
        if self._current_profile_path:
            self._save_profile()

    def _new_profile(self):
        name, ok = QInputDialog.getText(self, "New Profile", "Profile name:")
        if not ok or not name.strip():
            return
        self._current_profile_path = PROFILES_DIR / f"{name.strip()}.json"
        self._engine.switches = self._starter_switches()
        self.switch_list.clear()
        for sw in self._engine.switches:
            self.switch_list.add_switch(sw)
        self.mouse_panel.set_source("none")
        self.mouse_panel.set_sensitivity(2.5)
        self._save_profile()
        self._refresh_profile_list()

    def _on_profile_selected(self, index: int):
        path_str = self.profile_combo.itemData(index)
        if path_str:
            self._load_profile(Path(path_str))

    # ── About / tray ─────────────────────────────────────────────────────────

    def _on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            if self.isVisible():
                self.hide()
            else:
                self.show()
                self.raise_()
                self.activateWindow()

    def _show_about(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("About OpenCV")
        msg.setIconPixmap(self.windowIcon().pixmap(64, 64))
        msg.setText(
            "<h2 style='color:#3dff7a;'>OpenCV</h2>"
            "<p><b>Adaptive Vision Controller</b></p>"
            "<p>Made by <b>Jacob Majors</b><br>"
            "For <b>Ramsey Mussalum's</b> Design for Social Good class<br>"
            "at <b>Sonoma Academy</b> — March 2026</p>"
            "<hr>"
            "<p style='color:#888; font-size:11px;'>"
            "Global hotkey: <b>8</b> — toggle tracking from any app<br>"
            "Supported: face, head pose, hand gestures, hand position<br>"
            "Powered by MediaPipe · PyQt6 · pynput</p>"
        )
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        event.ignore()
        self.hide()
        self.tray.showMessage(
            "OpenCV running in menu bar",
            "Press 8 to toggle tracking, or click the menu bar icon.",
            QSystemTrayIcon.MessageIcon.Information, 2000,
        )

    def _quit(self):
        self._hotkey.stop()
        self.stop_tracking()
        self._disconnect_arduino()
        from PyQt6.QtWidgets import QApplication
        QApplication.quit()
