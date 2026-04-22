from __future__ import annotations

import ctypes
import ctypes.wintypes
import platform
import threading
import time

from pynput.keyboard import Key, Controller as KeyController
from pynput.mouse import Button, Controller as MouseController
from PyQt6.QtGui import QGuiApplication


_SPECIAL_KEYS = {
    "space": Key.space,
    "enter": Key.enter,
    "return": Key.enter,
    "tab": Key.tab,
    "escape": Key.esc,
    "esc": Key.esc,
    "backspace": Key.backspace,
    "delete": Key.delete,
    "up": Key.up,
    "down": Key.down,
    "left": Key.left,
    "right": Key.right,
    "home": Key.home,
    "end": Key.end,
    "page_up": Key.page_up,
    "page_down": Key.page_down,
    "f1": Key.f1,
    "f2": Key.f2,
    "f3": Key.f3,
    "f4": Key.f4,
    "f5": Key.f5,
    "f6": Key.f6,
    "f7": Key.f7,
    "f8": Key.f8,
    "f9": Key.f9,
    "f10": Key.f10,
    "f11": Key.f11,
    "f12": Key.f12,
    "cmd": Key.cmd,
    "ctrl": Key.ctrl,
    "alt": Key.alt,
    "shift": Key.shift,
    "media_play_pause": Key.media_play_pause,
    "media_next": Key.media_next,
    "media_previous": Key.media_previous,
    "volume_up": Key.media_volume_up,
    "volume_down": Key.media_volume_down,
    "volume_mute": Key.media_volume_mute,
}

_MODIFIER_NAMES = {
    "cmd": Key.cmd,
    "ctrl": Key.ctrl,
    "alt": Key.alt,
    "shift": Key.shift,
    "option": Key.alt,
    "command": Key.cmd,
    "control": Key.ctrl,
}


def _parse_key(key_str: str):
    parts = [p.strip().lower() for p in key_str.split("+")]
    modifiers = []
    main_key = None

    for part in parts:
        if part in _MODIFIER_NAMES:
            modifiers.append(_MODIFIER_NAMES[part])
        elif part in _SPECIAL_KEYS:
            main_key = _SPECIAL_KEYS[part]
        elif len(part) == 1:
            main_key = part
        else:
            main_key = _SPECIAL_KEYS.get(part, part)

    return modifiers, main_key


class _PynputBackend:
    def __init__(self):
        self._kb = KeyController()
        self._mouse = MouseController()

    def press_key(self, key):
        self._kb.press(key)

    def release_key(self, key):
        self._kb.release(key)

    def click_left(self):
        self._mouse.click(Button.left)

    def click_right(self):
        self._mouse.click(Button.right)

    def scroll(self, amount: int):
        self._mouse.scroll(0, amount)

    def set_cursor_pos(self, x: int, y: int):
        self._mouse.position = (x, y)

    def move_cursor_relative(self, dx: int, dy: int):
        curr_x, curr_y = self._mouse.position
        self._mouse.position = (curr_x + dx, curr_y + dy)


class _WinMouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.c_void_p),
    ]


class _WinKeyboardInput(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.c_void_p),
    ]


class _WinHardwareInput(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_ulong),
        ("wParamL", ctypes.c_ushort),
        ("wParamH", ctypes.c_ushort),
    ]


class _WinInputUnion(ctypes.Union):
    _fields_ = [
        ("mi", _WinMouseInput),
        ("ki", _WinKeyboardInput),
        ("hi", _WinHardwareInput),
    ]


class _WinInput(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("union", _WinInputUnion),
    ]


class _WindowsInputBackend:
    INPUT_MOUSE = 0
    INPUT_KEYBOARD = 1

    KEYEVENTF_KEYUP = 0x0002
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    MOUSEEVENTF_RIGHTDOWN = 0x0008
    MOUSEEVENTF_RIGHTUP = 0x0010
    MOUSEEVENTF_WHEEL = 0x0800

    VK_SHIFT = 0x10
    VK_CONTROL = 0x11
    VK_MENU = 0x12
    VK_LWIN = 0x5B

    def __init__(self):
        self._user32 = ctypes.windll.user32

    def _send_mouse(self, flags: int, mouse_data: int = 0):
        input_struct = _WinInput(
            type=self.INPUT_MOUSE,
            union=_WinInputUnion(
                mi=_WinMouseInput(
                    dx=0,
                    dy=0,
                    mouseData=mouse_data,
                    dwFlags=flags,
                    time=0,
                    dwExtraInfo=None,
                )
            ),
        )
        self._user32.SendInput(1, ctypes.byref(input_struct), ctypes.sizeof(_WinInput))

    def _send_key(self, vk_code: int, key_up: bool = False):
        flags = self.KEYEVENTF_KEYUP if key_up else 0
        input_struct = _WinInput(
            type=self.INPUT_KEYBOARD,
            union=_WinInputUnion(
                ki=_WinKeyboardInput(
                    wVk=vk_code,
                    wScan=0,
                    dwFlags=flags,
                    time=0,
                    dwExtraInfo=None,
                )
            ),
        )
        self._user32.SendInput(1, ctypes.byref(input_struct), ctypes.sizeof(_WinInput))

    def _vk_code_for_key(self, key) -> int | None:
        if isinstance(key, str):
            if len(key) == 1:
                if key.isalpha():
                    return ord(key.upper())
                if key.isdigit():
                    return ord(key)
                vk = self._user32.VkKeyScanW(ord(key))
                return None if vk == -1 else vk & 0xFF
            return None

        value = getattr(key, "value", None)
        vk_code = getattr(value, "vk", None)
        if vk_code is not None:
            return int(vk_code)
        return None

    def press_key(self, key):
        vk_code = self._vk_code_for_key(key)
        if vk_code is not None:
            self._send_key(vk_code, key_up=False)

    def release_key(self, key):
        vk_code = self._vk_code_for_key(key)
        if vk_code is not None:
            self._send_key(vk_code, key_up=True)

    def click_left(self):
        self._send_mouse(self.MOUSEEVENTF_LEFTDOWN)
        self._send_mouse(self.MOUSEEVENTF_LEFTUP)

    def click_right(self):
        self._send_mouse(self.MOUSEEVENTF_RIGHTDOWN)
        self._send_mouse(self.MOUSEEVENTF_RIGHTUP)

    def scroll(self, amount: int):
        self._send_mouse(self.MOUSEEVENTF_WHEEL, amount * 120)

    def set_cursor_pos(self, x: int, y: int):
        self._user32.SetCursorPos(int(x), int(y))

    def move_cursor_relative(self, dx: int, dy: int):
        point = ctypes.wintypes.POINT()
        if self._user32.GetCursorPos(ctypes.byref(point)):
            self._user32.SetCursorPos(int(point.x + dx), int(point.y + dy))


class ActionEmitter:
    def __init__(self):
        self._backend = _WindowsInputBackend() if platform.system() == "Windows" else _PynputBackend()
        self._held_keys: dict[str, list] = {}
        self._mouse_control_active = False
        self._mouse_source: str | None = None
        self._mouse_thread: threading.Thread | None = None
        self._mouse_pos = (0.5, 0.5)
        self._mouse_lock = threading.Lock()

    def set_mouse_control(self, source: str | None):
        with self._mouse_lock:
            self._mouse_source = source
            self._mouse_control_active = source is not None
        if source and (self._mouse_thread is None or not self._mouse_thread.is_alive()):
            self._mouse_thread = threading.Thread(target=self._mouse_loop, daemon=True)
            self._mouse_thread.start()

    def update_mouse_position(self, x: float, y: float):
        with self._mouse_lock:
            self._mouse_pos = (x, y)

    def move_mouse_relative(self, dx: float, dy: float):
        try:
            screen = QGuiApplication.primaryScreen()
            if screen is None:
                return
            geometry = screen.geometry()
            self._backend.move_cursor_relative(
                int(dx * geometry.width()),
                int(dy * geometry.height()),
            )
        except Exception:
            pass

    def _mouse_loop(self):
        while True:
            with self._mouse_lock:
                active = self._mouse_control_active
                x, y = self._mouse_pos
            if not active:
                break
            try:
                screen = QGuiApplication.primaryScreen()
                if screen is not None:
                    geometry = screen.geometry()
                    self._backend.set_cursor_pos(
                        geometry.left() + int(x * geometry.width()),
                        geometry.top() + int(y * geometry.height()),
                    )
            except Exception:
                pass
            time.sleep(0.016)

    def execute(self, action_type: str, action_key: str, switch_id: str, active: bool):
        try:
            if action_type == "tap":
                self._tap(action_key)
            elif action_type == "hold":
                self._hold(action_key, switch_id, active)
            elif action_type == "mouse_left":
                self._backend.click_left()
            elif action_type == "mouse_right":
                self._backend.click_right()
            elif action_type == "scroll_up":
                self._backend.scroll(3)
            elif action_type == "scroll_down":
                self._backend.scroll(-3)
        except Exception:
            pass

    def release_all(self):
        self.set_mouse_control(None)
        for key_list in self._held_keys.values():
            for key in key_list:
                try:
                    self._backend.release_key(key)
                except Exception:
                    pass
        self._held_keys.clear()

    def _tap(self, key_str: str):
        modifiers, main_key = _parse_key(key_str)
        for key in modifiers:
            self._backend.press_key(key)
        if main_key:
            self._backend.press_key(main_key)
            self._backend.release_key(main_key)
        for key in reversed(modifiers):
            self._backend.release_key(key)

    def _hold(self, key_str: str, switch_id: str, active: bool):
        modifiers, main_key = _parse_key(key_str)
        all_keys = modifiers + ([main_key] if main_key else [])

        if active and switch_id not in self._held_keys:
            for key in all_keys:
                self._backend.press_key(key)
            self._held_keys[switch_id] = all_keys
        elif not active and switch_id in self._held_keys:
            for key in reversed(self._held_keys.pop(switch_id)):
                self._backend.release_key(key)
