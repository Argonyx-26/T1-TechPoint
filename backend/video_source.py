"""Wraps cv2.VideoCapture with a swappable source (webcam / uploaded file /
bundled sample clip) so the dashboard's upload button can hot-swap the feed
without restarting the server.
"""
import threading
from pathlib import Path
from typing import Optional, Union

import cv2


class VideoSource:
    def __init__(self, default: Union[int, str] = 0):
        self._lock = threading.Lock()
        self._cap: Optional[cv2.VideoCapture] = None
        self._current: Union[int, str] = default
        self._open(default)

    def _open(self, source: Union[int, str]):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
        with self._lock:
            old_cap = self._cap
            self._cap = cap
            self._current = source
        if old_cap is not None:
            old_cap.release()

    def switch(self, source: Union[int, str, Path]):
        self._open(str(source) if isinstance(source, Path) else source)

    def read(self):
        with self._lock:
            cap = self._cap
            current = self._current
        if cap is None:
            return False, None
        ok, frame = cap.read()
        if not ok and isinstance(current, str):
            # Loop file-based sources; a webcam that fails just stays failed.
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        return ok, frame

    @property
    def current(self):
        return self._current

    def release(self):
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
