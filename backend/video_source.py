"""Frame sources: webcam index, video file (looped), or stream URL."""
import time
from pathlib import Path

import cv2


class VideoSource:
    def __init__(self, kind: str, value):
        self.kind = kind  # "webcam" | "file" | "url"
        self.value = value
        self.cap = None
        self._frame_interval = 0.0
        self._last_read = 0.0

    @property
    def label(self) -> str:
        if self.kind == "webcam":
            return f"webcam:{self.value}"
        if self.kind == "file":
            return f"file:{Path(self.value).name}"
        return f"url:{self.value}"

    def open(self) -> None:
        if self.kind == "webcam":
            # CAP_DSHOW opens much faster than the default backend on Windows
            self.cap = cv2.VideoCapture(int(self.value), cv2.CAP_DSHOW)
        else:
            if self.kind == "file" and not Path(self.value).is_file():
                raise FileNotFoundError(self.value)
            self.cap = cv2.VideoCapture(str(self.value))
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open source {self.label}")
        if self.kind == "file":
            # Pace files at their native FPS so the demo does not play in fast-forward
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
            self._frame_interval = 1.0 / fps

    def read(self):
        """Return the next BGR frame, or None when a live source drops."""
        if self._frame_interval:
            wait = self._frame_interval - (time.perf_counter() - self._last_read)
            if wait > 0:
                time.sleep(wait)
        ok, frame = self.cap.read()
        if not ok and self.kind == "file":
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        self._last_read = time.perf_counter()
        return frame if ok else None

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
