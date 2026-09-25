"""Frame sources: webcam index, video file (looped), or stream URL."""
import time
from pathlib import Path

import cv2

URL_TIMEOUT_MS = 5000


class VideoSource:
    def __init__(self, kind: str, value, label: str | None = None):
        self.kind = kind  # "webcam" | "file" | "url"
        self.value = value
        self._label = label
        self.cap = None
        self._frame_interval = 0.0
        self._last_read = 0.0

    @property
    def label(self) -> str:
        """Readable name for the dashboard, e.g. "Webcam 0" or "sample: vtest.avi"."""
        if self._label:
            return self._label
        if self.kind == "webcam":
            return f"Webcam {self.value}"
        if self.kind == "file":
            return f"file: {Path(self.value).name}"
        return f"camera: {self.value}"

    def open(self) -> None:
        if self.kind == "webcam":
            # CAP_DSHOW opens much faster than the default backend on Windows
            self.cap = cv2.VideoCapture(int(self.value), cv2.CAP_DSHOW)
        elif self.kind == "url":
            # Without timeouts an unreachable camera blocks the request for ~30s
            self.cap = cv2.VideoCapture(str(self.value), cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, URL_TIMEOUT_MS,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, URL_TIMEOUT_MS,
            ])
        else:
            if not Path(self.value).is_file():
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
