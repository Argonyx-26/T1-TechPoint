"""Per-frame loop on a background thread.

Current stages: read frame -> general detection + tracking -> draw overlays -> JPEG encode.
Still to add: rule evaluation, weapon detection, temporal filter, alert ingestion.
"""
import threading
import time

import cv2
import numpy as np

from backend import config
from backend.tracking import GeneralDetector, TrackManager
from backend.video_source import VideoSource

PERSON_COLOR = (0, 200, 0)
OBJECT_COLOR = (200, 160, 0)


class Pipeline:
    def __init__(self):
        self.detector = GeneralDetector()
        self.tracks = TrackManager(self.detector.names)
        self.source: VideoSource | None = None
        self.started_at = time.time()

        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        self._jpeg = self._encode(self._idle_frame("No source - pick a camera or video"))
        self.frame_id = 0
        self.fps = 0.0
        self.latency_ms = 0.0
        self.active_tracks: list = []
        self.last_error: str | None = None

    # ---- lifecycle -------------------------------------------------------

    def load(self) -> None:
        self.detector.warmup()

    def start(self, kind: str, value) -> None:
        self.stop()
        src = VideoSource(kind, value)
        src.open()  # raise here so the API can report a bad source
        self.detector.reset()
        self.tracks.reset()
        self.source = src
        self.last_error = None
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="pipeline", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=5)
            self._thread = None
        if self.source is not None:
            self.source.close()
            self.source = None
        self.active_tracks = []
        self.fps = 0.0
        self._publish(self._idle_frame("Stopped - pick a camera or video"))

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ---- frame loop ------------------------------------------------------

    def _run(self) -> None:
        ema_fps = 0.0
        prev = time.perf_counter()
        while not self._stop.is_set():
            frame = self.source.read()
            if frame is None:
                self.last_error = "source returned no frame"
                self._publish(self._idle_frame("Source lost"))
                time.sleep(0.5)
                continue

            t0 = time.perf_counter()
            try:
                detections = self.detector.track(frame)
            except Exception as exc:  # keep the stream alive on a bad frame
                self.last_error = f"detection failed: {exc}"
                detections = []
            active = self.tracks.update(detections)
            self._draw(frame, active)
            self.latency_ms = (time.perf_counter() - t0) * 1000

            now = time.perf_counter()
            inst = 1.0 / max(now - prev, 1e-6)
            prev = now
            ema_fps = inst if ema_fps == 0 else 0.9 * ema_fps + 0.1 * inst
            self.fps = ema_fps
            self.active_tracks = active
            self._publish(frame)

    def _draw(self, frame, tracks) -> None:
        for t in tracks:
            x1, y1, x2, y2 = map(int, t.bbox)
            color = PERSON_COLOR if t.cls_name == "person" else OBJECT_COLOR
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{t.cls_name} #{t.track_id} {t.conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # ---- frame publishing -----------------------------------------------

    def _publish(self, frame) -> None:
        jpeg = self._encode(frame)
        with self._lock:
            self._jpeg = jpeg
            self.frame_id += 1

    def latest_jpeg(self) -> tuple[int, bytes]:
        with self._lock:
            return self.frame_id, self._jpeg

    @staticmethod
    def _encode(frame) -> bytes:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
        return buf.tobytes() if ok else b""

    @staticmethod
    def _idle_frame(text: str):
        w, h = config.IDLE_FRAME_SIZE
        img = np.full((h, w, 3), 24, dtype=np.uint8)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(img, text, ((w - tw) // 2, (h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (180, 180, 180), 2, cv2.LINE_AA)
        return img
