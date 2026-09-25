"""Owns the single background video-processing loop and the shared state the
FastAPI endpoints read from (latest annotated frame, fps, current source).
"""
import threading
import time
from typing import Optional, Union

import cv2

from backend import config
from backend.audit import audit
from backend.pipeline import FramePipeline
from backend.video_source import VideoSource


class AppState:
    def __init__(self):
        self.pipeline = FramePipeline()
        self.video_source: Optional[VideoSource] = None
        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._frame_shape = (0, 0)
        self._fps = 0.0
        self._capture_ms = 0.0    # time spent reading a frame from the source
        self._inference_ms = 0.0  # time spent in the detection/rules pipeline
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._source_label = "none"
        self._start_time = time.time()

    def start(self, source: Union[int, str], label: str):
        self.stop()
        self.video_source = VideoSource(source)
        self._source_label = label
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self.video_source is not None:
            self.video_source.release()
            self.video_source = None

    def switch_source(self, source: Union[int, str], label: str):
        self.start(source, label)
        audit("SOURCE_CHANGED", label, actor="operator")

    def _loop(self):
        last_ts = time.time()
        min_frame_time = 1.0 / config.STREAM_MAX_FPS
        while self._running:
            t_read_start = time.time()
            ok, frame = self.video_source.read()
            capture_ms = (time.time() - t_read_start) * 1000.0
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            t_infer_start = time.time()
            try:
                annotated = self.pipeline.process(frame)
            except Exception as exc:  # keep the stream alive if one frame errors
                annotated = frame
                print(f"[pipeline] frame error: {exc}")
            inference_ms = (time.time() - t_infer_start) * 1000.0

            ok_enc, buf = cv2.imencode(
                ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), config.STREAM_JPEG_QUALITY]
            )
            if ok_enc:
                with self._lock:
                    self._latest_jpeg = buf.tobytes()
                    self._frame_shape = annotated.shape[:2]

            with self._lock:
                self._capture_ms = 0.9 * self._capture_ms + 0.1 * capture_ms
                self._inference_ms = 0.9 * self._inference_ms + 0.1 * inference_ms

            now = time.time()
            dt = now - last_ts
            last_ts = now
            if dt > 0:
                self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
            if dt < min_frame_time:
                time.sleep(min_frame_time - dt)

    def frame_size(self):
        """(width, height) of the current source's frames, (0, 0) if none yet."""
        with self._lock:
            h, w = self._frame_shape
        return w, h

    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def status(self) -> dict:
        with self._lock:
            h, w = self._frame_shape
            capture_ms = round(self._capture_ms, 1)
            inference_ms = round(self._inference_ms, 1)
        return {
            "running": self._running,
            "source": self._source_label,
            "fps": round(self._fps, 1),
            "frame_width": w,
            "frame_height": h,
            "weapon_detector_enabled": self.pipeline.weapon_detector.enabled,
            "device": config.DEVICE,
            "objects": dict(self.pipeline.last_object_counts),
            "object_counts": dict(self.pipeline.last_object_counts),
            "object_count": sum(self.pipeline.last_object_counts.values()),
            "alert_count": self.pipeline.alert_manager.active_count(),
            "zone_counts": dict(self.pipeline.rule_engine.zone_counts),
            "capture_ms": capture_ms,
            "inference_ms": inference_ms,
            "uptime_seconds": round(time.time() - self._start_time),
            "total_alerts": self.pipeline.alert_manager.total_count(),
        }


app_state = AppState()
