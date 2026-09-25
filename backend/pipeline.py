"""Per-frame loop on a background thread.

Stages: read frame -> general detection + tracking -> rule evaluation ->
weapon detection -> temporal filter -> draw overlays -> JPEG encode -> alert ingestion
(the annotated JPEG is kept as the alert's evidence).
"""
import threading
import time

import cv2
import numpy as np

from backend import config
from backend.alerts.manager import AlertManager
from backend.detection.weapon_detector import WeaponDetector
from backend.rules import RulesEngine, Thresholds
from backend.tracking import GeneralDetector, TrackManager
from backend.video_source import VideoSource
from backend.weapon_filter import WeaponFilter
from backend.zones import ZoneStore, point_in_polygon

WEAPON_RAW_COLOR = (0, 140, 255)
CRITICAL_COLOR = (0, 0, 220)
RESTRICTED_ZONE_COLOR = (0, 0, 255)
ZONE_COLOR = (0, 191, 255)  # amber


class Pipeline:
    def __init__(self, use_weapon_model: bool = True):
        self.detector = GeneralDetector()
        self.tracks = TrackManager(self.detector.names)
        self.weapons = WeaponDetector(enabled=use_weapon_model)
        self.weapon_filter = WeaponFilter()
        self.weapon_dets: list = []
        self.confirmed_weapons: set[str] = set()
        self.frame_index = 0
        self.source: VideoSource | None = None
        self.started_at = time.time()

        self.alerts = AlertManager()
        self.zones = ZoneStore()
        self.thresholds = Thresholds()
        self.rules = RulesEngine(self.thresholds)

        self._lock = threading.Lock()
        self._control = threading.Lock()  # serializes start/stop from API threads
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        self._jpeg = self._encode(self._idle_frame("No source - pick a camera or video"))
        self.frame_id = 0
        self.fps = 0.0
        self.latency_ms = 0.0
        self.active_tracks: list = []
        self.object_counts: dict[str, int] = {}
        self.last_error: str | None = None

    # ---- lifecycle -------------------------------------------------------

    def load(self) -> None:
        self.detector.warmup()
        self.weapons.warmup()

    def reset_state(self) -> None:
        """Forget per-source state so nothing leaks from the previous video."""
        self.detector.reset()
        self.tracks.reset()
        self.weapon_filter.reset()
        self.rules.reset()
        self.weapon_dets = []
        self.confirmed_weapons = set()
        self.frame_index = 0

    def start(self, kind: str, value, label: str | None = None) -> None:
        with self._control:
            self._stop_locked()
            src = VideoSource(kind, value, label)
            src.open()  # raise here so the API can report a bad source
            self.reset_state()
            self.source = src
            self.last_error = None
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="pipeline", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._control:
            self._stop_locked()

    def _stop_locked(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=5)
            self._thread = None
        if self.source is not None:
            self.source.close()
            self.source = None
        self.active_tracks = []
        self.object_counts = {}
        self.fps = 0.0
        self._publish(self._idle_frame("Stopped - pick a camera or video"))

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> dict:
        running = self.running
        src = self.source
        counts = dict(self.object_counts) if running else {}
        return {
            "running": running,
            "source": src.label if (running and src) else None,
            "object_count": sum(counts.values()),
            "object_counts": counts,
            "fps": round(self.fps, 1) if running else 0.0,
            "latency_ms": round(self.latency_ms, 1) if running else 0.0,
            "device": config.DEVICE,
            "alert_count": self.alerts.count(),
        }

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
            jpeg = self.process_frame(frame)
            self.latency_ms = (time.perf_counter() - t0) * 1000

            now = time.perf_counter()
            inst = 1.0 / max(now - prev, 1e-6)
            prev = now
            ema_fps = inst if ema_fps == 0 else 0.9 * ema_fps + 0.1 * inst
            self.fps = ema_fps
            self._publish_jpeg(jpeg)

    def process_frame(self, frame) -> bytes:
        """Run every stage on one frame (annotated in place). Returns the JPEG."""
        try:
            detections = self.detector.track(frame)
        except Exception as exc:  # keep the stream alive on a bad frame
            self.last_error = f"detection failed: {exc}"
            detections = []
        active = self.tracks.update(detections)
        self.active_tracks = active
        self.object_counts = self._count(active)
        h, w = frame.shape[:2]
        zones = self.zones.all()
        events = self.rules.evaluate(active, zones, (w, h))

        # Between inference frames the last raw boxes and confirmed state carry over
        if self.weapons.available and self.frame_index % config.WEAPON_EVERY_N_FRAMES == 0:
            try:
                self.weapon_dets = self.weapons.detect(frame)
            except Exception as exc:
                self.last_error = f"weapon detection failed: {exc}"
                self.weapon_dets = []
            self.confirmed_weapons = set(self.weapon_filter.update(self.weapon_dets))
        self.frame_index += 1

        self._draw_zones(frame, zones)
        self._draw(frame, active, self._in_restricted(active, zones, (w, h)))
        # Weapon boxes go on top; the counts panel moves below the threat banner when shown
        banner_h = self._draw_weapons(frame, self.weapon_dets, self.confirmed_weapons)
        self._draw_counts(frame, self.object_counts, top=banner_h)
        jpeg = self._encode(frame)

        # Weapon alerts come only from the confirmed state; the manager's cooldown
        # (per weapon class) stops a held weapon from re-firing every frame
        for name in sorted(self.confirmed_weapons):
            self.alerts.raise_alert("weapon", f"{name.capitalize()} detected in view "
                                    f"(confirmed in {config.WEAPON_MIN_HITS} of the last "
                                    f"{config.WEAPON_WINDOW} frames)", key=name, evidence=jpeg)
        for e in events:
            self.alerts.raise_alert(e.rule, e.description, zone=e.zone, track_id=e.track_id,
                                    key=e.key, evidence=jpeg)
        return jpeg

    def _draw_zones(self, frame, zones) -> None:
        h, w = frame.shape[:2]
        for z in zones:
            pts = np.array(z.pixel_polygon(w, h), dtype=np.int32)
            color = RESTRICTED_ZONE_COLOR if z.restricted else ZONE_COLOR
            cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)
            x, y = pts[0]
            label = f"{z.name} (RESTRICTED)" if z.restricted else z.name
            cv2.putText(frame, label, (int(x) + 6, max(int(y) - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, color, 2, cv2.LINE_AA)

    @staticmethod
    def _count(tracks) -> dict[str, int]:
        counts = {name: 0 for name in config.SECURITY_CLASSES}
        for t in tracks:
            counts[t.cls_name] = counts.get(t.cls_name, 0) + 1
        return {k: v for k, v in counts.items() if v}

    @staticmethod
    def _in_restricted(tracks, zones, frame_size) -> set[int]:
        w, h = frame_size
        polys = [z.pixel_polygon(w, h) for z in zones if z.restricted]
        return {t.track_id for t in tracks if t.cls_name == "person"
                and any(point_in_polygon(t.foot_point, p) for p in polys)}

    def _draw_weapons(self, frame, dets, confirmed: set) -> int:
        """Draw weapon boxes and the threat banner. Returns the banner height (0 if none)."""
        for d in dets:
            x1, y1, x2, y2 = map(int, d.bbox)
            color = CRITICAL_COLOR if d.cls_name in confirmed else WEAPON_RAW_COLOR
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{d.cls_name} {d.conf:.2f}", (x1, max(y2 + 16, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        if not confirmed:
            return 0
        # Gated on the sustained/confirmed state only, never on a raw single-frame hit
        text = "CRITICAL THREAT: " + ", ".join(sorted(confirmed)).upper()
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(frame, (0, 0), (tw + 24, th + 24), CRITICAL_COLOR, -1)
        cv2.putText(frame, text, (12, th + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        return th + 24

    @staticmethod
    def _draw_counts(frame, counts: dict, top: int = 0) -> None:
        text = " | ".join(f"{k}: {v}" for k, v in counts.items()) or "no objects"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x, y = 8, top + 8
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + tw + 12, y + th + 12), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, dst=frame)
        cv2.putText(frame, text, (x + 6, y + th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

    def _draw(self, frame, tracks, in_restricted: set[int]) -> None:
        for t in tracks:
            x1, y1, x2, y2 = map(int, t.bbox)
            if t.track_id in in_restricted:
                color = config.IN_RESTRICTED_COLOR
            else:
                color = config.SECURITY_CLASSES.get(t.cls_name, (200, 160, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{t.cls_name} #{t.track_id} {t.conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # ---- frame publishing -----------------------------------------------

    def _publish(self, frame) -> None:
        self._publish_jpeg(self._encode(frame))

    def _publish_jpeg(self, jpeg: bytes) -> None:
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
