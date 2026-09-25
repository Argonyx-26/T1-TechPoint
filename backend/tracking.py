"""General detector (pretrained yolov8n, COCO) with ByteTrack, plus a track manager
that keeps per-track history for the rules engine (loitering, zone intrusion)."""
import time
from collections import deque
from dataclasses import dataclass, field

from ultralytics import YOLO

from backend import config


@dataclass
class Track:
    track_id: int
    cls_id: int
    cls_name: str
    conf: float
    bbox: tuple  # x1, y1, x2, y2 in pixels
    first_seen: float
    last_seen: float
    history: deque = field(default_factory=lambda: deque(maxlen=config.TRACK_HISTORY_LEN))

    @property
    def foot_point(self) -> tuple:
        # Bottom-center of the box: where the object touches the ground, best for zone tests
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)

    @property
    def dwell_seconds(self) -> float:
        return self.last_seen - self.first_seen


class GeneralDetector:
    def __init__(self):
        weights = config.GENERAL_MODEL_PATH
        # Loaded once per process; overwriting the file on disk needs a server restart
        self.model = YOLO(str(weights) if weights.is_file() else "yolov8n.pt")
        self.names = self.model.names
        by_name = {n: i for i, n in self.names.items()}
        missing = set(config.SECURITY_CLASSES) - set(by_name)
        if missing:
            raise ValueError(f"not COCO classes: {', '.join(sorted(missing))}")
        self.classes = sorted(by_name[n] for n in config.SECURITY_CLASSES)

    def warmup(self) -> None:
        import numpy as np

        self.model.predict(np.zeros((640, 640, 3), dtype="uint8"), device=config.DEVICE, verbose=False)

    def track(self, frame) -> list:
        """Run detection + tracking on one frame. Returns [(track_id, cls_id, conf, bbox)]."""
        result = self.model.track(
            frame,
            persist=True,
            tracker=config.TRACKER_CFG,
            conf=config.DISPLAY_CONF,
            classes=self.classes,
            imgsz=config.INFER_IMGSZ,
            device=config.DEVICE,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return []
        ids = boxes.id.int().tolist()
        cls = boxes.cls.int().tolist()
        conf = boxes.conf.tolist()
        xyxy = boxes.xyxy.tolist()
        return [(i, c, p, tuple(b)) for i, c, p, b in zip(ids, cls, conf, xyxy)]

    def reset(self) -> None:
        # Drop tracker state so IDs from a previous source do not leak into a new one
        predictor = getattr(self.model, "predictor", None)
        if predictor is not None and getattr(predictor, "trackers", None):
            for t in predictor.trackers:
                t.reset()


class TrackManager:
    def __init__(self, names: dict):
        self.names = names
        self.tracks: dict[int, Track] = {}

    def update(self, detections: list, now: float | None = None) -> list[Track]:
        now = now or time.time()
        active = []
        for track_id, cls_id, conf, bbox in detections:
            t = self.tracks.get(track_id)
            if t is None:
                t = Track(track_id, cls_id, self.names[cls_id], conf, bbox, now, now)
                self.tracks[track_id] = t
            else:
                t.cls_id, t.cls_name, t.conf, t.bbox, t.last_seen = (
                    cls_id, self.names[cls_id], conf, bbox, now)
            t.history.append((now, t.foot_point))
            active.append(t)
        stale = [k for k, t in self.tracks.items() if now - t.last_seen > config.TRACK_STALE_SECONDS]
        for k in stale:
            del self.tracks[k]
        return active

    def reset(self) -> None:
        self.tracks.clear()
