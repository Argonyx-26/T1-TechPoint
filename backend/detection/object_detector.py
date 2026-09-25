"""General-purpose object/person detector + tracker.

Wraps a pretrained YOLOv8 (COCO) model via ultralytics. Works out of the box
on any footage: no custom training required. Track IDs are provided by
ultralytics' built-in ByteTrack/BoT-SORT integration so downstream rules can
reason about individual people/objects across frames.
"""
from dataclasses import dataclass
from typing import List, Tuple

from backend import config


@dataclass
class Detection:
    track_id: int          # -1 if the detector could not assign/maintain an id
    cls_name: str
    conf: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 in pixel coords


class ObjectDetector:
    """Thin wrapper around ultralytics YOLO with tracking enabled."""

    def __init__(self, model_path: str = None, device: str = None):
        self.model_path = model_path or config.GENERAL_MODEL_PATH
        self.device = device or config.DEVICE
        self._model = None  # lazy-loaded so the module imports without ultralytics

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from ultralytics import YOLO  # local import: heavy, optional at import time

        self._model = YOLO(self.model_path)
        from backend.audit import audit

        audit("MODEL_LOADED", f"general detector: {self.model_path}")

    def infer(self, frame) -> List[Detection]:
        """Run detection + tracking on a single BGR frame (numpy array)."""
        self._ensure_loaded()
        results = self._model.track(
            frame,
            persist=True,
            tracker=config.TRACKER_CONFIG,
            conf=config.DETECTOR_CONF_THRESHOLD,
            device=self.device,
            half=config.USE_HALF_PRECISION,
            verbose=False,
        )
        detections: List[Detection] = []
        if not results:
            return detections
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        names = result.names
        ids = boxes.id
        for i in range(len(boxes)):
            cls_idx = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[i].tolist()]
            track_id = int(ids[i].item()) if ids is not None else -1
            detections.append(
                Detection(
                    track_id=track_id,
                    cls_name=names.get(cls_idx, str(cls_idx)),
                    conf=conf,
                    bbox=(x1, y1, x2, y2),
                )
            )
        return detections
