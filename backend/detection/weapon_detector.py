"""Weapon detector: a separate pretrained YOLO model (gun/knife classes).

Runs alongside the general detector rather than replacing it. Any detection
from this model is treated as critical severity immediately -- no
corroboration from the behavior rules is required, per the design brief.

The system degrades gracefully if no weapon model weights are present: the
detector reports itself disabled and simply contributes no detections, so
the rest of the pipeline (general detection, behavior rules) keeps working
for demos where a weapon model hasn't been supplied yet.
"""
from dataclasses import dataclass
from typing import List, Tuple

from backend import config


@dataclass
class WeaponDetection:
    cls_name: str
    conf: float
    bbox: Tuple[float, float, float, float]


def is_implausibly_large(bbox, frame_area: float) -> bool:
    """True for a weapon box covering more than WEAPON_MAX_BOX_FRACTION of the
    frame -- the model reacting to the whole scene, not an object in it."""
    x1, y1, x2, y2 = bbox
    return frame_area > 0 and (x2 - x1) * (y2 - y1) / frame_area > config.WEAPON_MAX_BOX_FRACTION


class WeaponDetector:
    def __init__(self, model_path=None, device: str = None, conf_threshold: float = None):
        self.model_path = model_path or config.WEAPON_MODEL_PATH
        self.device = device or config.DEVICE
        # Low floor here on purpose -- this is what the model reports at all,
        # not what raises an alert. process() in pipeline.py separately
        # requires WEAPON_ALERT_MIN_CONF before anything reaches weapon_filter
        # / the alert manager. Keeping this low just gives the *display*
        # smoothing something to work with on frames where confidence dips
        # briefly, instead of a hard gap.
        self.conf_threshold = conf_threshold if conf_threshold is not None else config.WEAPON_MIN_CONF
        self._model = None
        self._load_attempted = False

    @property
    def enabled(self) -> bool:
        from pathlib import Path

        return Path(self.model_path).exists()

    def _ensure_loaded(self):
        if self._model is not None or self._load_attempted:
            return
        self._load_attempted = True
        if not self.enabled:
            return
        from ultralytics import YOLO

        self._model = YOLO(str(self.model_path))

    def infer(self, frame) -> List[WeaponDetection]:
        self._ensure_loaded()
        if self._model is None:
            return []

        results = self._model.predict(
            frame,
            conf=self.conf_threshold,
            device=self.device,
            half=config.USE_HALF_PRECISION,
            imgsz=config.WEAPON_IMGSZ,
            verbose=False,
        )
        detections: List[WeaponDetection] = []
        if not results:
            return detections
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        names = result.names
        frame_area = float(frame.shape[0] * frame.shape[1])
        for i in range(len(boxes)):
            cls_idx = int(boxes.cls[i].item())
            cls_name = names.get(cls_idx, str(cls_idx))
            if cls_name not in config.WEAPON_THREAT_CLASSES:
                continue  # a confusor class (smartphone/wallet/banknote/card), not a threat
            conf = float(boxes.conf[i].item())
            x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[i].tolist()]
            if is_implausibly_large((x1, y1, x2, y2), frame_area):
                continue
            detections.append(
                WeaponDetection(
                    cls_name=cls_name,
                    conf=conf,
                    bbox=(x1, y1, x2, y2),
                )
            )
        return detections
