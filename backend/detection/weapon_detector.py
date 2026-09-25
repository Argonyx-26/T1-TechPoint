"""Pretrained weapon model wrapper. Degrades gracefully: if the weights are missing or
fail to load, `available` is False and `detect` returns nothing, so the rest of the
system keeps running without weapon alerts."""
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend import config

log = logging.getLogger(__name__)


@dataclass
class WeaponDetection:
    cls_name: str
    conf: float
    bbox: tuple  # x1, y1, x2, y2 in pixels


class WeaponDetector:
    def __init__(self, weights: Path | None = None, enabled: bool = True):
        weights = Path(weights or config.WEAPON_MODEL_PATH)
        self.model = None
        self.names: dict = {}
        if not enabled:
            return
        if not weights.is_file():
            log.warning("Weapon model not found at %s - running without weapon alerts", weights)
            return
        try:
            from ultralytics import YOLO

            # Loaded once per process; replacing the file needs a server restart
            self.model = YOLO(str(weights))
            self.names = self.model.names
            log.info("Weapon model loaded: %s classes=%s device=%s half=%s",
                     weights.name, list(self.names.values()), config.DEVICE, config.HALF)
        except Exception as exc:
            log.warning("Weapon model failed to load (%s) - running without weapon alerts", exc)
            self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    def warmup(self) -> None:
        if self.available:
            self.model.predict(np.zeros((640, 640, 3), dtype="uint8"), device=config.DEVICE,
                               quantize=config.PRECISION, verbose=False)

    def detect(self, frame) -> list[WeaponDetection]:
        """Threat-class detections at or above the display threshold."""
        if not self.available:
            return []
        result = self.model.predict(
            frame,
            conf=config.WEAPON_DISPLAY_CONF,
            imgsz=config.INFER_IMGSZ,
            device=config.DEVICE,
            quantize=config.PRECISION,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        out = []
        for c, p, b in zip(boxes.cls.int().tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()):
            name = self.names[c]
            if name in config.WEAPON_THREAT_CLASSES:
                out.append(WeaponDetection(name, p, tuple(b)))
        return out
