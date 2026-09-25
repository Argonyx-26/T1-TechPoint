"""Sustained-detection filter for the weapon model.

A single frame is never enough: a class is CONFIRMED only when it was seen at or
above the alert threshold in at least `min_hits` of the last `window` inference
frames. The on-screen "CRITICAL THREAT" label and weapon alerts both key off this
confirmed state, never off a raw detection.
"""
from collections import deque

from backend import config


class WeaponFilter:
    def __init__(self, window: int = config.WEAPON_WINDOW, min_hits: int = config.WEAPON_MIN_HITS,
                 alert_conf: float = config.WEAPON_ALERT_CONF,
                 threat_classes=frozenset(config.WEAPON_THREAT_CLASSES)):
        if not 1 <= min_hits <= window:
            raise ValueError("min_hits must be between 1 and window")
        self.min_hits = min_hits
        self.alert_conf = alert_conf
        self.threat_classes = set(threat_classes)
        self.history: deque[set] = deque(maxlen=window)
        self.confirmed: set[str] = set()

    def update(self, detections) -> set[str]:
        """Feed one inference frame of detections (objects with .cls_name and .conf).

        Returns the set of confirmed threat classes after this frame.
        """
        hits = {d.cls_name for d in detections
                if d.cls_name in self.threat_classes and d.conf >= self.alert_conf}
        self.history.append(hits)
        counts: dict[str, int] = {}
        for frame_hits in self.history:
            for name in frame_hits:
                counts[name] = counts.get(name, 0) + 1
        self.confirmed = {name for name, n in counts.items() if n >= self.min_hits}
        return self.confirmed

    def reset(self) -> None:
        self.history.clear()
        self.confirmed = set()
