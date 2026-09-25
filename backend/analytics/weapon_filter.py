"""Temporal smoothing for weapon detections.

A single-frame weapon detection is treated as unconfirmed "noise" until the
same class has appeared in enough of the last N frames -- this is what
prevents a lone spurious detection (a shadow, a phone, a hand gesture) from
firing a CRITICAL alert on its own. It does not fix a detector that is
systematically wrong about what it sees; it only suppresses one-off
flickers from an otherwise-reasonable detector. Validate the underlying
model's false-positive rate on real, weapon-free footage before trusting
its output here.
"""
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple, Union

Box = Tuple[float, float, float, float]


def _near(a: Box, b: Box, factor: float) -> bool:
    """Centres within `factor` x the larger box side: the same object, moved a little."""
    side = max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1])
    ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 <= factor * side


class WeaponTemporalFilter:
    def __init__(self, window: int = 8, min_hits: int = 5, track_dist: Optional[float] = None):
        if min_hits > window:
            raise ValueError("min_hits cannot exceed window")
        self.window = window
        self.min_hits = min_hits
        if track_dist is None:
            from backend import config
            track_dist = config.WEAPON_TRACK_DIST
        self.track_dist = track_dist
        # per frame: class -> bbox (None when the caller only passes class names)
        self._history: Deque[Dict[str, Optional[Box]]] = deque(maxlen=window)

    def update(self, detected: Union[set, Dict[str, Box]]) -> List[str]:
        """Feed one frame's weapon detections: a {class: bbox} dict, or a bare
        set of class names (no location check). Returns the classes that are
        *sustained* as of this frame (i.e. hit the min_hits/window threshold)
        -- usually empty. With boxes, only hits near this frame's box count,
        so flickers in different places never add up to one weapon.
        Cooldown/dedup against repeated firing is the alert manager's job."""
        frame = dict(detected) if isinstance(detected, dict) else {c: None for c in detected}
        self._history.append(frame)
        if len(self._history) < self.window:
            return []
        confirmed = []
        for cls, box in frame.items():
            hits = sum(
                1 for past in self._history
                if cls in past and (box is None or past[cls] is None or _near(box, past[cls], self.track_dist))
            )
            if hits >= self.min_hits:
                confirmed.append(cls)
        return confirmed
