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


def _near(a: Box, b: Box, factor: float, min_dist: float = 0.0) -> bool:
    """Centres within `factor` x the larger box side (or min_dist, whichever
    is larger): the same object, moved a little."""
    side = max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1])
    ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 <= max(factor * side, min_dist)


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
        self._confs: Deque[Dict[str, float]] = deque(maxlen=window)

    def update(self, detected: Union[set, Dict[str, Box]], frame_size: Optional[Tuple[int, int]] = None,
               confs: Optional[Dict[str, float]] = None) -> List[str]:
        """Feed one frame's weapon detections: a {class: bbox} dict, or a bare
        set of class names (no location check). Returns the classes that are
        *sustained* as of this frame (i.e. hit the min_hits/window threshold)
        -- usually empty. With boxes, only hits near this frame's box count,
        so flickers in different places never add up to one weapon.
        Cooldown/dedup against repeated firing is the alert manager's job."""
        frame = dict(detected) if isinstance(detected, dict) else {c: None for c in detected}
        min_dist = 0.0
        if frame_size:
            from backend import config
            min_dist = config.WEAPON_TRACK_MIN_FRAC * (frame_size[0] ** 2 + frame_size[1] ** 2) ** 0.5
        self._history.append(frame)
        self._confs.append(dict(confs or {}))
        from backend import config
        confirmed = []
        for cls, box in frame.items():
            same = [i for i, past in enumerate(self._history)
                    if cls in past and (box is None or past[cls] is None or _near(box, past[cls], self.track_dist, min_dist))]
            if len(self._history) >= self.window and len(same) >= self.min_hits:
                confirmed.append(cls)
            elif confs is not None and sum(
                    1 for i in same if self._confs[i].get(cls, 0.0) >= config.WEAPON_STRONG_CONF) >= config.WEAPON_STRONG_HITS:
                confirmed.append(cls)   # fast path: two very confident sightings at the same spot
        return confirmed
