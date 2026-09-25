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
from typing import Deque, List, Set


class WeaponTemporalFilter:
    def __init__(self, window: int = 8, min_hits: int = 5):
        if min_hits > window:
            raise ValueError("min_hits cannot exceed window")
        self.window = window
        self.min_hits = min_hits
        self._history: Deque[Set[str]] = deque(maxlen=window)

    def update(self, detected_classes: Set[str]) -> List[str]:
        """Feed one frame's set of detected weapon class names. Returns the
        classes that are *sustained* as of this frame (i.e. hit the
        min_hits/window threshold) -- usually empty. Cooldown/dedup against
        repeated firing is the alert manager's job, not this filter's."""
        self._history.append(set(detected_classes))
        if len(self._history) < self.window:
            return []
        all_seen = set().union(*self._history)
        return [
            cls
            for cls in all_seen
            if sum(1 for frame_classes in self._history if cls in frame_classes) >= self.min_hits
        ]
