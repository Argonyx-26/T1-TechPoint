"""Behaviour rules over tracked people and zones. Pure logic (no model), driven per frame.

Implemented: restricted_intrusion, loitering, crowd_surge.
unattended_seconds and wrong_direction_angle_deg are stored and editable for the
dashboard, but those two rules are not implemented (reference sheet: skip them).
"""
import threading
import time
from dataclasses import dataclass

from backend import config
from backend.zones import Zone, point_in_polygon

# A person briefly lost by the tracker keeps their zone timer for this long
ZONE_EXIT_GRACE_SECONDS = 2.0

_LIMITS = {
    # name: (type, min, max)
    "loiter_seconds": (float, 1, 3600),
    "crowd_threshold": (int, 1, 1000),
    "unattended_seconds": (float, 1, 3600),
    "wrong_direction_angle_deg": (float, 0, 180),
}


class Thresholds:
    def __init__(self, values: dict | None = None):
        self._lock = threading.Lock()
        self._values = dict(config.DEFAULT_THRESHOLDS)
        if values:
            self.update(values)

    def get(self) -> dict:
        with self._lock:
            return dict(self._values)

    def __getitem__(self, name):
        with self._lock:
            return self._values[name]

    def update(self, raw) -> dict:
        """Validate a partial or full update and apply it atomically. Raises ValueError."""
        if not isinstance(raw, dict):
            raise ValueError("body must be a JSON object")
        unknown = set(raw) - set(_LIMITS)
        if unknown:
            raise ValueError(f"unknown threshold(s): {', '.join(sorted(unknown))}")
        clean = {}
        for name, value in raw.items():
            kind, lo, hi = _LIMITS[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number")
            if kind is int:
                if float(value) != int(value):
                    raise ValueError(f"{name} must be a whole number")
                value = int(value)
            else:
                value = float(value)
            if not lo <= value <= hi:
                raise ValueError(f"{name} must be between {lo} and {hi}")
            clean[name] = value
        with self._lock:
            self._values.update(clean)
            return dict(self._values)


@dataclass
class RuleEvent:
    rule: str
    description: str
    zone: str | None = None      # zone name, for display
    track_id: int | None = None
    key: tuple | None = None     # cooldown key (zone id, track id)


class RulesEngine:
    def __init__(self, thresholds: Thresholds):
        self.thresholds = thresholds
        self._entered: dict[tuple, float] = {}    # (zone id, track id) -> first seen inside
        self._last_inside: dict[tuple, float] = {}
        self._loiter_fired: set[tuple] = set()
        self._intruding: set[tuple] = set()
        self._crowded: set = set()

    def reset(self) -> None:
        self.__init__(self.thresholds)

    def evaluate(self, tracks, zones: list[Zone], frame_size: tuple[int, int],
                 now: float | None = None) -> list[RuleEvent]:
        now = time.time() if now is None else now
        w, h = frame_size
        people = [t for t in tracks if t.cls_name == "person"]
        loiter_limit = self.thresholds["loiter_seconds"]
        crowd_limit = self.thresholds["crowd_threshold"]
        events: list[RuleEvent] = []
        intruding_now, crowded_now = set(), set()

        for zone in zones:
            poly = zone.pixel_polygon(w, h)
            inside = [t for t in people if point_in_polygon(t.foot_point, poly)]
            for t in inside:
                key = (zone.id, t.track_id)
                self._last_inside[key] = now
                entered = self._entered.setdefault(key, now)

                if zone.restricted:
                    intruding_now.add(key)
                    if key not in self._intruding:
                        events.append(RuleEvent(
                            "restricted_intrusion",
                            f"Person #{t.track_id} entered restricted zone '{zone.name}'",
                            zone.name, t.track_id, key))

                dwell = now - entered
                if dwell >= loiter_limit and key not in self._loiter_fired:
                    self._loiter_fired.add(key)
                    events.append(RuleEvent(
                        "loitering",
                        f"Person #{t.track_id} has stayed in zone '{zone.name}' for "
                        f"{dwell:.0f}s (limit {loiter_limit:g}s)",
                        zone.name, t.track_id, key))

            if len(inside) >= crowd_limit:
                crowded_now.add(zone.id)
                if zone.id not in self._crowded:
                    events.append(RuleEvent(
                        "crowd_surge",
                        f"{len(inside)} people in zone '{zone.name}' (threshold {crowd_limit})",
                        zone.name, None, (zone.id, None)))

        # With no zones drawn, crowd surge still watches the whole frame
        if not zones and len(people) >= crowd_limit:
            crowded_now.add(None)
            if None not in self._crowded:
                events.append(RuleEvent(
                    "crowd_surge", f"{len(people)} people in view (threshold {crowd_limit})",
                    None, None, (None, None)))

        self._intruding = intruding_now
        self._crowded = crowded_now
        for key in [k for k, t in self._last_inside.items() if now - t > ZONE_EXIT_GRACE_SECONDS]:
            self._last_inside.pop(key, None)
            self._entered.pop(key, None)
            self._loiter_fired.discard(key)
        return events
