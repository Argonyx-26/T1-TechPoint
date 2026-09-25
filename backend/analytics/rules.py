"""Behavior-based anomaly rules layered on top of raw detection + tracking:

  * Loitering               -- person stationary in a zone past a time limit
  * Crowd threshold         -- person count in a zone reaches its limit (MEDIUM)
  * Crowd surge             -- zone count jumps suddenly, or well above its
                               recent average (HIGH)
  * Unattended object       -- an item is left behind once its owner departs
  * Restricted zone intrusion -- a person enters a marked no-go polygon
  * Wrong-direction movement  -- movement opposes a zone's allowed direction

Each rule is pure logic over `TrackState` + `Zone` objects (no ML involved),
so it can be exercised with synthetic data independent of the detector.
"""
import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from backend import config
from backend.geometry import angle_between_deg, euclidean, point_in_polygon
from backend.tracking import TrackState
from backend.zones import Zone

LOITERING = "LOITERING"
CROWD_SURGE = "CROWD_SURGE"
CROWD_THRESHOLD = "CROWD_THRESHOLD"
UNATTENDED_OBJECT = "UNATTENDED_OBJECT"
RESTRICTED_ZONE_INTRUSION = "RESTRICTED_ZONE_INTRUSION"
WRONG_DIRECTION = "WRONG_DIRECTION"


@dataclass
class RuleAlert:
    rule: str
    track_ids: List[int]
    message: str
    timestamp: float
    bbox: Optional[Tuple[float, float, float, float]] = None
    zone_id: Optional[str] = None
    zone_name: Optional[str] = None


class ZoneCountHistory:
    """Per-zone person-count history for the surge rule.

    Raw per-frame counts are median-smoothed over `smooth_s` first, so a
    one-frame detector dropout (6 -> 0 -> 6) can't masquerade as a +6 rise,
    and 5-6-5-6 flicker stays a flat ~5.5. Surge checks run on the smoothed
    series only.
    """

    def __init__(self):
        self._raw: deque = deque()       # (ts, raw count), last smooth_s seconds
        self._smooth: deque = deque()    # (ts, smoothed count), last avg_window_s seconds
        self.armed = True                # surge fires on the rising edge, re-arms once it clears

    def add(self, ts: float, count: int, smooth_s: float, avg_window_s: float) -> float:
        self._raw.append((ts, count))
        while self._raw and self._raw[0][0] < ts - smooth_s:
            self._raw.popleft()
        smoothed = statistics.median_low(c for _, c in self._raw)
        self._smooth.append((ts, smoothed))
        while self._smooth and self._smooth[0][0] < ts - avg_window_s:
            self._smooth.popleft()
        return smoothed

    def lowest_since(self, since_ts: float):
        """(ts, count) of the most recent minimum in the window -- where a rise started."""
        return min(((ts, c) for ts, c in self._smooth if ts >= since_ts), key=lambda x: (x[1], -x[0]))

    def average(self) -> float:
        return sum(c for _, c in self._smooth) / len(self._smooth)

    def span_s(self) -> float:
        return self._smooth[-1][0] - self._smooth[0][0] if self._smooth else 0.0


def _zone_for_point(point, zones: List[Zone]) -> Optional[Zone]:
    for zone in zones:
        if point_in_polygon(point, zone.polygon):
            return zone
    return None


class RuleEngine:
    """Stateful evaluator: call `evaluate` once per frame with the current
    track snapshot and zone config; it returns any freshly-triggered alerts.
    """

    def __init__(self):
        self.loiter_seconds = config.LOITER_SECONDS
        self.crowd_threshold = config.CROWD_COUNT_THRESHOLD
        self.surge_min_increase = config.SURGE_MIN_INCREASE
        self.surge_window_s = config.SURGE_WINDOW_S
        self.surge_avg_window_s = config.SURGE_AVG_WINDOW_S
        self.surge_avg_multiplier = config.SURGE_AVG_MULTIPLIER
        self.surge_min_people = config.SURGE_MIN_PEOPLE
        self.surge_smooth_s = config.SURGE_SMOOTH_S
        self.unattended_seconds = config.UNATTENDED_SECONDS
        self.unattended_radius_px = config.UNATTENDED_RADIUS_PX
        self.stationary_speed_px_s = config.STATIONARY_SPEED_PX_S
        self.wrong_direction_angle_deg = config.WRONG_DIRECTION_ANGLE_DEG
        self.min_track_history = config.MIN_TRACK_HISTORY

        self._dwell: Dict[Tuple[int, str], float] = {}
        self._dwell_last_ts: Dict[Tuple[int, str], float] = {}
        self._item_state: Dict[int, dict] = {}
        self._zone_counts: Dict[str, ZoneCountHistory] = {}
        # Latest raw person count per zone id, for the overlay / status API.
        self.zone_counts: Dict[str, int] = {}

    def update_thresholds(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)

    def evaluate(
        self, tracks: Dict[int, TrackState], zones: List[Zone], timestamp: float
    ) -> List[RuleAlert]:
        persons = [t for t in tracks.values() if t.cls_name == config.PERSON_CLASS]
        items = [t for t in tracks.values() if t.cls_name in config.ITEM_CLASSES]

        alerts: List[RuleAlert] = []
        alerts += self._check_restricted_zones(persons, zones, timestamp)
        alerts += self._check_crowd(persons, zones, timestamp)
        alerts += self._check_loitering(persons, zones, timestamp)
        alerts += self._check_wrong_direction(persons, zones, timestamp)
        alerts += self._check_unattended_objects(persons, items, zones, timestamp)
        return alerts

    def _check_restricted_zones(self, persons, zones, timestamp) -> List[RuleAlert]:
        alerts = []
        for zone in zones:
            if not zone.restricted:
                continue
            for person in persons:
                if len(person.history) < self.min_track_history:
                    continue
                if point_in_polygon(person.centroid, zone.polygon):
                    alerts.append(
                        RuleAlert(
                            rule=RESTRICTED_ZONE_INTRUSION,
                            track_ids=[person.track_id],
                            message=f"Person #{person.track_id} entered restricted zone '{zone.name}'",
                            timestamp=timestamp,
                            bbox=person.bbox,
                            zone_id=zone.id,
                            zone_name=zone.name,
                        )
                    )
        return alerts

    def _check_crowd(self, persons, zones, timestamp) -> List[RuleAlert]:
        """Two crowd rules on zones with crowd monitoring enabled (a
        crowd_threshold set): the absolute count limit (MEDIUM), and a surge
        -- a sudden rise, or a count well above the zone's recent average
        (HIGH). Every zone's live count is recorded for the overlay."""
        alerts = []
        live_zone_ids = set()
        for zone in zones:
            inside = [p for p in persons if point_in_polygon(p.centroid, zone.polygon)]
            self.zone_counts[zone.id] = len(inside)
            live_zone_ids.add(zone.id)
            if not zone.crowd_threshold:
                continue

            def make(rule, message):
                return RuleAlert(
                    rule=rule,
                    track_ids=[p.track_id for p in inside],
                    message=message,
                    timestamp=timestamp,
                    bbox=inside[0].bbox if inside else None,
                    zone_id=zone.id,
                    zone_name=zone.name,
                )

            if len(inside) >= zone.crowd_threshold:
                alerts.append(make(
                    CROWD_THRESHOLD,
                    f"Crowd threshold: {len(inside)} people in '{zone.name}' (threshold {zone.crowd_threshold})",
                ))

            surge = self._surge_message(zone, len(inside), timestamp)
            if surge:
                alerts.append(make(CROWD_SURGE, surge))

        for zone_id in list(self._zone_counts):
            if zone_id not in live_zone_ids:
                del self._zone_counts[zone_id]
        for zone_id in list(self.zone_counts):
            if zone_id not in live_zone_ids:
                del self.zone_counts[zone_id]
        return alerts

    def _surge_message(self, zone, count: int, timestamp: float) -> Optional[str]:
        hist = self._zone_counts.setdefault(zone.id, ZoneCountHistory())
        now = hist.add(timestamp, count, self.surge_smooth_s, self.surge_avg_window_s)

        from_ts, low = hist.lowest_since(timestamp - self.surge_window_s)
        rose = now - low >= self.surge_min_increase
        # The rolling-average check needs a real baseline, not the first
        # few seconds after a zone is drawn or the feed starts.
        avg = hist.average()
        above_avg = (
            hist.span_s() >= self.surge_window_s
            and now >= self.surge_min_people
            and now > self.surge_avg_multiplier * avg
        )

        if not (rose or above_avg):
            hist.armed = True
            return None
        if not hist.armed:
            return None  # still the same surge that already fired
        hist.armed = False
        if rose:
            secs = max(1, round(timestamp - from_ts))
            return f"Crowd surge in '{zone.name}': {low:g} -> {now:g} people in {secs}s"
        return (
            f"Crowd surge in '{zone.name}': {now:g} people vs "
            f"{avg:.1f} avg over {self.surge_avg_window_s:.0f}s"
        )

    def _check_loitering(self, persons, zones, timestamp) -> List[RuleAlert]:
        alerts = []
        active_keys = set()
        for person in persons:
            if len(person.history) < self.min_track_history:
                continue
            is_stationary = person.speed_px_s() < self.stationary_speed_px_s
            for zone in zones:
                key = (person.track_id, zone.id)
                inside = point_in_polygon(person.centroid, zone.polygon)
                if inside and is_stationary:
                    active_keys.add(key)
                    prev_dwell = self._dwell.get(key, 0.0)
                    prev_ts = self._dwell_last_ts.get(key, timestamp)
                    dt = min(max(0.0, timestamp - prev_ts), 1.0)
                    dwell = prev_dwell + dt
                    self._dwell[key] = dwell
                    self._dwell_last_ts[key] = timestamp

                    limit = zone.loiter_seconds or self.loiter_seconds
                    if dwell >= limit:
                        alerts.append(
                            RuleAlert(
                                rule=LOITERING,
                                track_ids=[person.track_id],
                                message=(
                                    f"Person #{person.track_id} stationary in "
                                    f"'{zone.name}' for {dwell:.0f}s"
                                ),
                                timestamp=timestamp,
                                bbox=person.bbox,
                                zone_id=zone.id,
                                zone_name=zone.name,
                            )
                        )
                else:
                    self._dwell[key] = 0.0
                    self._dwell_last_ts[key] = timestamp

        # drop stale dwell keys for tracks/zones no longer relevant
        stale = [k for k in self._dwell if k not in active_keys and self._dwell[k] == 0.0]
        for k in stale:
            self._dwell.pop(k, None)
            self._dwell_last_ts.pop(k, None)
        return alerts

    def _check_wrong_direction(self, persons, zones, timestamp) -> List[RuleAlert]:
        alerts = []
        for zone in zones:
            if not zone.allowed_direction:
                continue
            for person in persons:
                if len(person.history) < self.min_track_history:
                    continue
                if not point_in_polygon(person.centroid, zone.polygon):
                    continue
                mv = person.movement_vector()
                if math.hypot(*mv) < 5.0:
                    continue  # not moving enough to judge direction reliably
                angle = angle_between_deg(mv, zone.allowed_direction)
                if angle >= self.wrong_direction_angle_deg:
                    alerts.append(
                        RuleAlert(
                            rule=WRONG_DIRECTION,
                            track_ids=[person.track_id],
                            message=(
                                f"Person #{person.track_id} moving against allowed "
                                f"direction in '{zone.name}'"
                            ),
                            timestamp=timestamp,
                            bbox=person.bbox,
                            zone_id=zone.id,
                            zone_name=zone.name,
                        )
                    )
        return alerts

    def _check_unattended_objects(self, persons, items, zones, timestamp) -> List[RuleAlert]:
        alerts = []
        live_ids = {item.track_id for item in items}
        for item in items:
            nearest_dist = min(
                (euclidean(item.centroid, p.centroid) for p in persons), default=math.inf
            )
            near = nearest_dist <= self.unattended_radius_px

            st = self._item_state.setdefault(
                item.track_id, {"last_person_near_ts": timestamp if near else None, "ever_had_person": near}
            )
            if near:
                st["last_person_near_ts"] = timestamp
                st["ever_had_person"] = True

            if st["ever_had_person"] and st["last_person_near_ts"] is not None:
                alone_duration = timestamp - st["last_person_near_ts"]
                stationary = item.speed_px_s() < self.stationary_speed_px_s
                if not near and stationary and alone_duration >= self.unattended_seconds:
                    zone = _zone_for_point(item.centroid, zones)
                    alerts.append(
                        RuleAlert(
                            rule=UNATTENDED_OBJECT,
                            track_ids=[item.track_id],
                            message=(
                                f"{item.cls_name.capitalize()} left unattended for "
                                f"{alone_duration:.0f}s"
                            ),
                            timestamp=timestamp,
                            bbox=item.bbox,
                            zone_id=zone.id if zone else None,
                            zone_name=zone.name if zone else None,
                        )
                    )

        # forget items no longer tracked
        for tid in list(self._item_state):
            if tid not in live_ids:
                del self._item_state[tid]
        return alerts
