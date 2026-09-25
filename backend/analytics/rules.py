"""Behavior-based anomaly rules layered on top of raw detection + tracking:

  * Loitering               -- person stationary in a zone past a time limit
  * Crowd surge             -- person count in a zone exceeds a threshold
  * Unattended object       -- an item is left behind once its owner departs
  * Restricted zone intrusion -- a person enters a marked no-go polygon
  * Wrong-direction movement  -- movement opposes a zone's allowed direction

Each rule is pure logic over `TrackState` + `Zone` objects (no ML involved),
so it can be exercised with synthetic data independent of the detector.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from backend import config
from backend.geometry import angle_between_deg, euclidean, point_in_polygon
from backend.tracking import TrackState
from backend.zones import Zone

LOITERING = "LOITERING"
CROWD_SURGE = "CROWD_SURGE"
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
        self.unattended_seconds = config.UNATTENDED_SECONDS
        self.unattended_radius_px = config.UNATTENDED_RADIUS_PX
        self.stationary_speed_px_s = config.STATIONARY_SPEED_PX_S
        self.wrong_direction_angle_deg = config.WRONG_DIRECTION_ANGLE_DEG
        self.min_track_history = config.MIN_TRACK_HISTORY

        self._dwell: Dict[Tuple[int, str], float] = {}
        self._dwell_last_ts: Dict[Tuple[int, str], float] = {}
        self._item_state: Dict[int, dict] = {}

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
        alerts += self._check_crowd_surge(persons, zones, timestamp)
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

    def _check_crowd_surge(self, persons, zones, timestamp) -> List[RuleAlert]:
        alerts = []
        for zone in zones:
            if not zone.crowd_threshold:
                continue
            inside = [p for p in persons if point_in_polygon(p.centroid, zone.polygon)]
            if len(inside) >= zone.crowd_threshold:
                alerts.append(
                    RuleAlert(
                        rule=CROWD_SURGE,
                        track_ids=[p.track_id for p in inside],
                        message=(
                            f"{len(inside)} people in '{zone.name}' "
                            f"(threshold {zone.crowd_threshold})"
                        ),
                        timestamp=timestamp,
                        bbox=inside[0].bbox if inside else None,
                        zone_id=zone.id,
                        zone_name=zone.name,
                    )
                )
        return alerts

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
