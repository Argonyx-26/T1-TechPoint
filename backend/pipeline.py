"""Ties detection, tracking, the rules engine, severity scoring, and the
alert manager together into a single per-frame `process()` call, and draws
the overlay the dashboard displays (bounding boxes, zone polygons).
"""
import time
from collections import deque
from typing import Dict, List

import cv2
import numpy as np

from backend import config
from backend.alerts.manager import AlertManager
from backend.analytics.rules import RuleEngine
from backend.analytics.weapon_filter import WeaponTemporalFilter
from backend.detection.object_detector import ObjectDetector
from backend.detection.weapon_detector import WeaponDetection, WeaponDetector
from backend.tracking import TrackManager
from backend.zones import Zone, ZoneStore

BOX_COLOR_PERSON = (255, 178, 60)   # BGR
BOX_COLOR_ITEM = (60, 200, 255)
BOX_COLOR_OTHER = (180, 180, 180)
BOX_COLOR_WEAPON = (0, 0, 255)
ZONE_COLOR_RESTRICTED = (0, 0, 220)
ZONE_COLOR_NORMAL = (0, 200, 0)


class FramePipeline:
    def __init__(
        self,
        object_detector: ObjectDetector = None,
        weapon_detector: WeaponDetector = None,
        zone_store: ZoneStore = None,
        alert_manager: AlertManager = None,
    ):
        self.object_detector = object_detector or ObjectDetector()
        self.weapon_detector = weapon_detector or WeaponDetector()
        self.zone_store = zone_store or ZoneStore()
        self.alert_manager = alert_manager or AlertManager()
        self.rule_engine = RuleEngine()
        self.track_manager = TrackManager()
        self.weapon_filter = WeaponTemporalFilter()
        self.last_object_counts: Dict[str, int] = {}
        # Purely visual smoothing for the weapon overlay: a per-class recent-
        # confidence buffer (so the on-screen % doesn't jump frame to frame)
        # and a short box-persistence window (so the box doesn't vanish for
        # one dropped frame). Does NOT affect alert-firing logic below, which
        # still runs on raw per-frame weapon_detections + weapon_filter.
        self._weapon_display_state: Dict[str, dict] = {}

    def process(self, frame: np.ndarray) -> np.ndarray:
        timestamp = time.time()
        zones = self.zone_store.list()

        detections = self.object_detector.infer(frame)
        tracks = self.track_manager.update(detections, timestamp)
        rule_alerts = self.rule_engine.evaluate(tracks, zones, timestamp)
        # Low-floor detections (down to WEAPON_MIN_CONF) -- used only for
        # keeping the on-screen box/label visually continuous. Anything that
        # can actually raise a CRITICAL alert must additionally clear the
        # stricter WEAPON_ALERT_MIN_CONF bar below.
        weapon_detections = self.weapon_detector.infer(frame)
        alert_eligible = [wd for wd in weapon_detections if wd.conf >= config.WEAPON_ALERT_MIN_CONF]

        # Only a *sustained* weapon detection (seen in most of the last few
        # frames) counts as confirmed -- a lone spurious hit from the
        # detector doesn't get to trigger an alert OR the on-screen label.
        # See weapon_filter.py. This runs before the display step below so
        # the overlay can gate on the same bar as a real alert, not a
        # single-frame threshold touch.
        latest_by_class = {wd.cls_name: wd for wd in alert_eligible}
        confirmed_classes = set(self.weapon_filter.update(set(latest_by_class)))

        display_weapons = self._smooth_weapon_display(weapon_detections, confirmed_classes, timestamp)

        annotated = frame.copy()
        self._draw_zones(annotated, zones)
        self._draw_tracks(annotated, tracks)
        self._draw_weapons(annotated, display_weapons)
        self._update_object_counts(tracks, alert_eligible)

        if rule_alerts:
            self.alert_manager.ingest_rule_alerts(rule_alerts, frame=annotated)

        for cls_name in confirmed_classes:
            wd = latest_by_class.get(cls_name)
            if wd is None:
                continue
            self.alert_manager.ingest_weapon_alert(
                wd.cls_name, wd.conf, wd.bbox, timestamp, frame=annotated
            )

        return annotated

    def _smooth_weapon_display(self, weapon_detections, confirmed_classes, timestamp: float) -> List[WeaponDetection]:
        """Purely cosmetic: average the last few frames' confidence per class
        (so the on-screen % stops jumping around) and keep the last known box
        drawn for a brief window even on a frame with no raw detection (so it
        doesn't flicker in and out).

        Gating: a class only enters display state once it's been *confirmed*
        by weapon_filter this frame -- the same bar a real alert requires,
        not just a single-frame threshold touch. Without this, a detection
        that briefly touched the alert threshold once but never sustained
        (and so never actually raised an alert) could still show a
        misleading "CRITICAL THREAT" label with nothing in the alert feed to
        back it up. Once confirmed and showing, any detection down to the
        low floor keeps it alive/refreshed, which is what carries it through
        a frame or two of motion blur without a fresh gap.
        """
        for wd in weapon_detections:
            already_acquired = wd.cls_name in self._weapon_display_state
            if not already_acquired and wd.cls_name not in confirmed_classes:
                continue  # not yet confirmed by the temporal filter -- don't show it
            state = self._weapon_display_state.setdefault(
                wd.cls_name, {"conf_history": deque(maxlen=config.WEAPON_CONF_SMOOTH_WINDOW)}
            )
            state["conf_history"].append(wd.conf)
            state["last_bbox"] = wd.bbox
            state["last_seen_ts"] = timestamp

        stale = [
            cls_name
            for cls_name, state in self._weapon_display_state.items()
            if timestamp - state["last_seen_ts"] > config.WEAPON_BOX_PERSIST_SECONDS
        ]
        for cls_name in stale:
            del self._weapon_display_state[cls_name]

        display: List[WeaponDetection] = []
        for cls_name, state in self._weapon_display_state.items():
            smoothed_conf = sum(state["conf_history"]) / len(state["conf_history"])
            display.append(WeaponDetection(cls_name=cls_name, conf=smoothed_conf, bbox=state["last_bbox"]))
        return display

    def _update_object_counts(self, tracks, weapon_detections):
        counts: Dict[str, int] = {}
        for track in tracks.values():
            counts[track.cls_name] = counts.get(track.cls_name, 0) + 1
        for wd in weapon_detections:
            counts[wd.cls_name] = counts.get(wd.cls_name, 0) + 1
        self.last_object_counts = counts

    def _draw_zones(self, frame, zones: List[Zone]):
        for zone in zones:
            if not zone.polygon:
                continue
            pts = np.array(zone.polygon, dtype=np.int32).reshape((-1, 1, 2))
            color = ZONE_COLOR_RESTRICTED if zone.restricted else ZONE_COLOR_NORMAL
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
            x, y = zone.polygon[0]
            cv2.putText(
                frame, zone.name, (int(x), int(y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
            )

    def _draw_tracks(self, frame, tracks):
        for track in tracks.values():
            x1, y1, x2, y2 = [int(v) for v in track.bbox]
            if track.cls_name == config.PERSON_CLASS:
                color = BOX_COLOR_PERSON
            elif track.cls_name in config.ITEM_CLASSES:
                color = BOX_COLOR_ITEM
            else:
                color = BOX_COLOR_OTHER
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{track.cls_name} #{track.track_id}"
            cv2.putText(
                frame, label, (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
            )

    def _draw_weapons(self, frame, weapon_detections):
        for wd in weapon_detections:
            x1, y1, x2, y2 = [int(v) for v in wd.bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR_WEAPON, 4)
            label = f"CRITICAL THREAT: {wd.cls_name.upper()} {wd.conf:.0%}"
            font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
            (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)
            label_y = max(text_h + baseline + 4, y1)
            # Solid background behind the label so it stays legible over any
            # footage, not just dark backgrounds.
            cv2.rectangle(
                frame,
                (x1, label_y - text_h - baseline - 4),
                (x1 + text_w + 8, label_y),
                BOX_COLOR_WEAPON,
                -1,
            )
            cv2.putText(
                frame, label, (x1 + 4, label_y - baseline),
                font, scale, (255, 255, 255), thickness,
            )
