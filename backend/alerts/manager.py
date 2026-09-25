"""Central alert store: turns raw rule/weapon triggers into a deduped,
cooldown-gated, ranked, evidence-backed feed for the dashboard.
"""
import itertools
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from backend import config
from backend.analytics import severity
from backend.analytics.rules import CROWD_SURGE, CROWD_THRESHOLD, RuleAlert
from backend.audit import audit

WEAPON_RULE = "WEAPON"
# Zone-level rules: the people inside change every frame, so cooldown is
# per (rule, zone) -- keying on track ids would let every membership change
# through as a "new" alert.
ZONE_KEYED_RULES = {CROWD_THRESHOLD, CROWD_SURGE}

_id_counter = itertools.count(1)


@dataclass
class Alert:
    id: int
    rule: str
    message: str
    score: int
    band: str
    timestamp: float
    track_ids: List[int]
    zone_id: Optional[str] = None
    zone_name: Optional[str] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    evidence_path: Optional[str] = None
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[float] = None
    camera_id: Optional[str] = None
    camera_name: Optional[str] = None
    camera_code: Optional[str] = None
    location: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "rule": self.rule,
            "type": self.rule.replace("_", " ").title(),
            "message": self.message,
            "description": self.message,
            "score": self.score,
            "band": self.band,
            "severity": self.band.lower(),
            # Wall-clock "HH:MM:SS" for display; created_at is the epoch value.
            "timestamp": time.strftime("%H:%M:%S", time.localtime(self.timestamp)),
            "created_at": self.timestamp,
            "track_ids": self.track_ids,
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "bbox": self.bbox,
            "has_evidence": self.evidence_path is not None,
            "acknowledged": self.acknowledged_at is not None,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at,
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "camera_code": self.camera_code,
            "location": self.location,
        }


class AlertManager:
    def __init__(self, max_alerts: int = 500, evidence_dir: Path = config.EVIDENCE_DIR):
        self._lock = threading.Lock()
        self._alerts: List[Alert] = []
        self._max_alerts = max_alerts
        self._evidence_dir = Path(evidence_dir)
        self._evidence_dir.mkdir(parents=True, exist_ok=True)
        self._last_fired: Dict[Tuple, float] = {}
        self._total_count = 0

    def _on_cooldown(self, key: Tuple, timestamp: float) -> bool:
        last = self._last_fired.get(key)
        return last is not None and (timestamp - last) < config.ALERT_COOLDOWN_SECONDS

    def _co_occurring_rules(self, zone_id, timestamp: float, camera_id=None) -> List[str]:
        if zone_id is None:
            return []
        window_start = timestamp - config.CO_OCCURRENCE_WINDOW_SECONDS
        with self._lock:
            return [
                a.rule for a in self._alerts
                if a.zone_id == zone_id and a.camera_id == camera_id and a.timestamp >= window_start
            ]

    @staticmethod
    def _camera_fields(camera) -> dict:
        if camera is None:
            return {}
        return {
            "camera_id": camera.id,
            "camera_name": camera.name,
            "camera_code": camera.code,
            "location": camera.location.to_dict(),
        }

    @staticmethod
    def _placed(message: str, camera) -> str:
        """"Main Gate (CAM-02): Weapon detected - knife 81%"."""
        if camera is None:
            return message
        # "CAM-01: ..." when the place is just the code, not "CAM-01 (CAM-01): ..."
        if not camera.place.strip() or camera.place.strip().upper() == camera.code:
            return f"{camera.code}: {message}"
        return f"{camera.place} ({camera.code}): {message}"

    def ingest_rule_alerts(self, rule_alerts: List[RuleAlert], frame=None, camera=None) -> List[Alert]:
        created = []
        cam_id = camera.id if camera is not None else None
        for ra in rule_alerts:
            # Cooldowns are per camera: the same rule on another camera is a new event.
            if ra.rule in ZONE_KEYED_RULES:
                key = (cam_id, ra.rule, ra.zone_id)
            else:
                key = (cam_id, ra.rule, ra.zone_id, tuple(sorted(ra.track_ids)))
            if self._on_cooldown(key, ra.timestamp):
                continue
            self._last_fired[key] = ra.timestamp

            siblings = self._co_occurring_rules(ra.zone_id, ra.timestamp, cam_id)
            score, band = severity.score_alert(ra.rule, siblings)

            alert = Alert(
                id=next(_id_counter),
                rule=ra.rule,
                message=self._placed(ra.message, camera),
                score=score,
                band=band,
                timestamp=ra.timestamp,
                track_ids=ra.track_ids,
                zone_id=ra.zone_id,
                zone_name=ra.zone_name,
                bbox=ra.bbox,
                **self._camera_fields(camera),
            )
            self._save_evidence(alert, frame)
            self._store(alert)
            created.append(alert)
        return created

    def ingest_weapon_alert(self, cls_name: str, conf: float, bbox, timestamp: float, frame=None,
                            camera=None) -> Optional[Alert]:
        # Weapon detections have no stable track id (single-shot detector), so
        # dedupe/cooldown on camera + class name instead of a sustained
        # detection spamming one alert per frame.
        key = (camera.id if camera is not None else None, WEAPON_RULE, cls_name)
        if self._on_cooldown(key, timestamp):
            return None
        self._last_fired[key] = timestamp

        score, band = severity.score_alert(WEAPON_RULE, [])
        alert = Alert(
            id=next(_id_counter),
            rule=WEAPON_RULE,
            message=self._placed(f"Weapon detected - {cls_name} {conf:.0%}", camera),
            score=score,
            band=band,
            timestamp=timestamp,
            track_ids=[],
            bbox=bbox,
            **self._camera_fields(camera),
        )
        self._save_evidence(alert, frame)
        self._store(alert)
        return alert

    BAND_SCORES = {"LOW": 30, "MEDIUM": 50, "HIGH": 70, "CRITICAL": 95}

    def ingest_event(self, rule: str, message: str, timestamp: float, camera=None, band: Optional[str] = None,
                     bbox=None, frame=None, track_ids: Optional[List[int]] = None,
                     dedupe_key: Optional[Tuple] = None, zone_id: Optional[str] = None,
                     zone_name: Optional[str] = None) -> Optional[Alert]:
        """Alerts raised outside the per-frame rules engine (blind-spot
        tracking, behaviour detection). `band` pins the severity; otherwise
        it is scored like any rule (weight + co-occurrence on this camera).
        Cooldown per `dedupe_key` (default: camera + rule)."""
        cam_id = camera.id if camera is not None else None
        key = dedupe_key or (cam_id, rule)
        if self._on_cooldown(key, timestamp):
            return None
        self._last_fired[key] = timestamp
        siblings = self._co_occurring_camera_rules(cam_id, timestamp)
        if band:
            score = self.BAND_SCORES[band]
        else:
            score, band = severity.score_alert(rule, siblings)
        alert = Alert(
            id=next(_id_counter),
            rule=rule,
            message=self._placed(message, camera),
            score=score,
            band=band,
            timestamp=timestamp,
            track_ids=list(track_ids or []),
            zone_id=zone_id,
            zone_name=zone_name,
            bbox=bbox,
            **self._camera_fields(camera),
        )
        self._save_evidence(alert, frame)
        self._store(alert)
        return alert

    def _co_occurring_camera_rules(self, camera_id, timestamp: float) -> List[str]:
        """Rules fired on the same camera within the co-occurrence window."""
        if camera_id is None:
            return []
        window_start = timestamp - config.CO_OCCURRENCE_WINDOW_SECONDS
        with self._lock:
            return [a.rule for a in self._alerts if a.camera_id == camera_id and a.timestamp >= window_start]

    def _save_evidence(self, alert: Alert, frame):
        if frame is None:
            return
        import cv2  # local import: keeps this module importable without opencv

        path = self._evidence_dir / f"alert_{alert.id}.jpg"
        try:
            cv2.imwrite(str(path), frame)
            alert.evidence_path = str(path)
        except Exception:
            pass

    def _store(self, alert: Alert):
        with self._lock:
            self._alerts.append(alert)
            self._total_count += 1
            if len(self._alerts) > self._max_alerts:
                self._alerts = self._alerts[-self._max_alerts :]
        audit("ALERT_FIRED", f"{alert.band} {alert.rule}: {alert.message}", alert_id=alert.id,
              camera_id=alert.camera_id)

    def acknowledge(self, alert_id: int, who: str = "operator") -> Optional[Alert]:
        with self._lock:
            alert = next((a for a in self._alerts if a.id == alert_id), None)
            if alert is None:
                return None
            if alert.acknowledged_at is None:
                alert.acknowledged_by = who
                alert.acknowledged_at = time.time()
                first = True
            else:
                first = False
        if first:
            audit("ALERT_ACKNOWLEDGED", f"{alert.rule}: {alert.message}", actor="operator", alert_id=alert_id)
        return alert

    def ranked(self, limit: int = 50) -> List[Alert]:
        with self._lock:
            alerts = sorted(self._alerts, key=lambda a: (-a.score, -a.timestamp))
            return alerts[:limit]

    def clear(self) -> int:
        """Drop every stored alert; returns how many were removed."""
        with self._lock:
            removed = len(self._alerts)
            self._alerts = []
            return removed

    def active_count(self, camera_id: Optional[str] = None, band: Optional[str] = None) -> int:
        """Unacknowledged alerts currently in the feed, optionally for one
        camera and/or one severity band."""
        with self._lock:
            return sum(
                1 for a in self._alerts
                if a.acknowledged_at is None
                and (camera_id is None or a.camera_id == camera_id)
                and (band is None or a.band == band)
            )

    def camera_summary(self, camera_id: str) -> dict:
        """Worst band and latest message among a camera's unacknowledged alerts
        (map marker colour + popup)."""
        order = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1, "LOW": 0}
        with self._lock:
            active = [a for a in self._alerts if a.camera_id == camera_id and a.acknowledged_at is None]
        if not active:
            return {"worst_alert": None, "latest_alert": None}
        worst = max(active, key=lambda a: (order.get(a.band, 0), a.timestamp))
        latest = max(active, key=lambda a: a.timestamp)
        return {
            "worst_alert": worst.band,
            "latest_alert": {"id": latest.id, "message": latest.message, "band": latest.band,
                             "created_at": latest.timestamp},
        }

    def total_count(self) -> int:
        with self._lock:
            return self._total_count

    def get(self, alert_id: int) -> Optional[Alert]:
        with self._lock:
            for a in self._alerts:
                if a.id == alert_id:
                    return a
            return None
