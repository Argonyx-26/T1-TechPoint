"""Alert feed: cooldown-gated, capped, severity-ranked, with in-memory evidence JPEGs.
Thread-safe: the pipeline thread writes, API request threads read."""
import threading
import time
from dataclasses import dataclass
from datetime import datetime

from backend import config


@dataclass
class Alert:
    id: int
    rule: str
    type: str
    severity: str
    description: str
    created: float  # epoch seconds
    zone: str | None = None
    track_id: int | None = None
    evidence: bytes | None = None

    def to_dict(self) -> dict:
        dt = datetime.fromtimestamp(self.created).astimezone()
        return {
            "id": self.id,
            "type": self.type,
            "rule": self.rule,
            "severity": self.severity,
            "description": self.description,
            "message": self.description,
            "timestamp": dt.strftime("%H:%M:%S"),
            "created_at": dt.isoformat(timespec="seconds"),
            "zone": self.zone,
            "track_id": self.track_id,
            "has_evidence": self.evidence is not None,
        }


class AlertManager:
    def __init__(self, max_alerts: int = config.ALERT_MAX,
                 cooldown: float = config.ALERT_COOLDOWN_SECONDS):
        self.max_alerts = max_alerts
        self.cooldown = cooldown
        self._alerts: list[Alert] = []
        self._last_fired: dict[tuple, float] = {}
        self._next_id = 1
        self._lock = threading.Lock()

    def raise_alert(self, rule: str, description: str, zone: str | None = None,
                    track_id: int | None = None, key=None, evidence: bytes | None = None,
                    now: float | None = None) -> Alert | None:
        """Record an alert unless the same (rule, key) fired within the cooldown.

        `key` is the zone id for zone rules and the weapon class for weapons; without it
        the key falls back to (zone, track). Rules never share a key, so a busy zone
        can never suppress a weapon alert. Returns the new Alert, or None if suppressed.
        """
        if rule not in config.ALERT_RULES:
            raise ValueError(f"unknown rule {rule}")
        now = time.time() if now is None else now
        cooldown_key = (rule, key if key is not None else (zone, track_id))
        type_, severity = config.ALERT_RULES[rule]
        with self._lock:
            last = self._last_fired.get(cooldown_key)
            if last is not None and now - last < self.cooldown:
                return None
            self._last_fired[cooldown_key] = now
            alert = Alert(self._next_id, rule, type_, severity, description, now,
                          zone, track_id, evidence)
            self._next_id += 1
            self._alerts.append(alert)
            if len(self._alerts) > self.max_alerts:
                del self._alerts[: len(self._alerts) - self.max_alerts]
            return alert

    def list(self, limit: int = 50) -> list[dict]:
        with self._lock:
            ranked = sorted(self._alerts,
                            key=lambda a: (config.SEVERITY_RANK[a.severity], -a.created, -a.id))
            return [a.to_dict() for a in ranked[:max(limit, 0)]]

    def evidence(self, alert_id: int) -> bytes | None:
        with self._lock:
            for a in self._alerts:
                if a.id == alert_id:
                    return a.evidence
        return None

    def count(self) -> int:
        with self._lock:
            return len(self._alerts)

    def clear(self) -> None:
        with self._lock:
            self._alerts.clear()
            self._last_fired.clear()
