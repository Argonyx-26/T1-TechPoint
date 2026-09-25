"""Blind-spot tracking between cameras.

Every global subject's sightings (backend/search.py: time in, time out,
exit edge, best thumbnail per camera visit) form a movement log. When a
subject reappears on another camera the transit gap is compared with the
walking time of the link between the two cameras (Site Map):

  * gap <= 2x expected                      -> normal (logged)
  * gap >  max(2x expected, GAP_ALERT_S)    -> MEDIUM "spent 2m 10s in the blind
                                               spot between Main Gate and Lobby"
  * left a linked camera, not seen anywhere
    for MISSING_ALERT_S                     -> MEDIUM "unaccounted for since ..."

Either alert escalates to HIGH when the subject was involved in a HIGH or
CRITICAL alert in the last ESCALATE_WINDOW_S (e.g. a weapon).
evaluate() is pure over the current sightings, so it is unit-testable.
"""
import csv
import io
import threading
import time
from typing import Dict, List, Optional, Set, Tuple

from backend import config, features
from backend.audit import audit

BLIND_SPOT_GAP = "BLIND_SPOT_GAP"
SUBJECT_MISSING = "SUBJECT_MISSING"
MOVEMENT_RULES = {BLIND_SPOT_GAP, SUBJECT_MISSING}


def fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def sighting_key(s) -> Tuple:
    return (s.camera_id, s.track_id, round(s.first_seen, 3))


class MovementTracker:
    def __init__(self, search_index, camera_manager, alert_manager, start: bool = True):
        self.search = search_index
        self.cameras = camera_manager
        self.alerts = alert_manager
        self._lock = threading.Lock()
        self._alerted_gaps: Set[Tuple] = set()
        self._alerted_missing: Set[Tuple] = set()
        self._logged_transits: Set[Tuple] = set()
        self._alert_subjects: Dict[int, Optional[str]] = {}   # alert id -> global id (cache)
        self._running = start
        if start:
            threading.Thread(target=self._loop, name="movements", daemon=True).start()

    def _loop(self):
        while self._running:
            try:
                if features.enabled("gap_tracking"):
                    self.evaluate(time.time())
            except Exception as exc:
                print(f"[movements] {exc}")
            time.sleep(1.0)

    # -- helpers ----------------------------------------------------------------------
    def _people(self) -> Dict[str, list]:
        by_gid: Dict[str, list] = {}
        for s in self.search.all_sightings():
            if s.cls == "person" and s.global_id:
                by_gid.setdefault(s.global_id, []).append(s)
        for rows in by_gid.values():
            rows.sort(key=lambda s: s.first_seen)
        return by_gid

    def _place(self, cam_id: str) -> str:
        cam = self.cameras.get(cam_id)
        return cam.place if cam else cam_id

    def _linked(self, cam_id: str) -> bool:
        return any(cam_id in (l["a"], l["b"]) for l in self.cameras.links)

    def _involved_in_serious_alert(self, gid: str, now: float) -> bool:
        for alert in self.alerts.ranked(500):
            if (alert.band not in ("HIGH", "CRITICAL") or alert.rule in MOVEMENT_RULES
                    or now - alert.timestamp > config.ESCALATE_WINDOW_S):
                continue
            if alert.id not in self._alert_subjects:
                self._alert_subjects[alert.id] = self.search.subject_for_alert(alert)
            if self._alert_subjects[alert.id] == gid:
                return True
        return False

    # -- the rules ------------------------------------------------------------------------
    def evaluate(self, now: float) -> List:
        """Raise any new blind-spot / missing alerts. Returns the alerts created."""
        created = []
        for gid, rows in self._people().items():
            for a, b in zip(rows, rows[1:]):
                if not a.closed or a.camera_id == b.camera_id:
                    continue
                expected = self.cameras.expected_seconds(a.camera_id, b.camera_id)
                if expected is None:
                    continue
                gap = b.first_seen - a.last_seen
                limit = max(2 * expected, config.GAP_ALERT_S)
                key = (gid, sighting_key(a), sighting_key(b))
                if gap <= limit and key not in self._logged_transits:
                    self._logged_transits.add(key)
                    audit("SUBJECT_TRANSIT", f"Subject #{gid} {self._place(a.camera_id)} -> {self._place(b.camera_id)} "
                                             f"in {fmt_duration(gap)} (expected ~{fmt_duration(expected)})",
                          camera_id=b.camera_id)
                if gap > limit and key not in self._alerted_gaps:
                    self._alerted_gaps.add(key)
                    msg = (f"Subject #{gid} spent {fmt_duration(gap)} in the blind spot between "
                           f"{self._place(a.camera_id)} and {self._place(b.camera_id)} (expected ~{fmt_duration(expected)})")
                    created.append(self._raise(BLIND_SPOT_GAP, msg, gid, b, now))
            last = rows[-1]
            key = (gid, sighting_key(last))
            if (last.closed and self._linked(last.camera_id)
                    and now - last.last_seen > config.MISSING_ALERT_S and key not in self._alerted_missing):
                self._alerted_missing.add(key)
                left_at = time.strftime("%H:%M", time.localtime(last.last_seen))
                msg = f"Subject #{gid} unaccounted for since leaving {self._place(last.camera_id)} at {left_at}"
                created.append(self._raise(SUBJECT_MISSING, msg, gid, last, now))
        return [a for a in created if a is not None]

    def _raise(self, rule: str, message: str, gid: str, sighting, now: float):
        escalate = self._involved_in_serious_alert(gid, now)
        if escalate:
            message += " - involved in a serious alert in the last 15 min"
        camera = self.cameras.get(sighting.camera_id)
        return self.alerts.ingest_event(
            rule, message, now, camera=camera, band="HIGH" if escalate else "MEDIUM",
            bbox=sighting.bbox, frame=camera.latest_frame() if camera and hasattr(camera, "latest_frame") else None,
            dedupe_key=(rule, gid, sighting_key(sighting)),
        )

    # -- movement log ---------------------------------------------------------------------------
    def log(self, minutes: float = 30, now: Optional[float] = None) -> List[dict]:
        now = now or time.time()
        since = now - minutes * 60
        out = []
        for gid, rows in self._people().items():
            rows = [s for s in rows if s.last_seen >= since]
            if not rows:
                continue
            segments = []
            for i, s in enumerate(rows):
                nxt = rows[i + 1] if i + 1 < len(rows) else None
                gap = (nxt.first_seen - s.last_seen) if nxt and s.closed else None
                expected = self.cameras.expected_seconds(s.camera_id, nxt.camera_id) if nxt and nxt.camera_id != s.camera_id else None
                segments.append({
                    "camera_id": s.camera_id,
                    "camera": self.search._camera_label(s.camera_id),
                    "time_in": s.first_seen,
                    "time_out": s.last_seen if s.closed else None,
                    "dwell_s": round(s.last_seen - s.first_seen, 1),
                    "exit_edge": s.exit_edge,
                    "gap_to_next_s": round(gap, 1) if gap is not None else None,
                    "expected_s": expected,
                    "status": self._status(s, nxt, gap, expected, now),
                    "thumb_url": f"/api/search/thumb/{s.best_entry}" if s.best_entry else None,
                })
            thumb = next((seg["thumb_url"] for seg in reversed(segments) if seg["thumb_url"]), None)
            out.append({"global_id": gid, "thumb_url": thumb, "first_seen": rows[0].first_seen,
                        "last_seen": rows[-1].last_seen, "segments": segments})
        out.sort(key=lambda r: -r["last_seen"])
        return out

    def _status(self, s, nxt, gap, expected, now) -> str:
        if not s.closed:
            return "in_view"
        if nxt is None:
            if self._linked(s.camera_id) and now - s.last_seen > config.MISSING_ALERT_S:
                return "missing"
            return "left"
        if nxt.camera_id == s.camera_id:
            return "returned"
        if expected is None:
            return "unlinked"
        if gap <= 2 * expected:
            return "normal"
        if gap > max(2 * expected, config.GAP_ALERT_S):
            return "long_gap"
        return "slow"

    def export_csv(self, minutes: float = 30, now: Optional[float] = None) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["subject", "camera", "camera_name", "time_in", "time_out", "dwell_s", "exit_edge",
                    "gap_to_next_s", "expected_s", "status"])
        stamp = lambda t: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) if t else ""
        for subject in self.log(minutes, now=now):
            for seg in subject["segments"]:
                w.writerow([subject["global_id"], seg["camera"]["code"], seg["camera"]["name"], stamp(seg["time_in"]),
                            stamp(seg["time_out"]), seg["dwell_s"], seg["exit_edge"] or "", seg["gap_to_next_s"] if seg["gap_to_next_s"] is not None else "",
                            seg["expected_s"] if seg["expected_s"] is not None else "", seg["status"]])
        return buf.getvalue()
