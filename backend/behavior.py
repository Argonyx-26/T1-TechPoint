"""Behaviour detection from body pose: fighting, throwing, distress.

yolov8n-pose runs per camera at up to POSE_FPS, only while people are in
view; pose people are matched to the camera's existing tracks by IoU. The
rules below are pure functions over keypoint / track history (unit-tested
with synthetic keypoints), and every alert is sustained-gated like the
weapon 5-of-8 filter: never a single frame, and always worded "possible".

Body language only: no face or emotion analysis.
"""
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend import config, features
from backend.audit import audit
from backend.geometry import point_in_polygon

FIGHT = "FIGHT"
THROW = "OBJECT_THROWN"
HANDS_RAISED = "HANDS_RAISED"
PERSON_DOWN = "PERSON_DOWN"
CROWD_PANIC = "CROWD_PANIC"
BEHAVIOUR_RULES = {FIGHT, THROW, HANDS_RAISED, PERSON_DOWN, CROWD_PANIC}

# COCO keypoints
NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_ANK, R_ANK = 0, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16
ARMS = ((L_SH, L_EL, L_WR), (R_SH, R_EL, R_WR))


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------
@dataclass
class PoseObs:
    ts: float
    kps: np.ndarray                     # (17, 3): x, y, confidence
    bbox: Tuple[float, float, float, float]

    def kp(self, i) -> Optional[Tuple[float, float]]:
        x, y, c = self.kps[i]
        return (float(x), float(y)) if c >= config.POSE_KP_CONF else None

    @property
    def height(self) -> float:
        return max(1.0, self.bbox[3] - self.bbox[1])

    @property
    def width(self) -> float:
        return max(1.0, self.bbox[2] - self.bbox[0])

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _mid(a, b):
    if a is None or b is None:
        return a or b
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


class SustainGate:
    """Fraction of recent frames that were positive, over a time window."""

    def __init__(self):
        self.samples: Deque[Tuple[float, bool]] = deque()

    def add(self, ts: float, positive: bool, window_s: float):
        self.samples.append((ts, positive))
        while self.samples and self.samples[0][0] < ts - window_s:
            self.samples.popleft()

    def sustained(self, window_s: float, min_ratio: float) -> bool:
        if len(self.samples) < 3:
            return False
        span = self.samples[-1][0] - self.samples[0][0]
        if span < window_s * 0.8:
            return False
        return sum(p for _, p in self.samples) / len(self.samples) >= min_ratio

    def held_for(self, ts: float) -> float:
        """How long the condition has been continuously true up to ts."""
        start = None
        for t, p in reversed(self.samples):
            if not p:
                break
            start = t
        return ts - start if start is not None else 0.0


# ---------------------------------------------------------------------------
# Pure rules
# ---------------------------------------------------------------------------
def hands_raised(obs: PoseObs) -> bool:
    """Both wrists above the nose (image y grows downward)."""
    nose, lw, rw = obs.kp(NOSE), obs.kp(L_WR), obs.kp(R_WR)
    if lw is None or rw is None:
        return False
    head_y = nose[1] if nose else min((p[1] for p in (obs.kp(L_SH), obs.kp(R_SH)) if p), default=None)
    if head_y is None:
        return False
    if nose is None:
        head_y -= 0.12 * obs.height  # shoulders visible, nose not: approximate the nose
    return lw[1] < head_y and rw[1] < head_y


def is_upright(obs: PoseObs) -> bool:
    return obs.height / obs.width >= config.UPRIGHT_ASPECT


def is_down(obs: PoseObs) -> bool:
    """Horizontal body: a wide box, and one of: the box is very wide, the
    torso lies more than DOWN_TORSO_DEG from vertical, or the hips are near
    ankle height. (Hips-vs-ankles alone missed people lying at an angle to
    the camera with bent legs: UR Fall fall-06.)"""
    aspect = obs.width / obs.height
    if aspect <= config.DOWN_ASPECT:
        return False
    if aspect >= config.DOWN_STRONG_ASPECT:
        return True
    shoulder = _mid(obs.kp(L_SH), obs.kp(R_SH))
    hip = _mid(obs.kp(L_HIP), obs.kp(R_HIP))
    if shoulder is not None and hip is not None:
        dx, dy = abs(shoulder[0] - hip[0]), abs(shoulder[1] - hip[1])
        if math.degrees(math.atan2(dx, dy)) >= config.DOWN_TORSO_DEG:
            return True
    ankle = _mid(obs.kp(L_ANK), obs.kp(R_ANK))
    if hip is None or ankle is None:
        return True  # box alone is horizontal; lying legs are often occluded
    return abs(hip[1] - ankle[1]) <= config.DOWN_HIP_ANKLE_FRAC * obs.width


def box_upright(box) -> bool:
    return (box[3] - box[1]) / max(1.0, box[2] - box[0]) >= config.UPRIGHT_ASPECT


def box_down(box) -> bool:
    """Tracker box alone: the pose model often finds no skeleton on someone
    lying on the floor, but the person detector still boxes them."""
    return (box[2] - box[0]) / max(1.0, box[3] - box[1]) > config.DOWN_ASPECT


def _relative_speed(prev: PoseObs, cur: PoseObs, i: int) -> Optional[float]:
    """Speed of keypoint i RELATIVE to the body (box centre), in person
    heights per second: walking across the frame is not limb motion."""
    dt = cur.ts - prev.ts
    a, b = prev.kp(i), cur.kp(i)
    if dt <= 0 or not a or not b:
        return None
    (pcx, pcy), (ccx, ccy) = prev.center, cur.center
    return _dist((a[0] - pcx, a[1] - pcy), (b[0] - ccx, b[1] - ccy)) / dt / cur.height


def wrist_speed(prev: PoseObs, cur: PoseObs) -> float:
    """Fastest wrist/elbow speed relative to the body, person heights per second."""
    speeds = [s for s in (_relative_speed(prev, cur, i) for i in (L_WR, R_WR, L_EL, R_EL)) if s is not None]
    return max(speeds, default=0.0)


UPPER_BODY = tuple(range(0, 11))   # head, shoulders, elbows, wrists


def motion_energy(prev: PoseObs, cur: PoseObs) -> float:
    """Mean upper-body keypoint speed relative to the body, person heights
    per second (legs excluded: walking is not agitation)."""
    speeds = [s for s in (_relative_speed(prev, cur, i) for i in UPPER_BODY) if s is not None]
    return float(np.mean(speeds)) if speeds else 0.0


def arm_extended_toward(obs: PoseObs, target: Tuple[float, float]) -> bool:
    """An arm is stretched out (wrist beyond the elbow) and pointing at target."""
    for sh_i, el_i, wr_i in ARMS:
        sh, el, wr = obs.kp(sh_i), obs.kp(el_i), obs.kp(wr_i)
        if not (sh and el and wr):
            continue
        reach = _dist(sh, wr)
        if reach < 0.8 * (_dist(sh, el) + _dist(el, wr)) or reach < 0.25 * obs.height:
            continue
        arm = (wr[0] - sh[0], wr[1] - sh[1])
        to_target = (target[0] - sh[0], target[1] - sh[1])
        norm = math.hypot(*arm) * math.hypot(*to_target)
        if norm and (arm[0] * to_target[0] + arm[1] * to_target[1]) / norm >= config.FIGHT_EXTEND_COS:
            return True
    return False


def boxes_close(a: PoseObs, b: PoseObs) -> bool:
    """Close in the scene, not just overlapping on screen: on a fixed camera
    two people side by side have their feet at a similar image height and a
    similar size; someone metres behind overlaps but stands higher up."""
    mean_h = (a.height + b.height) / 2
    if abs(a.bbox[3] - b.bbox[3]) > config.FIGHT_DEPTH_FRAC * mean_h:
        return False
    if not 1 / config.FIGHT_SIZE_RATIO <= a.height / b.height <= config.FIGHT_SIZE_RATIO:
        return False
    gap_x = max(0.0, max(a.bbox[0], b.bbox[0]) - min(a.bbox[2], b.bbox[2]))
    gap_y = max(0.0, max(a.bbox[1], b.bbox[1]) - min(a.bbox[3], b.bbox[3]))
    return math.hypot(gap_x, gap_y) < config.FIGHT_GAP_FRAC * (a.width + b.width) / 2


def fight_frame(prev_a: PoseObs, a: PoseObs, prev_b: PoseObs, b: PoseObs) -> bool:
    """One frame of a possible fight between a and b."""
    if not boxes_close(a, b):
        return False
    if min(motion_energy(prev_a, a), motion_energy(prev_b, b)) < config.FIGHT_ENERGY:
        return False
    striking = [
        wrist_speed(prev_a, a) >= config.FIGHT_WRIST_SPEED and arm_extended_toward(a, b.center),
        wrist_speed(prev_b, b) >= config.FIGHT_WRIST_SPEED and arm_extended_toward(b, a.center),
    ]
    return any(striking)


def panic_origin(movers: List[Tuple[Tuple[float, float], Tuple[float, float]]]) -> Optional[Tuple[float, float]]:
    """Least-squares point that the movers' velocity lines radiate from.
    movers: [(position, velocity)]."""
    if len(movers) < 2:
        return None
    a = np.zeros((2, 2))
    rhs = np.zeros(2)
    for (px, py), (vx, vy) in movers:
        n = math.hypot(vx, vy)
        if n == 0:
            continue
        d = np.array([vx / n, vy / n])
        proj = np.eye(2) - np.outer(d, d)
        a += proj
        rhs += proj @ np.array([px, py])
    if abs(np.linalg.det(a)) < 1e-6:
        return None
    x, y = np.linalg.solve(a, rhs)
    return float(x), float(y)


def panic_frame(movers: List[Tuple[Tuple[float, float], Tuple[float, float]]]) -> Tuple[bool, Optional[Tuple[float, float]]]:
    """>= PANIC_MIN_PEOPLE sudden movers all heading away from a common point."""
    if len(movers) < config.PANIC_MIN_PEOPLE:
        return False, None
    origin = panic_origin(movers)
    if origin is None:
        return False, None
    away = 0
    for (px, py), (vx, vy) in movers:
        out = (px - origin[0], py - origin[1])
        norm = math.hypot(*out) * math.hypot(vx, vy)
        if norm and (out[0] * vx + out[1] * vy) / norm >= config.PANIC_AWAY_COS:
            away += 1
    return away >= config.PANIC_MIN_PEOPLE, origin


def throw_from_track(obj_hist, wrists_hist, diag: float) -> Optional[int]:
    """A tracked object that was near a person's wrist, then moved away fast
    for THROW_MIN_FRAMES observations with growing distance. Returns the
    thrower's track id.

    obj_hist: [(ts, (x, y))] for the object; wrists_hist: {person_id: [(ts,
    [wrist points], person_height)]}."""
    if len(obj_hist) < config.THROW_MIN_FRAMES + 1:
        return None
    tail = obj_hist[-(config.THROW_MIN_FRAMES + 1):]
    for pid, hist in wrists_hist.items():
        near_t = None
        for ts, wrists, h in hist:
            obj = _nearest_in_time(obj_hist, ts)
            if obj and wrists and min(_dist(obj, w) for w in wrists) <= config.THROW_NEAR_WRIST * h:
                near_t = ts
        if near_t is None or tail[0][0] < near_t - 0.5:
            continue
        # distance from the thrower's last wrist position must grow every step
        _, wrists, _ = hist[-1]
        if not wrists:
            continue
        anchor = min(wrists, key=lambda w: _dist(w, tail[0][1]))
        dists = [_dist(p, anchor) for _, p in tail]
        if not all(d2 > d1 for d1, d2 in zip(dists, dists[1:])):
            continue
        speeds = [_dist(p2, p1) / max(1e-3, t2 - t1) / diag for (t1, p1), (t2, p2) in zip(tail, tail[1:])]
        if min(speeds) >= config.THROW_SPEED:
            return pid
    return None


def _nearest_in_time(hist, ts, tol=0.3):
    best = min(hist, key=lambda h: abs(h[0] - ts), default=None)
    return best[1] if best and abs(best[0] - ts) <= tol else None


def throw_target(start, end, people: Dict[int, Tuple[float, float]], thrower: int,
                 restricted: List[List[Tuple[float, float]]]) -> Optional[str]:
    """What the throw is heading at: another person, or a restricted zone."""
    for poly in restricted:
        if point_in_polygon(end, poly):
            return "into a restricted zone"
    v = (end[0] - start[0], end[1] - start[1])
    for pid, pos in people.items():
        if pid == thrower:
            continue
        to = (pos[0] - start[0], pos[1] - start[1])
        norm = math.hypot(*v) * math.hypot(*to)
        if norm and (v[0] * to[0] + v[1] * to[1]) / norm >= 0.85:
            return f"toward #{pid}"
    return None


def moving_blob_near(prev_gray, gray, wrist, radius) -> Optional[Tuple[float, float]]:
    """Centroid of a small moving blob (frame difference) within radius of
    the wrist: an object we have no class for, leaving the hand."""
    if prev_gray is None or wrist is None:
        return None
    h, w = gray.shape[:2]
    x1, y1 = max(0, int(wrist[0] - radius)), max(0, int(wrist[1] - radius))
    x2, y2 = min(w, int(wrist[0] + radius)), min(h, int(wrist[1] + radius))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    diff = cv2.absdiff(gray[y1:y2, x1:x2], prev_gray[y1:y2, x1:x2])
    _, mask = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = [c for c in contours if 20 <= cv2.contourArea(c) <= 0.02 * (radius * 2) ** 2 * 4]
    if not blobs:
        return None
    far = max(blobs, key=lambda c: _dist(_centroid(c, x1, y1), wrist))
    return _centroid(far, x1, y1)


def _centroid(contour, ox, oy):
    m = cv2.moments(contour)
    if not m["m00"]:
        x, y, w, h = cv2.boundingRect(contour)
        return (ox + x + w / 2, oy + y + h / 2)
    return (ox + m["m10"] / m["m00"], oy + m["m01"] / m["m00"])


# ---------------------------------------------------------------------------
# Per-camera state
# ---------------------------------------------------------------------------
@dataclass
class BehaviourEvent:
    rule: str
    message: str
    track_ids: List[int]
    bbox: Optional[Tuple[float, float, float, float]]
    band: Optional[str] = None
    marker: Optional[Tuple[float, float]] = None
    label: str = ""


@dataclass
class CameraBehaviour:
    """All behaviour state for one camera. update() is pure over its inputs."""
    poses: Dict[int, Deque[PoseObs]] = field(default_factory=dict)
    gates: Dict[Tuple, SustainGate] = field(default_factory=dict)
    fired: Dict[Tuple, float] = field(default_factory=dict)
    upright_seen: Dict[int, float] = field(default_factory=dict)
    # (ts, box) of every upright sighting: a fall often breaks the track (the
    # detector loses the person mid-fall and re-finds them under a new id),
    # so "was upright" is also judged by place, not only by track id.
    upright_spots: Deque[Tuple[float, Tuple[float, float, float, float], int]] = field(default_factory=deque)
    objects: Dict[int, Deque[Tuple[float, Tuple[float, float], str]]] = field(default_factory=dict)
    object_last_seen: Dict[int, float] = field(default_factory=dict)
    speeds: Dict[int, Deque[Tuple[float, float]]] = field(default_factory=dict)
    panic_gate: SustainGate = field(default_factory=SustainGate)
    prev_gray: Optional[np.ndarray] = None
    crowd_hist: Deque[Tuple[float, int, float]] = field(default_factory=deque)  # (ts, people, motion)
    prev_small: Optional[np.ndarray] = None
    blob_trails: Dict[int, List[Tuple[float, Tuple[float, float]]]] = field(default_factory=dict)
    overlays: List[Tuple[float, BehaviourEvent]] = field(default_factory=list)

    def gate(self, *key) -> SustainGate:
        return self.gates.setdefault(key, SustainGate())

    def _once(self, key, ts, episode_gap=config.BEHAVIOUR_REARM_S) -> bool:
        """Fire once per episode: re-arms after the condition was absent a while."""
        last = self.fired.get(key)
        self.fired[key] = ts
        return last is None or ts - last > episode_gap

    def update(self, ts: float, people: Dict[int, PoseObs], tracks: Dict, frame_size: Tuple[int, int],
               restricted: List[List[Tuple[float, float]]], gray=None, weapon_recent: bool = False,
               enabled=lambda name: True) -> List[BehaviourEvent]:
        events: List[BehaviourEvent] = []
        w, h = frame_size
        diag = math.hypot(w, h) or 1.0
        for tid, obs in people.items():
            hist = self.poses.setdefault(tid, deque(maxlen=64))
            hist.append(obs)
            if is_upright(obs):
                self.upright_seen[tid] = ts
                self.upright_spots.append((ts, obs.bbox, tid))
        for tid, track in tracks.items():
            if tid not in people and getattr(track, "cls_name", "") == "person" and box_upright(track.bbox):
                self.upright_seen[tid] = ts
                self.upright_spots.append((ts, tuple(track.bbox), tid))
        while self.upright_spots and self.upright_spots[0][0] < ts - config.UPRIGHT_LOOKBACK_S:
            self.upright_spots.popleft()

        # Low-confidence people (DOWN_LOW_CONF..threshold) exist for person
        # down only; every other rule sees the same people as the rest of the system.
        thr = config.DETECTOR_CONF_THRESHOLD
        sure_tracks = {tid: t for tid, t in tracks.items() if getattr(t, "conf", 1.0) >= thr}
        sure = {tid: o for tid, o in people.items() if tid not in tracks or tid in sure_tracks}
        if enabled("distress"):
            events += self._dispersal(ts, len([t for t in sure_tracks.values() if t.cls_name == "person"]), gray)
            events += self._hands_raised(ts, sure, weapon_recent)
            events += self._person_down(ts, people, tracks, frame_w=w, now_upright={
                tid for tid, t in tracks.items() if getattr(t, "cls_name", "") == "person" and box_upright(t.bbox)})
            events += self._panic(ts, sure_tracks, diag)
        if enabled("fighting"):
            events += self._fights(ts, sure)
        if enabled("throwing"):
            events += self._throws(ts, sure, tracks, diag, restricted, gray)
        for tid in [t for t in self.poses if t not in people and ts - self.poses[t][-1].ts > 5]:
            self.poses.pop(tid, None)
        self.overlays = [(t, e) for t, e in self.overlays if ts - t < 4] + [(ts, e) for e in events]
        return events

    # -- distress ---------------------------------------------------------------------
    def _hands_raised(self, ts, people, weapon_recent):
        out = []
        for tid, obs in people.items():
            hands = self.gate(HANDS_RAISED, tid)
            hands.add(ts, hands_raised(obs), config.HANDS_RAISED_S * 2)
            if hands.held_for(ts) >= config.HANDS_RAISED_S and self._once((HANDS_RAISED, tid), ts):
                out.append(BehaviourEvent(
                    HANDS_RAISED,
                    f"Hands raised - possible hold-up (#{tid})" + (" with a weapon on this camera" if weapon_recent else ""),
                    [tid], obs.bbox, band="CRITICAL" if weapon_recent else None, label="HANDS RAISED"))
        return out

    def _was_upright_here(self, tid, box, ts, now_upright=frozenset()) -> bool:
        """Upright on this track, or someone stood upright right where this
        person now lies (feet within one body height of this box) AND is no
        longer standing there: in a fall the standing person turns into the
        lying one. In a crowd the people standing next to a false box are
        still standing, so they do not count."""
        if ts - self.upright_seen.get(tid, -1e9) <= config.UPRIGHT_LOOKBACK_S:
            return True
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        for t, ub, utid in self.upright_spots:
            if utid in now_upright or ts - t > config.UPRIGHT_LOOKBACK_S:
                continue
            height = ub[3] - ub[1]
            fx, fy = (ub[0] + ub[2]) / 2, ub[3]
            long_enough = box[2] - box[0] >= config.DOWN_MIN_LENGTH * height   # not a head at the frame edge
            if long_enough and math.hypot(cx - fx, cy - fy) <= config.DOWN_NEAR_UPRIGHT * height:
                return True
        return False

    def _person_down(self, ts, people, tracks, now_upright=frozenset(), frame_w=None):
        """Every tracked person, with or without a skeleton: pose decides
        when there is one, the box shape when there is not."""
        bodies = {tid: (obs.bbox, obs) for tid, obs in people.items()}
        for tid, track in tracks.items():
            if tid not in bodies and getattr(track, "cls_name", "") == "person" and tid >= 0:
                bodies[tid] = (tuple(track.bbox), None)
        out = []
        for tid, (box, obs) in bodies.items():
            if obs is not None:
                lying = is_down(obs)
            else:   # no skeleton: trust the box only when it is body-sized, not a fragment at the frame edge
                lying = box_down(box) and (not frame_w or box[2] - box[0] >= config.DOWN_BOX_MIN_WIDTH * frame_w)
            gate = self.gate(PERSON_DOWN, tid)
            gate.add(ts, lying and self._was_upright_here(tid, box, ts, now_upright), config.PERSON_DOWN_S * 2)
            if gate.held_for(ts) >= config.PERSON_DOWN_S and self._once((PERSON_DOWN, tid), ts):
                out.append(BehaviourEvent(PERSON_DOWN, f"Person down - possible medical emergency (#{tid})",
                                          [tid], box, label="PERSON DOWN"))
        return out

    def _dispersal(self, ts, count, gray):
        """Crowd dispersing: the count collapses while the scene's motion
        spikes (see PANIC_DISPERSE_*)."""
        motion = 0.0
        if gray is not None:
            small = cv2.resize(gray, (160, max(1, int(160 * gray.shape[0] / gray.shape[1]))))
            if self.prev_small is not None and self.prev_small.shape == small.shape:
                motion = float(np.mean(cv2.absdiff(small, self.prev_small)))
            self.prev_small = small
        hist = self.crowd_hist
        hist.append((ts, count, motion))
        w = config.PANIC_DISPERSE_WINDOW_S
        while hist and hist[0][0] < ts - 3 * w:
            hist.popleft()
        before = [c for t, c, _ in hist if ts - 2 * w <= t < ts - w]
        recent = [c for t, c, _ in hist if t >= ts - 1.0]
        calm = [m for t, _, m in hist if t < ts - w]
        moving = [m for t, _, m in hist if t >= ts - w]
        if len(before) < 3 or not recent or len(calm) < 3 or not moving:
            return []
        base = float(np.median(before))
        collapsed = base >= config.PANIC_DISPERSE_MIN and max(recent) <= config.PANIC_DISPERSE_FRAC * base
        spiked = max(moving) >= config.PANIC_MOTION_RATIO * max(float(np.median(calm)), 1.0)
        if collapsed and spiked and self._once((CROWD_PANIC,), ts):
            return [BehaviourEvent(CROWD_PANIC, f"Possible panic - crowd dispersing ({base:.0f} -> {max(recent)} people "
                                                f"in {w:.0f}s, sudden motion)", [], None, label="PANIC")]
        return []

    def _panic(self, ts, tracks, diag):
        movers = []
        for tid, track in tracks.items():
            if track.cls_name != "person" or len(track.history) < 3:
                continue
            (t0, p0), (t1, p1) = track.history[-3], track.history[-1]
            if t1 - t0 <= 0:
                continue
            v = ((p1[0] - p0[0]) / (t1 - t0), (p1[1] - p0[1]) / (t1 - t0))
            speed = math.hypot(*v) / diag
            hist = self.speeds.setdefault(tid, deque())
            hist.append((ts, speed))
            while hist and hist[0][0] < ts - config.PANIC_BASELINE_S:
                hist.popleft()
            older = [s for t, s in hist if t < ts - 1.5]
            baseline = float(np.median(older)) if len(older) >= 3 else None
            if (baseline is not None and speed >= config.PANIC_MIN_SPEED
                    and speed > config.PANIC_SPEED_RATIO * max(baseline, 0.01)):
                movers.append((p1, v))
        positive, origin = panic_frame(movers)
        self.panic_gate.add(ts, positive, config.PANIC_SUSTAIN_S * 2)
        if positive and self.panic_gate.held_for(ts) >= config.PANIC_SUSTAIN_S and self._once((CROWD_PANIC,), ts):
            return [BehaviourEvent(CROWD_PANIC, f"Possible panic - crowd dispersing ({len(movers)} people running)",
                                   [], None, marker=origin, label="PANIC ORIGIN?")]
        return []

    # -- fighting -------------------------------------------------------------------------
    def _fights(self, ts, people):
        out = []
        ids = sorted(people)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                ha, hb = self.poses.get(a), self.poses.get(b)
                if not ha or not hb or len(ha) < 2 or len(hb) < 2:
                    continue
                gate = self.gate(FIGHT, a, b)
                gate.add(ts, fight_frame(ha[-2], ha[-1], hb[-2], hb[-1]), config.FIGHT_WINDOW_S)
                if gate.sustained(config.FIGHT_WINDOW_S, config.FIGHT_MIN_RATIO) and self._once((FIGHT, a, b), ts):
                    x1 = min(ha[-1].bbox[0], hb[-1].bbox[0]); y1 = min(ha[-1].bbox[1], hb[-1].bbox[1])
                    x2 = max(ha[-1].bbox[2], hb[-1].bbox[2]); y2 = max(ha[-1].bbox[3], hb[-1].bbox[3])
                    out.append(BehaviourEvent(FIGHT, f"Possible fight between #{a} and #{b}", [a, b],
                                              (x1, y1, x2, y2), label="POSSIBLE FIGHT"))
        return out

    # -- throwing ---------------------------------------------------------------------------
    def _throws(self, ts, people, tracks, diag, restricted, gray):
        out = []
        wrists = {tid: [(o.ts, [p for p in (o.kp(L_WR), o.kp(R_WR)) if p], o.height) for o in hist]
                  for tid, hist in self.poses.items() if tid in people}
        live_objects = set()
        for tid, track in tracks.items():
            if track.cls_name not in config.THROW_CLASSES:
                continue
            live_objects.add(tid)
            hist = self.objects.setdefault(tid, deque(maxlen=20))
            hist.append((ts, track.centroid, track.cls_name))
            self.object_last_seen[tid] = ts
            thrower = throw_from_track([(t, p) for t, p, _ in hist], wrists, diag)
            if thrower is not None and self._once((THROW, tid), ts):
                people_pos = {pid: o.center for pid, o in people.items()}
                target = throw_target(hist[-4][1], hist[-1][1], people_pos, thrower, restricted)
                out.append(self._throw_event(track.cls_name, thrower, target, track.bbox))

        # wrist speed spike, then the object that was in that hand is gone
        for pid, hist in self.poses.items():
            if pid not in people or len(hist) < 2 or wrist_speed(hist[-2], hist[-1]) < config.WRIST_SPIKE_SPEED:
                continue
            for oid, ohist in self.objects.items():
                if oid in live_objects or not ohist or ts - self.object_last_seen.get(oid, 0) > 0.8:
                    continue
                t_last, pos, cls = ohist[-1]
                prev_wrists = [p for p in (hist[-2].kp(L_WR), hist[-2].kp(R_WR)) if p]
                if prev_wrists and min(_dist(pos, w) for w in prev_wrists) <= config.THROW_NEAR_WRIST * hist[-1].height:
                    if self._once((THROW, oid), ts):
                        out.append(self._throw_event(cls, pid, None, hist[-1].bbox))

        # an unknown small blob leaving a fast-moving hand
        if gray is not None:
            for pid, hist in self.poses.items():
                if pid not in people or len(hist) < 2:
                    continue
                if wrist_speed(hist[-2], hist[-1]) < config.WRIST_SPIKE_SPEED and pid not in self.blob_trails:
                    continue
                wrist = max((p for p in (hist[-1].kp(L_WR), hist[-1].kp(R_WR)) if p), default=None,
                            key=lambda p: -p[1])
                blob = moving_blob_near(self.prev_gray, gray, wrist, 0.6 * hist[-1].height)
                trail = self.blob_trails.setdefault(pid, [])
                if blob is None:
                    if len(trail) > 1 or ts - (trail[-1][0] if trail else ts) > 0.6:
                        self.blob_trails.pop(pid, None)
                    continue
                trail.append((ts, blob))
                if len(trail) >= config.THROW_MIN_FRAMES + 1:
                    thrower = throw_from_track(trail, {pid: [(t, [w for w in (o.kp(L_WR), o.kp(R_WR)) if w], o.height)
                                                             for t, o in ((o.ts, o) for o in hist)]}, diag)
                    if thrower is not None and self._once((THROW, "blob", pid), ts):
                        out.append(self._throw_event("object", pid, None, hist[-1].bbox))
                    self.blob_trails.pop(pid, None)
            self.prev_gray = gray
        for oid in [o for o, t in self.object_last_seen.items() if ts - t > 3]:
            self.objects.pop(oid, None)
            self.object_last_seen.pop(oid, None)
        return out

    @staticmethod
    def _throw_event(cls, thrower, target, bbox):
        msg = f"Object thrown: {cls}, by #{thrower}" + (f" {target}" if target else "")
        return BehaviourEvent(THROW, msg, [thrower], bbox, band="HIGH" if target else "MEDIUM", label="THROW")


# ---------------------------------------------------------------------------
# Engine (camera hook)
# ---------------------------------------------------------------------------
def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class BehaviourEngine:
    def __init__(self, shared_lock: threading.Lock, alert_manager):
        self._lock = shared_lock          # the GPU lock shared with detection
        self.alerts = alert_manager
        self._model = None
        self.cameras: Dict[str, CameraBehaviour] = {}
        self._last_pose: Dict[str, float] = {}

    def _ensure(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(config.POSE_MODEL)
            audit("MODEL_LOADED", f"behaviour: {config.POSE_MODEL}")

    def on_frame(self, camera, frame, annotated):
        if not features.enabled("pose"):
            return
        state = self.cameras.setdefault(camera.id, CameraBehaviour())
        tracks = getattr(camera.pipeline, "last_tracks", {}) or {}
        people_tracks = {tid: t for tid, t in tracks.items() if t.cls_name == "person" and tid >= 0}
        now = time.time()
        if people_tracks and now - self._last_pose.get(camera.id, 0) >= 1.0 / config.POSE_FPS:
            self._last_pose[camera.id] = now
            people = self._pose(frame, people_tracks, now)
            h, w = frame.shape[:2]
            restricted = [z.to_pixels(w, h).polygon for z in camera.pipeline.zone_store.list() if z.restricted]
            gray = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if features.enabled("throwing") or features.enabled("distress") else None)
            events = state.update(now, people, tracks, (w, h), restricted, gray=gray,
                                  weapon_recent=self._weapon_recent(camera.id, now), enabled=features.enabled)
            for event in events:
                self.alerts.ingest_event(
                    event.rule, event.message, now, camera=camera, band=event.band, bbox=event.bbox,
                    frame=annotated, track_ids=event.track_ids,
                    dedupe_key=(camera.id, event.rule, tuple(event.track_ids)),
                )
        self._draw(state, annotated, now)

    def _pose(self, frame, people_tracks, now) -> Dict[int, PoseObs]:
        with self._lock:
            self._ensure()
            result = self._model.predict(frame, conf=config.POSE_CONF, device=config.DEVICE,
                                         half=config.USE_HALF_PRECISION, verbose=False)[0]
        if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
            return {}
        boxes = result.boxes.xyxy.cpu().numpy()
        kps = result.keypoints.data.cpu().numpy()   # (n, 17, 3)
        matched: Dict[int, PoseObs] = {}
        used = set()
        for tid, track in people_tracks.items():
            best, best_iou = None, config.POSE_MATCH_IOU
            for i, box in enumerate(boxes):
                if i in used:
                    continue
                v = iou(track.bbox, box)
                if v > best_iou:
                    best, best_iou = i, v
            if best is not None:
                used.add(best)
                matched[tid] = PoseObs(now, kps[best], tuple(float(v) for v in boxes[best]))
        return matched

    def _weapon_recent(self, camera_id: str, now: float) -> bool:
        return any(a.rule == "WEAPON" and a.camera_id == camera_id and now - a.timestamp <= config.HANDS_RAISED_WEAPON_S
                   for a in self.alerts.ranked(200))

    @staticmethod
    def _draw(state: CameraBehaviour, frame, now):
        for ts, event in state.overlays:
            if now - ts > 4:
                continue
            color = (58, 90, 239) if event.band in ("HIGH", "CRITICAL") or event.rule in (FIGHT, HANDS_RAISED, PERSON_DOWN, CROWD_PANIC) else (148, 215, 248)
            if event.bbox:
                x1, y1, x2, y2 = [int(v) for v in event.bbox]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, event.label, (x1, min(frame.shape[0] - 6, y2 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if event.marker:
                x, y = int(event.marker[0]), int(event.marker[1])
                cv2.circle(frame, (x, y), 14, color, 3)
                cv2.drawMarker(frame, (x, y), color, cv2.MARKER_CROSS, 28, 2)
                cv2.putText(frame, event.label, (x + 16, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
