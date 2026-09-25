from collections import deque

import numpy as np
import pytest

from backend import config
from backend.behavior import (
    CROWD_PANIC, FIGHT, HANDS_RAISED, PERSON_DOWN, THROW,
    CameraBehaviour, PoseObs, fight_frame, hands_raised, is_down, panic_frame, throw_from_track,
)
from backend.tracking import TrackState

FRAME = (640, 480)


def pose(ts, cx, top=100.0, h=240.0, lw=None, rw=None, le=None, re=None, lying=False):
    """Synthetic COCO skeleton. lw/rw/le/re override wrists/elbows (x, y)."""
    k = np.zeros((17, 3), np.float32)
    if lying:  # body along x, at floor height
        y = top + 0.9 * h
        pts = {0: (cx - 0.5 * h, y - 10), 5: (cx - 0.35 * h, y), 6: (cx - 0.35 * h, y - 8), 7: (cx - 0.2 * h, y),
               8: (cx - 0.2 * h, y - 8), 9: (cx - 0.1 * h, y), 10: (cx - 0.1 * h, y - 8), 11: (cx + 0.05 * h, y),
               12: (cx + 0.05 * h, y - 8), 13: (cx + 0.25 * h, y), 14: (cx + 0.25 * h, y - 8), 15: (cx + 0.45 * h, y),
               16: (cx + 0.45 * h, y - 8)}
        box = (cx - 0.55 * h, y - 0.2 * h, cx + 0.5 * h, y + 0.05 * h)
    else:
        pts = {0: (cx, top + 0.08 * h), 5: (cx - 0.12 * h, top + 0.2 * h), 6: (cx + 0.12 * h, top + 0.2 * h),
               7: le or (cx - 0.16 * h, top + 0.35 * h), 8: re or (cx + 0.16 * h, top + 0.35 * h),
               9: lw or (cx - 0.16 * h, top + 0.5 * h), 10: rw or (cx + 0.16 * h, top + 0.5 * h),
               11: (cx - 0.08 * h, top + 0.55 * h), 12: (cx + 0.08 * h, top + 0.55 * h),
               13: (cx - 0.08 * h, top + 0.75 * h), 14: (cx + 0.08 * h, top + 0.75 * h),
               15: (cx - 0.08 * h, top + 0.97 * h), 16: (cx + 0.08 * h, top + 0.97 * h)}
        box = (cx - 0.2 * h, top, cx + 0.2 * h, top + h)
    for i, (x, y) in pts.items():
        k[i] = (x, y, 0.9)
    return PoseObs(ts, k, box)


def run(state, frames, fps=6.0, tracks_fn=None, weapon_recent=False, restricted=()):
    """frames: list of {track_id: callable(ts) -> PoseObs}. Returns all events."""
    events = []
    for i, people_fns in enumerate(frames):
        ts = 1000.0 + i / fps
        people = {tid: fn(ts) for tid, fn in people_fns.items()}
        tracks = tracks_fn(ts, i) if tracks_fn else {}
        events += state.update(ts, people, tracks, FRAME, list(restricted), weapon_recent=weapon_recent)
    return events


# ---------------- hands raised ----------------
def up(ts, cx=300):
    return pose(ts, cx, lw=(cx - 30, 70), rw=(cx + 30, 70), le=(cx - 35, 110), re=(cx + 35, 110))


def test_hands_raised_needs_two_seconds():
    assert hands_raised(up(0)) and not hands_raised(pose(0, 300))
    brief = run(CameraBehaviour(), [{1: up}] * 8)                 # ~1.2s
    assert not [e for e in brief if e.rule == HANDS_RAISED]
    held = run(CameraBehaviour(), [{1: up}] * 16)                 # ~2.5s
    ev = [e for e in held if e.rule == HANDS_RAISED]
    assert len(ev) == 1 and ev[0].message.startswith("Hands raised - possible hold-up") and ev[0].band is None


def test_hands_raised_with_recent_weapon_is_critical():
    ev = [e for e in run(CameraBehaviour(), [{1: up}] * 16, weapon_recent=True) if e.rule == HANDS_RAISED]
    assert ev and ev[0].band == "CRITICAL"


# ---------------- person down ----------------
def test_person_down_after_being_upright():
    assert is_down(pose(0, 300, lying=True)) and not is_down(pose(0, 300))
    frames = [{1: lambda ts: pose(ts, 300)}] * 6 + [{1: lambda ts: pose(ts, 300, lying=True)}] * 24
    ev = [e for e in run(CameraBehaviour(), frames) if e.rule == PERSON_DOWN]
    assert len(ev) == 1 and ev[0].message == "Person down - possible medical emergency (#1)"


def test_lying_person_never_seen_upright_is_not_alerted():
    frames = [{1: lambda ts: pose(ts, 300, lying=True)}] * 30
    assert not [e for e in run(CameraBehaviour(), frames) if e.rule == PERSON_DOWN]


def test_briefly_bending_down_is_not_person_down():
    frames = [{1: lambda ts: pose(ts, 300)}] * 6 + [{1: lambda ts: pose(ts, 300, lying=True)}] * 10 + \
             [{1: lambda ts: pose(ts, 300)}] * 10
    assert not [e for e in run(CameraBehaviour(), frames) if e.rule == PERSON_DOWN]


# ---------------- fighting ----------------
def punching(cx, other_cx, phase_shift=0):
    """Arm swinging out toward the other person and back, every frame."""
    def fn(ts):
        i = int(round((ts - 1000.0) * 6)) + phase_shift
        out = i % 2 == 0
        d = 1 if other_cx > cx else -1
        wrist = (cx + d * (110 if out else 20), 175 + (5 if out else 60))
        elbow = (cx + d * (60 if out else 30), 165 if out else 185)
        body_dx = d * (8 if out else -8)
        return pose(ts, cx + body_dx, rw=wrist if d > 0 else None, lw=wrist if d < 0 else None,
                    re=elbow if d > 0 else None, le=elbow if d < 0 else None)
    return fn


def test_scuffle_is_a_possible_fight():
    frames = [{4: punching(260, 360), 7: punching(360, 260, 1)}] * 16
    ev = [e for e in run(CameraBehaviour(), frames) if e.rule == FIGHT]
    assert len(ev) == 1 and ev[0].message == "Possible fight between #4 and #7" and ev[0].track_ids == [4, 7]


def test_standing_close_and_still_is_not_a_fight():
    frames = [{4: lambda ts: pose(ts, 260), 7: lambda ts: pose(ts, 360)}] * 16
    assert not [e for e in run(CameraBehaviour(), frames) if e.rule == FIGHT]


def test_talking_with_slow_gestures_is_not_a_fight():
    def gesture(cx):
        return lambda ts: pose(ts, cx, rw=(cx + 60 + 10 * np.sin(ts), 170))
    frames = [{4: gesture(260), 7: gesture(360)}] * 16
    assert not [e for e in run(CameraBehaviour(), frames) if e.rule == FIGHT]


def test_fast_people_far_apart_are_not_fighting():
    frames = [{4: punching(100, 560), 7: punching(560, 100, 1)}] * 16
    assert not [e for e in run(CameraBehaviour(), frames) if e.rule == FIGHT]


# ---------------- crowd panic ----------------
def crowd_tracks(origin=(320, 240), n=4, calm_s=12, run_s=3, fps=6.0):
    """n people idling near origin, then all running outward."""
    dirs = [(np.cos(a), np.sin(a)) for a in np.linspace(0, 2 * np.pi, n, endpoint=False)]
    tracks = {i: TrackState(i, "person", (0, 0, 1, 1), 0.9, 1000.0, 1000.0, deque(maxlen=600)) for i in range(n)}

    def fn(ts, frame_i):
        t = ts - 1000.0
        for i, (dx, dy) in enumerate(dirs):
            if t < calm_s:
                r = 60 + 3 * np.sin(t + i)
            else:
                r = 60 + 250 * (t - calm_s)
            p = (origin[0] + dx * r, origin[1] + dy * r)
            tracks[i].history.append((ts, p))
            tracks[i].bbox = (p[0] - 15, p[1] - 40, p[0] + 15, p[1] + 40)
        return tracks
    frames = int((calm_s + run_s) * fps)
    return fn, frames


def test_crowd_dispersing_from_a_point_is_possible_panic():
    fn, n = crowd_tracks()
    events = run(CameraBehaviour(), [{}] * n, tracks_fn=fn)
    ev = [e for e in events if e.rule == CROWD_PANIC]
    assert len(ev) == 1 and ev[0].message.startswith("Possible panic - crowd dispersing")
    ox, oy = ev[0].marker
    assert abs(ox - 320) < 40 and abs(oy - 240) < 40, "estimated origin marked"


def test_group_walking_the_same_way_is_not_panic():
    tracks = {i: TrackState(i, "person", (0, 0, 1, 1), 0.9, 1000.0, 1000.0, deque(maxlen=600)) for i in range(4)}

    def fn(ts, frame_i):
        t = ts - 1000.0
        for i in range(4):
            speed = 5 if t < 12 else 60       # speeds up, but all in the same direction
            p = (50 + speed * t, 100 + 60 * i)
            tracks[i].history.append((ts, p))
        return tracks
    assert not [e for e in run(CameraBehaviour(), [{}] * 90, tracks_fn=fn) if e.rule == CROWD_PANIC]


def test_panic_frame_needs_three_people():
    movers = [((100, 100), (-50, -50)), ((300, 300), (50, 50))]
    assert panic_frame(movers) == (False, None)


# ---------------- throwing ----------------
def thrower_pose(ts):
    return pose(ts, 200, rw=(260, 170))


def bottle_tracks(path):
    """Object track following path(i) -> (x, y) or None (not detected)."""
    track = TrackState(50, "bottle", (0, 0, 1, 1), 0.3, 1000.0, 1000.0, deque(maxlen=600))

    def fn(ts, i):
        p = path(i)
        if p is None:
            return {}
        track.bbox = (p[0] - 6, p[1] - 12, p[0] + 6, p[1] + 12)
        track.history.append((ts, p))
        return {50: track}
    return fn


def test_object_thrown_toward_a_person_is_high():
    def path(i):
        return (262, 172) if i < 6 else (262 + 90 * (i - 5), 172)   # in hand, then flies right
    frames = [{5: thrower_pose, 9: lambda ts: pose(ts, 560)}] * 10
    ev = [e for e in run(CameraBehaviour(), frames, tracks_fn=bottle_tracks(path)) if e.rule == THROW]
    assert len(ev) == 1 and ev[0].message == "Object thrown: bottle, by #5 toward #9" and ev[0].band == "HIGH"


def test_object_thrown_into_empty_space_is_medium():
    def path(i):
        return (262, 172) if i < 6 else (262 + 70 * (i - 5), 172 - 40 * (i - 5))
    ev = [e for e in run(CameraBehaviour(), [{5: thrower_pose}] * 10, tracks_fn=bottle_tracks(path)) if e.rule == THROW]
    assert len(ev) == 1 and ev[0].message == "Object thrown: bottle, by #5" and ev[0].band == "MEDIUM"


def test_setting_an_object_down_slowly_is_not_a_throw():
    def path(i):
        return (262, 172 + 6 * i)                                   # slowly lowered
    assert not [e for e in run(CameraBehaviour(), [{5: thrower_pose}] * 12, tracks_fn=bottle_tracks(path)) if e.rule == THROW]


def test_wrist_spike_then_object_gone_is_a_throw():
    def swing(ts):
        i = int(round((ts - 1000.0) * 6))
        return pose(ts, 200, rw=(260, 170) if i < 6 else (360, 90))   # fast arm snap on frame 6
    def path(i):
        return (262, 172) if i < 6 else None                          # bottle vanishes from the hand
    ev = [e for e in run(CameraBehaviour(), [{5: swing}] * 9, tracks_fn=bottle_tracks(path)) if e.rule == THROW]
    assert ev and ev[0].message.startswith("Object thrown: bottle, by #5")


def test_throw_from_track_requires_growing_distance():
    hist = [(1000.0 + i / 6, (262 + 30 * ((-1) ** i), 172)) for i in range(6)]   # jitters in the hand
    wrists = {5: [(1000.0 + i / 6, [(260, 170)], 240.0) for i in range(6)]}
    assert throw_from_track(hist, wrists, 800.0) is None


def test_people_walking_past_each_other_are_not_fighting():
    """Pedestrians crossing close together move fast in the frame, but their
    limbs barely move relative to their bodies."""
    def walker(x0, v):
        def fn(ts):
            t = ts - 1000.0
            cx = x0 + v * t
            swing = 12 * np.sin(8 * t)          # normal arm swing
            return pose(ts, cx, lw=(cx - 38 + swing, 220), rw=(cx + 38 - swing, 220))
        return fn
    frames = [{1: walker(200, 90), 2: walker(420, -90)}] * 18
    assert not [e for e in run(CameraBehaviour(), frames) if e.rule == FIGHT]


def test_people_at_different_depths_overlapping_on_screen_are_not_fighting():
    """One person metres behind the other overlaps on screen (the vtest false
    positive): feet at different heights, different size."""
    near_person = punching(300, 330)
    def far_person(ts):
        i = int(round((ts - 1000.0) * 6))
        p = punching(330, 300, 1)(ts)
        # scale the far person down and up the frame
        k = p.kps.copy()
        k[:, 0] = 330 + (k[:, 0] - 330) * 0.6
        k[:, 1] = 40 + (k[:, 1] - 100) * 0.6
        return PoseObs(ts, k, (330 - 0.2 * 144, 40, 330 + 0.2 * 144, 40 + 144))
    frames = [{4: near_person, 7: far_person}] * 16
    assert not [e for e in run(CameraBehaviour(), frames) if e.rule == FIGHT]
