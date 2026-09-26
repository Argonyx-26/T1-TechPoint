"""Weapon candidates on a face / t-shirt are suppressed; real held weapons are not.

The fixtures are real: alert 6 (2026-09-25 22:40:53, "CAM-01: Weapon
detected - knife 76%") fired on the user's face + glasses while looking down
with no knife in frame. Its box is measured from the evidence frame's red
overlay; the keypoints are yolov8n-pose on that same frame. Alert 4 is the
same session with a real knife held up beside the face.
"""
import numpy as np

from backend.analytics.weapon_filter import WeaponTemporalFilter
from backend.analytics.weapon_verify import covers_face, on_torso_without_hand, suppression_reason
from backend.detection.weapon_detector import WeaponDetection
from backend.pipeline import FramePipeline

# alert 6: looking down, glasses, no weapon
ALERT6_KNIFE_BOX = (420.0, 259.0, 640.0, 413.0)   # knife 0.76
ALERT6_POSE = np.array([
    (513, 372, 0.98), (559, 328, 0.95), (477, 307, 1.0), (0, 0, 0.08), (391, 251, 0.98),
    (0, 0, 0.48), (252, 397, 0.71), (0, 0, 0.01), (0, 0, 0.01), (0, 0, 0.02), (0, 0, 0.04),
    (0, 0, 0.0), (0, 0, 0.0), (0, 0, 0.0), (0, 0, 0.0), (0, 0, 0.0), (0, 0, 0.0),
], dtype=float)

# alert 4: real knife held upright beside the face
ALERT4_KNIFE_BOX = (440.0, 54.0, 528.0, 374.0)    # knife 0.53 on re-run, 0.72 live
ALERT4_POSE = np.array([
    (347, 199, 0.98), (372, 172, 0.99), (313, 173, 0.97), (405, 204, 0.91), (268, 209, 0.8),
    (488, 364, 0.97), (202, 382, 0.79), (0, 0, 0.24), (0, 0, 0.02), (0, 0, 0.23), (0, 0, 0.05),
    (0, 0, 0.01), (0, 0, 0.0), (0, 0, 0.01), (0, 0, 0.0), (0, 0, 0.0), (0, 0, 0.0),
], dtype=float)


def _person(kp=None):
    """Synthetic upright person facing the camera; override keypoints by index."""
    k = np.zeros((17, 3))
    base = {0: (320, 100), 1: (330, 90), 2: (310, 90), 3: (345, 95), 4: (295, 95),
            5: (380, 160), 6: (260, 160), 11: (360, 330), 12: (280, 330)}
    base.update(kp or {})
    for i, (x, y) in base.items():
        k[i] = (x, y, 0.9)
    return k


def test_alert6_face_box_is_suppressed():
    assert covers_face(ALERT6_KNIFE_BOX, ALERT6_POSE)
    assert suppression_reason(ALERT6_KNIFE_BOX, [ALERT6_POSE]) == "face"


def test_alert4_real_knife_beside_face_is_kept():
    assert not covers_face(ALERT4_KNIFE_BOX, ALERT4_POSE)
    assert suppression_reason(ALERT4_KNIFE_BOX, [ALERT4_POSE]) is None


def test_glasses_only_box_is_suppressed():
    # a small box on the eyes/glasses of an upright person
    assert suppression_reason((300, 80, 345, 105), [_person()]) == "face"


def test_dark_tshirt_box_without_hand_is_suppressed():
    # box in the middle of the chest, both wrists down at the sides
    person = _person({9: (400, 340), 10: (240, 340)})
    assert on_torso_without_hand((290, 200, 350, 260), person)
    assert suppression_reason((290, 200, 350, 260), [person]) == "torso"


def test_weapon_held_at_chest_is_kept():
    # same box, but a wrist is on it: something is being held
    person = _person({9: (330, 250), 10: (240, 340)})
    assert suppression_reason((290, 200, 350, 260), [person]) is None


def test_no_pose_means_no_suppression():
    assert suppression_reason(ALERT6_KNIFE_BOX, []) is None
    assert suppression_reason(ALERT6_KNIFE_BOX, [np.zeros((17, 3))]) is None


def test_filter_does_not_join_hits_from_different_places():
    wf = WeaponTemporalFilter(window=8, min_hits=5, track_dist=1.0)
    confirmed = []
    for i in range(8):
        # knife held up at the left for 3 frames, then flickers on a face at the right
        box = (40, 50, 100, 300) if i < 3 else (420, 259, 640, 413)
        confirmed += wf.update({"knife": box} if i < 3 or i % 2 else {})
    assert confirmed == []


def test_filter_confirms_a_steady_real_weapon():
    wf = WeaponTemporalFilter(window=8, min_hits=5, track_dist=1.0)
    confirmed = []
    for i in range(10):
        confirmed += wf.update({"knife": (440 + i * 3, 54, 528 + i * 3, 374)})
    assert "knife" in confirmed


# --- end to end through the pipeline -----------------------------------------
class _FixedWeapons:
    enabled = True

    def __init__(self, dets):
        self.dets = dets

    def infer(self, frame):
        return list(self.dets)


class _NoObjects:
    def infer(self, frame):
        return []


class _Alerts:
    def __init__(self):
        self.weapon = []

    def ingest_rule_alerts(self, *a, **k):
        pass

    def ingest_weapon_alert(self, cls_name, conf, bbox, ts, frame=None, camera=None):
        self.weapon.append((cls_name, conf))
        return None


def _run(det, pose, frames=20):
    alerts = _Alerts()
    pipe = FramePipeline(object_detector=_NoObjects(), weapon_detector=_FixedWeapons([det]),
                         alert_manager=alerts, lazy=True)
    pipe.pose_fn = lambda frame: [pose]
    frame = np.zeros((480, 640, 3), np.uint8)
    for _ in range(frames):
        pipe.process(frame)
    return alerts.weapon


def test_pipeline_alert6_sitting_still_never_fires():
    # the exact alert-6 detection, repeated every frame for 20 frames
    assert _run(WeaponDetection("knife", 0.76, ALERT6_KNIFE_BOX), ALERT6_POSE) == []


def test_pipeline_real_knife_still_fires():
    assert _run(WeaponDetection("knife", 0.72, ALERT4_KNIFE_BOX), ALERT4_POSE)


def test_small_gun_being_swung_is_one_weapon():
    # m2-res_480p (270x480): a ~20 px gun moving 25-60 px between analysed
    # frames. Measured in box sizes alone these never matched; the frame-size
    # floor keeps them one weapon (the face-flicker case above is ~350 px away).
    boxes = [(147, 203, 166, 222), (174, 208, 196, 230), (200, 213, 220, 231),
             (218, 216, 237, 234), (160, 208, 180, 229), (184, 206, 207, 228), (218, 216, 237, 234)]
    strict, fixed = WeaponTemporalFilter(window=8, min_hits=5, track_dist=1.0), WeaponTemporalFilter(window=8, min_hits=5, track_dist=1.0)
    s = f = []
    for b in [None] + boxes:
        frame = {"pistol": b} if b else {}
        s = strict.update(frame)
        f = fixed.update(frame, frame_size=(270, 480))
    assert s == [] and f == ["pistol"]


def test_two_very_confident_sightings_confirm_at_once():
    # m2-res_480p: the gun is strong (>= 0.70) in only a couple of frames per
    # 4.6 s play, never 5 of 8, so it used to take minutes of looping
    wf = WeaponTemporalFilter(window=8, min_hits=5, track_dist=1.0)
    frames = [({}, {}), ({"pistol": (160, 208, 180, 229)}, {"pistol": 0.74}),
              ({"pistol": (184, 206, 207, 228)}, {"pistol": 0.61}),
              ({"pistol": (218, 216, 237, 234)}, {"pistol": 0.76})]
    out = [wf.update(b, frame_size=(270, 480), confs=c) for b, c in frames]
    assert out[-1] == ["pistol"] and out[1] == []


def test_one_confident_sighting_alone_does_not_confirm():
    wf = WeaponTemporalFilter(window=8, min_hits=5, track_dist=1.0)
    out = [wf.update(b, frame_size=(640, 480), confs=c) for b, c in
           [({"knife": (100, 100, 130, 200)}, {"knife": 0.8})] + [({}, {})] * 7]
    assert not any(out)


def test_on_screen_box_does_not_jump_to_a_weak_hit_elsewhere():
    # live capture: a confirmed gun's box moved onto empty snow via a weak (0.3) hit
    class Seq:
        enabled = True

        def __init__(self, frames):
            self.frames = iter(frames)

        def infer(self, frame):
            return next(self.frames)

    gun = (160, 208, 180, 229)
    snow = (60, 380, 110, 420)
    frames = [[WeaponDetection("pistol", 0.8, gun)]] * 8 + [[WeaponDetection("pistol", 0.3, snow)]] * 3
    pipe = FramePipeline(object_detector=_NoObjects(), weapon_detector=Seq(frames), alert_manager=_Alerts(), lazy=True)
    img = np.zeros((480, 270, 3), np.uint8)
    for _ in frames:
        pipe.process(img)
    assert pipe._weapon_display_state["pistol"]["last_bbox"] == gun
