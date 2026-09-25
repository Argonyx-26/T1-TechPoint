"""Pose check for weapon candidates: is this box a face or a t-shirt?

The weapon model occasionally fires "knife" on a face with glasses (looking
down) or "pistol" on a dark t-shirt. Both are places a person's own body
explains the box, so a candidate that clears WEAPON_ALERT_MIN_CONF is checked
against the pose keypoints of the people in the frame before it may count
toward the 5-of-8 filter:

- face: the box mostly covers a face (nose / eyes / ears). A knife or pistol
  held up next to the face overlaps it a little, never mostly.
- torso: the box sits on someone's chest with neither wrist near it. A held
  weapon is at a hand; a print or fold on a t-shirt is not.

Pose runs only on frames that have a candidate, so an empty scene costs
nothing. These are pure functions over (17, 3) COCO keypoint arrays.
"""
from typing import Iterable, Optional, Tuple

import numpy as np

from backend import config

Box = Tuple[float, float, float, float]

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SH, R_SH, L_WR, R_WR, L_HIP, R_HIP = 5, 6, 9, 10, 11, 12


def _point(kps: np.ndarray, i: int) -> Optional[Tuple[float, float]]:
    x, y, c = (float(v) for v in kps[i])
    # ultralytics reports unseen keypoints at (0, 0), sometimes with mid confidence
    if c < config.POSE_KP_CONF or (x <= 1 and y <= 1):
        return None
    return x, y


def _area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _intersection(a: Box, b: Box) -> float:
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def face_box(kps: np.ndarray) -> Optional[Box]:
    """Head region from the face keypoints, padded to cover forehead/chin.
    None unless at least two of nose / eyes are visible."""
    core = [p for p in (_point(kps, i) for i in (NOSE, L_EYE, R_EYE)) if p]
    if len(core) < 2:
        return None
    pts = core + [p for p in (_point(kps, i) for i in (L_EAR, R_EAR)) if p]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    size = max(max(xs) - min(xs), max(ys) - min(ys), 20.0)
    pad = size * config.WEAPON_FACE_PAD
    return min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad * 1.5  # extra room for the chin


def covers_face(bbox: Box, kps: np.ndarray) -> bool:
    face = face_box(kps)
    area = _area(bbox)
    return face is not None and area > 0 and _intersection(bbox, face) / area >= config.WEAPON_FACE_OVERLAP


def on_torso_without_hand(bbox: Box, kps: np.ndarray) -> bool:
    ls, rs = _point(kps, L_SH), _point(kps, R_SH)
    if not (ls and rs):
        return False
    shoulder_w = abs(ls[0] - rs[0])
    if shoulder_w < 10:
        return False  # side-on: torso extent unknown
    hips = [p for p in (_point(kps, L_HIP), _point(kps, R_HIP)) if p]
    top = min(ls[1], rs[1])
    bottom = max(p[1] for p in hips) if hips else top + 1.6 * shoulder_w
    left, right = min(ls[0], rs[0]), max(ls[0], rs[0])
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    if not (left <= cx <= right and top <= cy <= bottom):
        return False
    reach = config.WEAPON_WRIST_REACH * shoulder_w
    for wrist in (_point(kps, L_WR), _point(kps, R_WR)):
        if wrist and (bbox[0] - reach <= wrist[0] <= bbox[2] + reach
                      and bbox[1] - reach <= wrist[1] <= bbox[3] + reach):
            return False
    return True


def suppression_reason(bbox: Box, people: Iterable[np.ndarray]) -> Optional[str]:
    """Why this weapon box is really part of someone's body, or None."""
    for kps in people:
        if covers_face(bbox, kps):
            return "face"
        if on_torso_without_hand(bbox, kps):
            return "torso"
    return None
