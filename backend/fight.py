"""Learned fight detection (see tools/train_fight_classifier.py).

A window of FIGHT_CLF_WINDOW_S seconds, sampled at FIGHT_CLF_FPS, of CLIP
embeddings of the people region becomes one vector:
[mean embedding, mean |frame-to-frame change|]; a logistic regression trained
on real surveillance fights scores it. The same functions are used for
training and live, so features match exactly.
"""
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from backend import config

Box = Tuple[float, float, float, float]


def people_region(frame, boxes: Sequence[Box]):
    """Crop around every person (padded); the whole frame when there are none."""
    h, w = frame.shape[:2]
    if not boxes:
        return frame
    x1 = min(b[0] for b in boxes); y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes); y2 = max(b[3] for b in boxes)
    px, py = (x2 - x1) * config.FIGHT_CLF_PAD, (y2 - y1) * config.FIGHT_CLF_PAD
    x1, y1 = int(max(0, x1 - px)), int(max(0, y1 - py))
    x2, y2 = int(min(w, x2 + px)), int(min(h, y2 + py))
    if x2 - x1 < 16 or y2 - y1 < 16:
        return frame
    return frame[y1:y2, x1:x2]


def window_vector(seq: np.ndarray) -> np.ndarray:
    seq = np.asarray(seq, dtype=np.float32)
    change = np.abs(np.diff(seq, axis=0)).mean(0) if len(seq) > 1 else np.zeros(seq.shape[1], np.float32)
    return np.concatenate([seq.mean(0), change])


def clip_embeddings(shared, embedder, path: Path, with_times: bool = False):
    """Offline: embeddings of the people region, sampled at FIGHT_CLF_FPS."""
    from backend.cameras import resize_to_width

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step, next_t, i = 1.0 / config.FIGHT_CLF_FPS, 0.0, 0
    crops, times = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        i += 1
        if t + 1e-6 < next_t:
            continue
        next_t = t + step
        small = resize_to_width(frame, config.PROCESS_WIDTH)
        boxes, names = shared.detect(small)
        people = [tuple(float(v) for v in boxes.xyxy[k]) for k in range(len(boxes))
                  if names.get(int(boxes.cls[k])) == config.PERSON_CLASS and boxes.conf[k] >= config.DETECTOR_CONF_THRESHOLD]
        crops.append(people_region(small, people))
        times.append(t)
    seq = embedder.encode_images(crops) if crops else np.zeros((0, 512), np.float32)
    return (seq, times) if with_times else seq


class FightClassifier:
    def __init__(self, path: Optional[Path] = None):
        path = Path(path or config.FIGHT_CLF_MODEL)
        self.enabled = path.exists()
        if self.enabled:
            d = np.load(path)
            self.w, self.b, self.mu, self.sd = d["w"], float(d["b"]), d["mu"], d["sd"]

    def prob(self, vector: np.ndarray) -> float:
        if not self.enabled:
            return 0.0
        z = ((vector - self.mu) / self.sd) @ self.w + self.b
        return float(1.0 / (1.0 + np.exp(-z)))


def _close(a: Box, b: Box) -> bool:
    """Close in the scene (same test as the pose rule, on boxes): feet at a
    similar image height, similar size, and a small gap between the boxes."""
    ha, hb = a[3] - a[1], b[3] - b[1]
    if ha <= 0 or hb <= 0:
        return False
    mean_h = (ha + hb) / 2
    if abs(a[3] - b[3]) > config.FIGHT_DEPTH_FRAC * 2 * mean_h:
        return False
    if not 1 / (config.FIGHT_SIZE_RATIO * 1.2) <= ha / hb <= config.FIGHT_SIZE_RATIO * 1.2:
        return False
    gap_x = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    gap_y = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return np.hypot(gap_x, gap_y) < config.FIGHT_GAP_FRAC * ((a[2] - a[0]) + (b[2] - b[0])) / 2


def close_group(boxes: Sequence[Box]) -> List[Box]:
    """The largest group of people in close contact (2+), else []."""
    n = len(boxes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if _close(boxes[i], boxes[j]):
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(boxes[i])
    best = max(groups.values(), key=len, default=[])
    return best if len(best) >= 2 else []


class FightMonitor:
    """Per camera: while 2+ people are in close contact, embed their region at
    FIGHT_CLF_FPS and score FIGHT_CLF_WINDOW_S windows every half second; a
    fight needs FIGHT_CLF_SUSTAIN positive windows in a row."""

    def __init__(self, embedder, classifier: Optional[FightClassifier] = None):
        self.embedder = embedder
        self.clf = classifier or FightClassifier()
        self.state = {}

    @property
    def enabled(self) -> bool:
        return self.clf.enabled and self.embedder is not None

    def update(self, cam_id: str, ts: float, frame, boxes: Sequence[Box]):
        """(probability, group box) when a fight is sustained, else None."""
        from collections import deque

        st = self.state.setdefault(cam_id, {"buf": deque(), "last": -1e9, "scored": -1e9, "streak": 0,
                                            "group": None, "group_ts": -1e9, "counts": deque()})
        st["counts"].append((ts, len(boxes)))
        while st["counts"] and st["counts"][0][0] < ts - config.FIGHT_CLF_PEOPLE_LOOKBACK_S:
            st["counts"].popleft()
        group = close_group(boxes)
        if not group and 1 <= len(boxes) <= 2 and max(c for _, c in st["counts"]) >= 2:
            # two people grappling are often detected as ONE box (CAVIAR
            # Fight_OneManDown): a pair seen here moments ago still counts
            group = list(boxes)
        if group:
            st["group"], st["group_ts"] = group, ts
        elif st["group"] is not None and ts - st["group_ts"] <= 1.0:
            group = st["group"]   # grappling people are often boxed as one for a moment
        else:
            st["buf"].clear()
            st["streak"] = 0
            st["group"] = None
            return None
        if ts - st["last"] < 1.0 / config.FIGHT_CLF_FPS:
            return None
        st["last"] = ts
        st["buf"].append((ts, self.embedder.encode_images([people_region(frame, group)])[0]))
        while st["buf"] and st["buf"][0][0] < ts - config.FIGHT_CLF_WINDOW_S:
            st["buf"].popleft()
        if len(st["buf"]) < 0.75 * config.FIGHT_CLF_WINDOW_S * config.FIGHT_CLF_FPS or ts - st["scored"] < 0.5:
            return None
        st["scored"] = ts
        p = self.clf.prob(window_vector(np.stack([e for _, e in st["buf"]])))
        st["streak"] = st["streak"] + 1 if p >= config.FIGHT_CLF_THRESHOLD else 0
        if st["streak"] < config.FIGHT_CLF_SUSTAIN:
            return None
        union = (min(b[0] for b in group), min(b[1] for b in group), max(b[2] for b in group), max(b[3] for b in group))
        return p, union
