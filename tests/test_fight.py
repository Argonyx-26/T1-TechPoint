"""Learned fight detector: gating and windowing (the model itself is measured
on footage by tools/train_fight_classifier.py and tools/eval_behaviour.py)."""
import numpy as np

from backend import config
from backend.fight import FightMonitor, close_group, people_region, window_vector


class FakeEmbedder:
    def __init__(self):
        self.calls = 0

    def encode_images(self, crops):
        self.calls += len(crops)
        return np.ones((len(crops), 512), np.float32)


class FixedClf:
    enabled = True

    def __init__(self, p):
        self.p = p

    def prob(self, v):
        return self.p


A = (100, 100, 160, 300)          # person, 60x200
B = (170, 105, 230, 302)          # beside A, same depth and size
FAR = (500, 100, 560, 300)
BEHIND = (180, 20, 215, 140)      # overlaps on screen but much smaller and higher: further back


def test_close_group_needs_two_people_in_contact_at_the_same_depth():
    assert len(close_group([A, B])) == 2
    assert close_group([A, FAR]) == []
    assert close_group([A, BEHIND]) == []
    assert close_group([A]) == []


def test_window_vector_is_mean_and_change():
    seq = np.array([[0, 0], [2, 4]], np.float32)
    assert window_vector(seq).tolist() == [1, 2, 2, 4]


def test_people_region_crops_the_group():
    frame = np.zeros((480, 640, 3), np.uint8)
    assert people_region(frame, [A, B]).shape[:2] == (int(302 + 0.2 * 202) - int(100 - 0.2 * 202),
                                                      int(230 + 0.2 * 130) - int(100 - 0.2 * 130))
    assert people_region(frame, []).shape == frame.shape


def _feed(mon, frames, fps=8.0):
    hits = []
    for i, boxes in enumerate(frames):
        hit = mon.update("c", 1000.0 + i / fps, np.zeros((480, 640, 3), np.uint8), boxes)
        if hit:
            hits.append(hit)
    return hits


def test_sustained_high_score_on_a_close_pair_is_a_fight():
    mon = FightMonitor(FakeEmbedder(), FixedClf(0.95))
    assert _feed(mon, [[A, B]] * 40)


def test_one_high_window_is_not_enough():
    class OneSpike(FixedClf):
        n = 0

        def prob(self, v):
            OneSpike.n += 1
            return 0.95 if OneSpike.n == 3 else 0.1
    assert _feed(FightMonitor(FakeEmbedder(), OneSpike(0)), [[A, B]] * 40) == []


def test_lone_person_or_spread_out_crowd_is_never_scored():
    emb = FakeEmbedder()
    mon = FightMonitor(emb, FixedClf(0.99))
    crowd = [(60 * k, 100, 60 * k + 40, 300) for k in range(0, 10, 2)]   # 5 people, 80 px apart
    assert _feed(mon, [[A]] * 40 + [crowd] * 40) == []
    assert emb.calls == 0


def test_pair_merged_into_one_box_is_still_scored():
    # grappling people are often detected as one box right after being two
    mon = FightMonitor(FakeEmbedder(), FixedClf(0.95))
    merged = (100, 100, 230, 302)
    assert _feed(mon, [[A, B]] * 4 + [[merged]] * 40)
