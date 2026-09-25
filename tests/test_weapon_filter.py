"""WeaponFilter: pure logic, no model or GPU needed."""
from collections import namedtuple

import pytest

from backend.weapon_filter import WeaponFilter

Det = namedtuple("Det", "cls_name conf")
KNIFE = [Det("knife", 0.8)]
NONE = []


def feed(f, frames):
    return [set(f.update(d)) for d in frames]


def test_single_frame_never_confirms():
    f = WeaponFilter(window=8, min_hits=5)
    assert f.update(KNIFE) == set()


def test_confirms_on_fifth_hit_within_window():
    f = WeaponFilter(window=8, min_hits=5)
    states = feed(f, [KNIFE] * 5)
    assert states[:4] == [set()] * 4
    assert states[4] == {"knife"}


def test_scattered_hits_confirm_only_when_five_of_last_eight():
    f = WeaponFilter(window=8, min_hits=5)
    # hits at frames 0,2,4,6 -> 4 of 8, not enough; frame 7 makes 5 of 8
    frames = [KNIFE, NONE, KNIFE, NONE, KNIFE, NONE, KNIFE, KNIFE]
    states = feed(f, frames)
    assert states[6] == set()
    assert states[7] == {"knife"}


def test_confirmation_expires_when_hits_leave_window():
    f = WeaponFilter(window=8, min_hits=5)
    feed(f, [KNIFE] * 5)
    states = feed(f, [NONE] * 4)
    assert states[2] == {"knife"}   # last 8 = 5 hits + 3 empty
    assert states[3] == set()       # first hit slid out: 4 hits + 4 empty


def test_below_alert_threshold_counts_for_nothing():
    f = WeaponFilter(window=8, min_hits=5, alert_conf=0.5)
    # 0.25-0.49 is display-only: box drawn but never confirms
    assert feed(f, [[Det("knife", 0.49)]] * 8)[-1] == set()
    assert feed(f, [[Det("knife", 0.5)]] * 5)[-1] == {"knife"}


def test_non_threat_classes_ignored():
    f = WeaponFilter(window=8, min_hits=5)
    phone = [Det("smartphone", 0.99), Det("billete", 0.99)]
    assert feed(f, [phone] * 8)[-1] == set()


def test_classes_counted_independently():
    f = WeaponFilter(window=8, min_hits=5)
    frames = [[Det("knife", 0.9), Det("pistol", 0.9)]] * 3 + [KNIFE] * 2
    assert feed(f, frames)[-1] == {"knife"}  # knife 5 hits, pistol only 3


def test_multiple_boxes_same_class_count_once_per_frame():
    f = WeaponFilter(window=8, min_hits=5)
    two_knives = [Det("knife", 0.9), Det("knife", 0.9)]
    assert feed(f, [two_knives] * 4)[-1] == set()


def test_reset_clears_history():
    f = WeaponFilter(window=8, min_hits=5)
    feed(f, [KNIFE] * 4)
    f.reset()
    assert f.update(KNIFE) == set()


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        WeaponFilter(window=4, min_hits=5)
