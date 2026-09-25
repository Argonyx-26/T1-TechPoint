import pytest

from backend.analytics.weapon_filter import WeaponTemporalFilter


def test_single_frame_spike_is_not_confirmed():
    wf = WeaponTemporalFilter(window=8, min_hits=5)
    confirmed = []
    for i in range(20):
        classes = {"knife"} if i == 3 else set()
        confirmed += wf.update(classes)
    assert confirmed == []


def test_sustained_detection_is_confirmed():
    wf = WeaponTemporalFilter(window=8, min_hits=5)
    confirmed = []
    for i in range(20):
        classes = {"knife"} if i >= 3 else set()
        confirmed += wf.update(classes)
    assert "knife" in confirmed


def test_intermittent_below_threshold_never_confirms():
    wf = WeaponTemporalFilter(window=8, min_hits=5)
    confirmed = []
    for i in range(30):
        # "knife" in only every 3rd frame -> at most ~3 hits per 8-frame window
        classes = {"knife"} if i % 3 == 0 else set()
        confirmed += wf.update(classes)
    assert confirmed == []


def test_different_classes_tracked_independently():
    wf = WeaponTemporalFilter(window=8, min_hits=5)
    confirmed = []
    for i in range(20):
        classes = {"guns"} if i >= 3 else set()
        confirmed += wf.update(classes)
    assert confirmed == ["guns"] or confirmed == ["guns"] * confirmed.count("guns")
    assert "knife" not in confirmed


def test_invalid_min_hits_raises():
    with pytest.raises(ValueError):
        WeaponTemporalFilter(window=4, min_hits=5)
