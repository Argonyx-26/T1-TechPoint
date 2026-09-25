from backend.analytics.severity import band_for_score, score_alert


def test_weapon_is_always_critical():
    score, band = score_alert("WEAPON")
    assert band == "CRITICAL"
    assert score >= 85


def test_loitering_alone_is_low_or_medium():
    score, band = score_alert("LOITERING")
    assert band in ("LOW", "MEDIUM")


def test_co_occurrence_raises_severity():
    base_score, _ = score_alert("LOITERING", sibling_rules=[])
    boosted_score, _ = score_alert("LOITERING", sibling_rules=["CROWD_SURGE", "RESTRICTED_ZONE_INTRUSION"])
    assert boosted_score > base_score


def test_co_occurrence_bonus_is_capped():
    score, _ = score_alert(
        "LOITERING",
        sibling_rules=["A", "B", "C", "D", "E", "F", "G", "H"],
    )
    assert score <= 100


def test_band_thresholds_are_monotonic():
    assert band_for_score(95) == "CRITICAL"
    assert band_for_score(70) == "HIGH"
    assert band_for_score(45) == "MEDIUM"
    assert band_for_score(10) == "LOW"
