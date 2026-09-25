from backend.alerts.manager import AlertManager
from backend.analytics.rules import CROWD_SURGE, CROWD_THRESHOLD, LOITERING, RuleAlert


def crowd(rule, zone_id, track_ids, ts):
    return RuleAlert(rule=rule, track_ids=track_ids, message="m", timestamp=ts, zone_id=zone_id, zone_name=zone_id)


def test_crowd_cooldown_is_per_rule_and_zone(tmp_path):
    mgr = AlertManager(evidence_dir=tmp_path)
    # Same zone, different people each frame -> still one alert per rule.
    created = mgr.ingest_rule_alerts([crowd(CROWD_THRESHOLD, "a", [1, 2, 3], 0.0)])
    created += mgr.ingest_rule_alerts([crowd(CROWD_THRESHOLD, "a", [4, 5, 6], 1.0)])
    assert len(created) == 1
    # A different rule or a different zone is not blocked by that cooldown.
    assert len(mgr.ingest_rule_alerts([crowd(CROWD_SURGE, "a", [1], 2.0)])) == 1
    assert len(mgr.ingest_rule_alerts([crowd(CROWD_THRESHOLD, "b", [1], 2.0)])) == 1


def test_crowd_alert_severity_bands(tmp_path):
    mgr = AlertManager(evidence_dir=tmp_path)
    (threshold,) = mgr.ingest_rule_alerts([crowd(CROWD_THRESHOLD, "a", [1], 0.0)])
    (surge,) = mgr.ingest_rule_alerts([crowd(CROWD_SURGE, "b", [1], 0.0)])
    assert threshold.band == "MEDIUM"
    assert surge.band == "HIGH"


def test_per_track_rules_still_keyed_by_track(tmp_path):
    mgr = AlertManager(evidence_dir=tmp_path)
    created = mgr.ingest_rule_alerts([crowd(LOITERING, "a", [1], 0.0), crowd(LOITERING, "a", [2], 0.0)])
    assert len(created) == 2
