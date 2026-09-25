from collections import deque

from backend.analytics.rules import (
    CROWD_SURGE,
    CROWD_THRESHOLD,
    LOITERING,
    RESTRICTED_ZONE_INTRUSION,
    UNATTENDED_OBJECT,
    WRONG_DIRECTION,
    RuleEngine,
)
from backend.tracking import TrackState
from backend.zones import Zone

SQUARE = [(0, 0), (100, 0), (100, 100), (0, 100)]


def make_track(track_id, cls_name, positions, base_ts=0.0, dt=0.3):
    """positions: chronological (x, y) samples, dt seconds apart."""
    history = deque()
    ts = base_ts
    for pos in positions:
        history.append((ts, pos))
        ts += dt
    last_ts, last_pos = history[-1]
    return TrackState(
        track_id=track_id,
        cls_name=cls_name,
        bbox=(last_pos[0] - 5, last_pos[1] - 5, last_pos[0] + 5, last_pos[1] + 5),
        conf=0.9,
        first_seen=base_ts,
        last_seen=last_ts,
        history=history,
    )


def test_restricted_zone_intrusion():
    engine = RuleEngine()
    zone = Zone(id="z1", name="Vault", polygon=SQUARE, restricted=True)
    person = make_track(1, "person", [(50, 50)] * 6)

    alerts = engine.evaluate({1: person}, [zone], timestamp=100.0)

    assert any(a.rule == RESTRICTED_ZONE_INTRUSION and a.track_ids == [1] for a in alerts)


def test_no_intrusion_outside_zone():
    engine = RuleEngine()
    zone = Zone(id="z1", name="Vault", polygon=SQUARE, restricted=True)
    person = make_track(1, "person", [(500, 500)] * 6)

    alerts = engine.evaluate({1: person}, [zone], timestamp=100.0)

    assert not any(a.rule == RESTRICTED_ZONE_INTRUSION for a in alerts)


def test_crowd_threshold_triggers_at_threshold():
    engine = RuleEngine()
    zone = Zone(id="z1", name="Lobby", polygon=SQUARE, crowd_threshold=3)
    persons = {
        i: make_track(i, "person", [(10 * i + 5, 50)] * 6) for i in range(1, 4)
    }

    alerts = engine.evaluate(persons, [zone], timestamp=100.0)

    threshold_alerts = [a for a in alerts if a.rule == CROWD_THRESHOLD]
    assert len(threshold_alerts) == 1
    assert sorted(threshold_alerts[0].track_ids) == [1, 2, 3]
    assert threshold_alerts[0].message.startswith("Crowd threshold")


def test_crowd_threshold_not_triggered_below_threshold():
    engine = RuleEngine()
    zone = Zone(id="z1", name="Lobby", polygon=SQUARE, crowd_threshold=5)
    persons = {i: make_track(i, "person", [(10 * i, 50)] * 6) for i in range(1, 4)}

    alerts = engine.evaluate(persons, [zone], timestamp=100.0)

    assert not any(a.rule == CROWD_THRESHOLD for a in alerts)


def test_loitering_triggers_after_dwell_limit():
    engine = RuleEngine()
    engine.loiter_seconds = 5.0
    zone = Zone(id="z1", name="Platform", polygon=SQUARE)
    # Stationary history (tight cluster of points) so speed stays near zero.
    person = make_track(1, "person", [(50, 50)] * 8, dt=0.1)

    fired = False
    for t in range(0, 12):
        alerts = engine.evaluate({1: person}, [zone], timestamp=float(t))
        if any(a.rule == LOITERING for a in alerts):
            fired = True
            break

    assert fired, "expected a LOITERING alert once dwell time exceeded the limit"


def test_loitering_resets_when_person_leaves_zone():
    engine = RuleEngine()
    engine.loiter_seconds = 5.0
    zone = Zone(id="z1", name="Platform", polygon=SQUARE)
    person = make_track(1, "person", [(50, 50)] * 8, dt=0.1)

    engine.evaluate({1: person}, [zone], timestamp=0.0)
    engine.evaluate({1: person}, [zone], timestamp=3.0)
    assert engine._dwell.get((1, "z1"), 0.0) > 0

    outside_person = make_track(1, "person", [(500, 500)] * 8, dt=0.1)
    engine.evaluate({1: outside_person}, [zone], timestamp=4.0)

    assert engine._dwell.get((1, "z1"), 0.0) == 0.0


def test_wrong_direction_triggers_against_allowed_vector():
    engine = RuleEngine()
    zone = Zone(id="z1", name="Exit Lane", polygon=SQUARE, allowed_direction=(1.0, 0.0))
    # Moving in -x while the lane requires +x movement.
    positions = [(90 - i * 10, 50) for i in range(6)]
    person = make_track(1, "person", positions, dt=0.3)

    alerts = engine.evaluate({1: person}, [zone], timestamp=100.0)

    assert any(a.rule == WRONG_DIRECTION and a.track_ids == [1] for a in alerts)


def test_correct_direction_does_not_trigger():
    engine = RuleEngine()
    zone = Zone(id="z1", name="Exit Lane", polygon=SQUARE, allowed_direction=(1.0, 0.0))
    positions = [(10 + i * 10, 50) for i in range(6)]
    person = make_track(1, "person", positions, dt=0.3)

    alerts = engine.evaluate({1: person}, [zone], timestamp=100.0)

    assert not any(a.rule == WRONG_DIRECTION for a in alerts)


def test_unattended_object_triggers_after_owner_leaves():
    engine = RuleEngine()
    engine.unattended_seconds = 10.0
    zone = Zone(id="z1", name="Concourse", polygon=SQUARE)
    bag = make_track(2, "backpack", [(50, 50)] * 6, dt=0.1)
    owner = make_track(1, "person", [(55, 55)] * 6, dt=0.1)

    # Frame 1: bag and owner together -> establishes "ever had a person".
    engine.evaluate({1: owner, 2: bag}, [zone], timestamp=0.0)
    # Frame 2: owner has left, bag remains stationary well past the limit.
    alerts = engine.evaluate({2: bag}, [zone], timestamp=15.0)

    assert any(a.rule == UNATTENDED_OBJECT and a.track_ids == [2] for a in alerts)


def test_object_never_accompanied_does_not_trigger():
    engine = RuleEngine()
    engine.unattended_seconds = 10.0
    zone = Zone(id="z1", name="Concourse", polygon=SQUARE)
    bag = make_track(2, "backpack", [(50, 50)] * 6, dt=0.1)

    engine.evaluate({2: bag}, [zone], timestamp=0.0)
    alerts = engine.evaluate({2: bag}, [zone], timestamp=20.0)

    assert not any(a.rule == UNATTENDED_OBJECT for a in alerts)


# ---------- crowd surge (synthetic count sequences) ----------

GATE = Zone(id="g", name="Gate A", polygon=[(0, 0), (1000, 0), (1000, 1000), (0, 1000)], crowd_threshold=50)


def people(n):
    """n stationary people inside GATE."""
    return {i: make_track(i, "person", [(10 + 10 * i, 50)] * 6) for i in range(1, n + 1)}


def feed(engine, counts, fps=5.0, start=0.0):
    """Feed one zone count per frame; returns [(timestamp, surge alert)]."""
    fired = []
    for i, n in enumerate(counts):
        ts = start + i / fps
        for a in engine.evaluate(people(n), [GATE], timestamp=ts):
            if a.rule == CROWD_SURGE:
                fired.append((ts, a))
    return fired


def test_gradual_rise_is_not_a_surge():
    engine = RuleEngine()
    # 60s steady at 4 to build a baseline, then +1 person every 15s: 4 -> 8
    # over a minute. Never +4 within 10s, never 1.5x the rolling average.
    counts = [4] * 300
    for n in range(5, 9):
        counts += [n] * 75
    assert feed(engine, counts) == []


def test_sudden_rise_is_a_surge():
    engine = RuleEngine()
    counts = [3] * 150 + [9] * 50   # a group of 6 walks in together
    fired = feed(engine, counts)
    assert len(fired) == 1, "one surge -> exactly one alert, not one per frame"
    alert = fired[0][1]
    assert alert.zone_name == "Gate A"
    assert alert.message.startswith("Crowd surge in 'Gate A': 3 -> 9 people in ")
    assert alert.message.endswith("s")


def test_ramping_surge_fires_once():
    engine = RuleEngine()
    counts = [3] * 150 + [5] * 10 + [7] * 10 + [9] * 50   # 3 -> 9 over ~4s
    # With a baseline of 3 the 1.5x-average check trips first (at 5); the
    # rest of the climb is the same surge, so it must not fire again.
    assert len(feed(engine, counts)) == 1


def test_noisy_flicker_is_not_a_surge():
    engine = RuleEngine()
    assert feed(engine, [5, 6] * 300) == []


def test_single_frame_detector_dropout_is_not_a_surge():
    engine = RuleEngine()
    # Detector loses everyone for one frame, then they're back: 6 -> 0 -> 6
    # must not read as a "+6 in 0.2s" rise.
    counts = ([6] * 20 + [0]) * 15
    assert feed(engine, counts) == []


def test_rise_well_above_rolling_average_is_a_surge():
    engine = RuleEngine()
    engine.surge_min_increase = 100  # isolate the rolling-average condition
    counts = [2] * 300 + [3] * 50 + [4] * 50 + [5] * 50   # 5 > 1.5 x ~2.6 avg
    fired = feed(engine, counts)
    assert fired, "expected the 1.5x rolling-average check to fire"
    assert "avg over 60s" in fired[0][1].message


def test_average_check_needs_minimum_people():
    engine = RuleEngine()
    engine.surge_min_increase = 100
    assert feed(engine, [0] * 300 + [2] * 200) == []   # 2 > 1.5x0 but under 3 people


def test_surge_rearms_after_it_clears():
    engine = RuleEngine()
    counts = [2] * 100 + [8] * 100 + [2] * 400 + [8] * 100
    assert len(feed(engine, counts)) == 2


def test_surge_thresholds_are_adjustable():
    engine = RuleEngine()
    engine.update_thresholds(surge_min_increase=2, surge_window_s=5.0)
    counts = [3] * 150 + [5] * 50
    fired = feed(engine, counts)
    assert fired and "3 -> 5 people" in fired[0][1].message


def test_zone_live_count_is_recorded():
    engine = RuleEngine()
    engine.evaluate(people(4), [GATE], timestamp=0.0)
    assert engine.zone_counts == {"g": 4}


def test_small_team_demo_surge_plus_two_in_five_seconds():
    # the dashboard's SMALL-TEAM DEMO preset: 1 person, then 2 more walk in
    engine = RuleEngine()
    engine.update_thresholds(surge_min_increase=2, surge_window_s=5)
    fired = feed(engine, [1] * 50 + [3] * 25)
    assert len(fired) == 1
    assert fired[0][1].message.startswith("Crowd surge in 'Gate A': 1 -> 3 people in ")
