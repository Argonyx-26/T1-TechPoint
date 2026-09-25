from collections import deque

from backend.analytics.rules import (
    CROWD_SURGE,
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


def test_crowd_surge_triggers_at_threshold():
    engine = RuleEngine()
    zone = Zone(id="z1", name="Lobby", polygon=SQUARE, crowd_threshold=3)
    persons = {
        i: make_track(i, "person", [(10 * i + 5, 50)] * 6) for i in range(1, 4)
    }

    alerts = engine.evaluate(persons, [zone], timestamp=100.0)

    surge_alerts = [a for a in alerts if a.rule == CROWD_SURGE]
    assert len(surge_alerts) == 1
    assert sorted(surge_alerts[0].track_ids) == [1, 2, 3]


def test_crowd_surge_not_triggered_below_threshold():
    engine = RuleEngine()
    zone = Zone(id="z1", name="Lobby", polygon=SQUARE, crowd_threshold=5)
    persons = {i: make_track(i, "person", [(10 * i, 50)] * 6) for i in range(1, 4)}

    alerts = engine.evaluate(persons, [zone], timestamp=100.0)

    assert not any(a.rule == CROWD_SURGE for a in alerts)


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
