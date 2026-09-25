"""Rules engine and zone geometry: pure logic over fake tracks, no model needed."""
from dataclasses import dataclass

from backend.rules import RulesEngine, Thresholds
from backend.zones import Zone, point_in_polygon

FRAME = (1000, 1000)
# Normalized square covering the middle of the frame
MIDDLE = [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]


@dataclass
class FakeTrack:
    track_id: int
    foot_point: tuple
    cls_name: str = "person"


def engine(**overrides):
    return RulesEngine(Thresholds(overrides))


def rules_of(events):
    return [e.rule for e in events]


def test_point_in_polygon():
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert point_in_polygon((5, 5), sq)
    assert not point_in_polygon((15, 5), sq)


def test_zone_scales_normalized_points_to_frame():
    z = Zone("z", "Z", MIDDLE)
    assert z.pixel_polygon(1280, 720)[1] == (960.0, 180.0)


def test_restricted_intrusion_fires_once_per_entry():
    e = engine()
    zone = Zone("z1", "Vault", MIDDLE, restricted=True)
    inside, outside = FakeTrack(1, (500, 500)), FakeTrack(1, (50, 50))
    assert rules_of(e.evaluate([inside], [zone], FRAME, now=0)) == ["restricted_intrusion"]
    assert e.evaluate([inside], [zone], FRAME, now=1) == []
    e.evaluate([outside], [zone], FRAME, now=2)
    assert rules_of(e.evaluate([inside], [zone], FRAME, now=3)) == ["restricted_intrusion"]


def test_uses_bottom_center_point_not_box_center():
    e = engine()
    zone = Zone("z1", "Vault", MIDDLE, restricted=True)
    feet_outside = FakeTrack(1, (500, 900))  # box may overlap the zone, feet do not
    assert e.evaluate([feet_outside], [zone], FRAME, now=0) == []


def test_loitering_after_threshold_only():
    e = engine(loiter_seconds=10)
    zone = Zone("z1", "Lobby", MIDDLE)
    p = FakeTrack(4, (500, 500))
    assert e.evaluate([p], [zone], FRAME, now=0) == []
    assert e.evaluate([p], [zone], FRAME, now=9) == []
    ev = e.evaluate([p], [zone], FRAME, now=10)
    assert rules_of(ev) == ["loitering"] and ev[0].track_id == 4 and ev[0].zone == "Lobby"
    assert e.evaluate([p], [zone], FRAME, now=11) == []  # once per visit


def test_crowd_surge_in_zone():
    e = engine(crowd_threshold=3)
    zone = Zone("z1", "Gate", MIDDLE)
    people = [FakeTrack(i, (400 + i * 10, 500)) for i in range(3)]
    assert rules_of(e.evaluate(people[:2], [zone], FRAME, now=0)) == []
    assert rules_of(e.evaluate(people, [zone], FRAME, now=1)) == ["crowd_surge"]


def test_non_person_tracks_ignored():
    e = engine()
    zone = Zone("z1", "Vault", MIDDLE, restricted=True)
    car = FakeTrack(9, (500, 500), cls_name="car")
    assert e.evaluate([car], [zone], FRAME, now=0) == []


def test_threshold_change_applies_live():
    th = Thresholds()
    e = RulesEngine(th)
    zone = Zone("z1", "Lobby", MIDDLE)
    p = FakeTrack(1, (500, 500))
    e.evaluate([p], [zone], FRAME, now=0)
    assert e.evaluate([p], [zone], FRAME, now=5) == []
    th.update({"loiter_seconds": 5})
    assert e.evaluate([p], [zone], FRAME, now=5.1) != []
