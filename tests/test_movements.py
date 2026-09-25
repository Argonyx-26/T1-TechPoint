from types import SimpleNamespace

import pytest

from backend.alerts.manager import AlertManager
from backend.movements import BLIND_SPOT_GAP, SUBJECT_MISSING, MovementTracker, fmt_duration
from backend.search import Sighting

T0 = 1_700_000_000.0


class FakeCameras:
    def __init__(self, links):
        self.links = links
        self.cams = {
            "cam-1": SimpleNamespace(id="cam-1", name="Gate", code="CAM-01", place="Main Gate",
                                     location=SimpleNamespace(to_dict=lambda: {"label": "Main Gate"})),
            "cam-2": SimpleNamespace(id="cam-2", name="Lobby", code="CAM-02", place="Lobby",
                                     location=SimpleNamespace(to_dict=lambda: {"label": "Lobby"})),
            "cam-3": SimpleNamespace(id="cam-3", name="Store", code="CAM-03", place="Store",
                                     location=SimpleNamespace(to_dict=lambda: {"label": "Store"})),
        }

    def get(self, cam_id):
        return self.cams.get(cam_id)

    def expected_seconds(self, a, b):
        for l in self.links:
            if {l["a"], l["b"]} == {a, b}:
                return l["seconds"]
        return None


class FakeSearch:
    def __init__(self):
        self.sightings = []
        self.alert_subject = {}

    def all_sightings(self):
        return list(self.sightings)

    def subject_for_alert(self, alert):
        return self.alert_subject.get(alert.id)

    def _camera_label(self, cam_id):
        return {"id": cam_id, "code": cam_id.upper(), "name": cam_id, "place": cam_id}


def visit(search, gid, cam, t_in, t_out, closed=True, track=1):
    s = Sighting(camera_id=cam, track_id=track, cls="person", first_seen=t_in, last_seen=t_out,
                 bbox=(10, 10, 50, 150), frame_size=(640, 480), global_id=gid, closed=closed)
    search.sightings.append(s)
    return s


@pytest.fixture
def world(tmp_path):
    search = FakeSearch()
    cams = FakeCameras([{"a": "cam-1", "b": "cam-2", "seconds": 20}])
    alerts = AlertManager(evidence_dir=tmp_path)
    return search, MovementTracker(search, cams, alerts, start=False), alerts


def test_fmt_duration():
    assert fmt_duration(45) == "45s"
    assert fmt_duration(130) == "2m 10s"


def test_normal_transit_is_logged_not_alerted(world):
    search, tracker, alerts = world
    visit(search, "G1", "cam-1", T0, T0 + 30)
    visit(search, "G1", "cam-2", T0 + 45, T0 + 60, closed=False, track=4)   # 15s gap, expected 20
    assert tracker.evaluate(T0 + 61) == []
    seg = tracker.log(now=T0 + 61)[0]["segments"]
    assert seg[0]["status"] == "normal" and seg[0]["gap_to_next_s"] == 15 and seg[0]["expected_s"] == 20
    assert seg[1]["status"] == "in_view"


def test_long_blind_spot_gap_alerts_medium_once(world):
    search, tracker, alerts = world
    visit(search, "G12", "cam-1", T0, T0 + 30)
    visit(search, "G12", "cam-2", T0 + 160, T0 + 170, closed=False, track=4)  # 130s gap
    created = tracker.evaluate(T0 + 171)
    assert len(created) == 1
    a = created[0]
    assert a.rule == BLIND_SPOT_GAP and a.band == "MEDIUM" and a.camera_id == "cam-2"
    assert "Subject #G12 spent 2m 10s in the blind spot between Main Gate and Lobby (expected ~20s)" in a.message
    assert tracker.evaluate(T0 + 172) == [], "one alert per transit"
    assert tracker.log(now=T0 + 172)[0]["segments"][0]["status"] == "long_gap"


def test_gap_between_2x_and_60s_is_slow_not_alerted(world):
    search, tracker, _ = world
    visit(search, "G1", "cam-1", T0, T0 + 30)
    visit(search, "G1", "cam-2", T0 + 80, T0 + 90, closed=False, track=4)   # 50s: > 2x20 but < 60
    assert tracker.evaluate(T0 + 91) == []
    assert tracker.log(now=T0 + 91)[0]["segments"][0]["status"] == "slow"


def test_missing_after_leaving_a_linked_camera(world):
    search, tracker, _ = world
    visit(search, "G7", "cam-1", T0, T0 + 30)
    assert tracker.evaluate(T0 + 100) == [], "not yet 120s"
    created = tracker.evaluate(T0 + 151)
    assert len(created) == 1 and created[0].rule == SUBJECT_MISSING and created[0].band == "MEDIUM"
    assert "Subject #G7 unaccounted for since leaving Main Gate at" in created[0].message
    assert tracker.evaluate(T0 + 200) == []
    assert tracker.log(now=T0 + 200)[0]["segments"][-1]["status"] == "missing"


def test_leaving_an_unlinked_camera_is_not_missing(world):
    search, tracker, _ = world
    visit(search, "G8", "cam-3", T0, T0 + 30)
    assert tracker.evaluate(T0 + 500) == []


def test_escalates_to_high_after_a_serious_alert(world):
    search, tracker, alerts = world
    weapon = alerts.ingest_event("WEAPON", "Weapon detected - knife 81%", T0 + 10, band="CRITICAL")
    search.alert_subject[weapon.id] = "G3"
    visit(search, "G3", "cam-1", T0, T0 + 30)
    created = tracker.evaluate(T0 + 160)
    assert created and created[0].band == "HIGH" and created[0].rule == SUBJECT_MISSING


def test_same_camera_return_is_not_a_blind_spot(world):
    search, tracker, _ = world
    visit(search, "G2", "cam-1", T0, T0 + 30)
    visit(search, "G2", "cam-1", T0 + 300, T0 + 310, closed=False, track=9)
    assert tracker.evaluate(T0 + 311) == []
    assert tracker.log(now=T0 + 311)[0]["segments"][0]["status"] == "returned"


def test_csv_export(world):
    search, tracker, _ = world
    visit(search, "G1", "cam-1", T0, T0 + 30)
    visit(search, "G1", "cam-2", T0 + 45, T0 + 60, track=4)
    csv_text = tracker.export_csv(minutes=10, now=T0 + 61)
    lines = csv_text.strip().splitlines()
    assert lines[0].startswith("subject,camera,camera_name,time_in,time_out")
    assert len(lines) == 3 and ",normal" in lines[1]
