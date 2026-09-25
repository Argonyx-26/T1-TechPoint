import json

import numpy as np
import pytest

from backend import config
from backend.cameras import CameraLimitError, CameraManager, Location
from backend.detection.object_detector import Detection
from backend.zones import Zone


class FakeWeapon:
    enabled = False


class FakeShared:
    """Stands in for the GPU models: tests never load YOLO."""
    weapon = FakeWeapon()


class FakeDetector:
    def __init__(self, detections):
        self.detections = detections

    def infer(self, frame):
        return list(self.detections)


class NoWeapons:
    enabled = False

    def infer(self, frame):
        return []


@pytest.fixture
def manager(tmp_path):
    return CameraManager(shared=FakeShared(), store_path=tmp_path / "cameras.json",
                         zones_dir=tmp_path, primary_zone_file=tmp_path / "zones.json")


def add(manager, name, **kw):
    return manager.add(name, "vtest.avi", start=False, **kw)


def test_limit_is_enforced(manager):
    for i in range(config.MAX_CAMERAS):
        add(manager, f"Cam {i}")
    with pytest.raises(CameraLimitError, match="Camera limit reached"):
        add(manager, "one too many")
    assert [c.id for c in manager.list()] == [f"cam-{i}" for i in range(1, config.MAX_CAMERAS + 1)]


def test_ids_are_reused_after_removal(manager):
    add(manager, "A"), add(manager, "B"), add(manager, "C")
    manager.remove("cam-2")
    assert add(manager, "D").id == "cam-2"


def test_unknown_source_is_rejected(manager):
    with pytest.raises(ValueError, match="Unknown source"):
        manager.add("x", "no_such_clip.mp4", start=False)


def test_sources_resolve(manager):
    assert manager.add("web", "0", start=False).source == 0
    phone = manager.add("phone", "http://192.168.1.20:4747/video", start=False)
    assert phone.to_dict()["source_label"] == "192.168.1.20:4747"


def run_frame(camera, detections):
    camera.pipeline.object_detector = FakeDetector(detections)
    camera.pipeline.weapon_detector = NoWeapons()
    camera.pipeline.process(np.zeros((480, 640, 3), np.uint8))


def person(track_id, x, y):
    return Detection(track_id=track_id, cls_name="person", conf=0.9, bbox=(x - 10, y - 30, x + 10, y + 30))


def test_zones_are_isolated_per_camera(manager):
    gate = add(manager, "Gate", location={"label": "Main Gate", "lat": 12.9, "lng": 77.5})
    lobby = add(manager, "Lobby")
    gate.pipeline.zone_store.replace_all(
        [Zone(id="vault", name="Vault", polygon=[(0, 0), (1, 0), (1, 1), (0, 1)], restricted=True)]
    )
    assert lobby.pipeline.zone_store.list() == []

    # A person stands in the same spot on both cameras for several frames.
    for _ in range(6):
        run_frame(gate, [person(1, 320, 240)])
        run_frame(lobby, [person(1, 320, 240)])

    alerts = manager.alert_manager.ranked()
    assert {a.camera_id for a in alerts} == {"cam-1"}, "only the camera with the zone alerts"
    alert = alerts[0].to_dict()
    assert alert["camera_name"] == "Gate"
    assert alert["location"]["label"] == "Main Gate"
    assert alert["location"]["lat"] == 12.9
    assert alert["description"].startswith("Main Gate (CAM-01): Person #1 entered restricted zone")


def test_cooldowns_are_per_camera(manager):
    a, b = add(manager, "A"), add(manager, "B")
    square = [Zone(id="z", name="Z", polygon=[(0, 0), (1, 0), (1, 1), (0, 1)], restricted=True)]
    a.pipeline.zone_store.replace_all(square)
    b.pipeline.zone_store.replace_all(square)
    for _ in range(6):
        run_frame(a, [person(1, 320, 240)])
        run_frame(b, [person(1, 320, 240)])
    per_camera = {}
    for alert in manager.alert_manager.ranked():
        per_camera[alert.camera_id] = per_camera.get(alert.camera_id, 0) + 1
    assert per_camera == {"cam-1": 1, "cam-2": 1}


def test_cameras_persist_and_restore_stopped(tmp_path, manager):
    add(manager, "Gate", location={"label": "Main Gate", "x": 0.2, "y": 0.4})
    add(manager, "Lobby")
    manager.update("cam-2", location={"label": "Lobby East"})
    manager.links.append({"a": "cam-1", "b": "cam-2", "seconds": 20})
    manager.save()

    saved = json.loads((tmp_path / "cameras.json").read_text())
    assert [c["id"] for c in saved["cameras"]] == ["cam-1", "cam-2"]

    restored = CameraManager(shared=FakeShared(), store_path=tmp_path / "cameras.json",
                             zones_dir=tmp_path, primary_zone_file=tmp_path / "zones.json")
    assert [c.name for c in restored.list()] == ["Gate", "Lobby"]
    assert all(c.status == "stopped" for c in restored.list())
    assert restored.get("cam-1").location.x == 0.2
    assert restored.get("cam-2").location.label == "Lobby East"
    assert restored.links == [{"id": "cam-1~cam-2", "a": "cam-1", "b": "cam-2", "seconds": 20}]


def test_removing_a_camera_drops_its_links(manager):
    add(manager, "A"), add(manager, "B")
    manager.links.append({"a": "cam-1", "b": "cam-2", "seconds": 20})
    manager.remove("cam-2")
    assert manager.links == []


def test_thresholds_apply_to_every_camera_including_new_ones(manager):
    add(manager, "A")
    manager.update_thresholds(loiter_seconds=3.0)
    later = add(manager, "B")
    assert manager.get("cam-1").pipeline.rule_engine.loiter_seconds == 3.0
    assert later.pipeline.rule_engine.loiter_seconds == 3.0


def test_location_defaults_to_camera_name():
    assert Location.from_dict(None, "Gate").label == "Gate"


def test_links_are_unordered_validated_and_persisted(tmp_path, manager):
    add(manager, "Gate"), add(manager, "Lobby"), add(manager, "Parking")
    manager.set_link("cam-2", "cam-1", 20)
    manager.set_link("cam-1", "cam-2", 25)  # same pair: replaces, not duplicates
    assert [(l["id"], l["seconds"]) for l in manager.links] == [("cam-1~cam-2", 25.0)]
    assert manager.expected_seconds("cam-2", "cam-1") == 25.0
    assert manager.expected_seconds("cam-1", "cam-3") is None
    with pytest.raises(ValueError):
        manager.set_link("cam-1", "cam-1", 5)
    with pytest.raises(KeyError):
        manager.set_link("cam-1", "cam-9", 5)
    restored = CameraManager(shared=FakeShared(), store_path=tmp_path / "cameras.json",
                             zones_dir=tmp_path, primary_zone_file=tmp_path / "zones.json")
    assert restored.expected_seconds("cam-1", "cam-2") == 25.0
    manager.remove_link("cam-1~cam-2")
    assert manager.links == []


def test_camera_reports_worst_and_latest_alert(manager):
    gate = add(manager, "Gate")
    gate.pipeline.zone_store.replace_all(
        [Zone(id="v", name="Vault", polygon=[(0, 0), (1, 0), (1, 1), (0, 1)], restricted=True)])
    assert gate.to_dict()["worst_alert"] is None
    for _ in range(6):
        run_frame(gate, [person(1, 320, 240)])
    info = gate.to_dict()
    assert info["worst_alert"] == "HIGH"
    assert "restricted zone" in info["latest_alert"]["message"]


def test_removed_camera_zones_do_not_leak_to_a_new_camera(manager):
    first = add(manager, "Old gate")
    first.pipeline.zone_store.replace_all(
        [Zone(id="v", name="Vault", polygon=[(0, 0), (1, 0), (1, 1), (0, 1)], restricted=True)])
    manager.remove("cam-1")
    assert add(manager, "New gate").pipeline.zone_store.list() == []


def test_title_does_not_repeat_the_code(manager):
    # the single-source flow names camera 1 "CAM-01": its title was "CAM-01 CAM-01"
    plain = manager.add("CAM-01", "vtest.avi", start=False)
    named = manager.add("Main Gate", "vtest.avi", start=False)
    assert plain.title == "CAM-01"
    assert named.title == "CAM-02 Main Gate"
    assert plain._overlay_text() == ""
    assert named._overlay_text() == "Main Gate"


def test_alert_message_does_not_repeat_the_code(manager):
    from backend.alerts.manager import AlertManager

    plain = manager.add("CAM-01", "vtest.avi", start=False)
    named = manager.add("Main Gate", "vtest.avi", start=False)
    assert AlertManager._placed("Weapon detected - knife 72%", plain) == "CAM-01: Weapon detected - knife 72%"
    assert AlertManager._placed("x", named) == "Main Gate (CAM-02): x"


def test_thresholds_persist_across_restart(tmp_path, manager):
    manager.add("Gate", "vtest.avi", start=False)
    manager.update_thresholds(save=True, surge_min_increase=2, surge_window_s=5.0)
    again = CameraManager(shared=FakeShared(), store_path=tmp_path / "cameras.json",
                          zones_dir=tmp_path, primary_zone_file=tmp_path / "zones.json")
    engine = again.get("cam-1").pipeline.rule_engine
    assert (engine.surge_min_increase, engine.surge_window_s) == (2, 5.0)
