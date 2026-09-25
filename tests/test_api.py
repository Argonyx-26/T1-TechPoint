"""API contract tests with FastAPI TestClient. Loads the real models once (GPU if available)."""
import time

import pytest
from fastapi.testclient import TestClient

from backend import config

STATUS_KEYS = {"running", "source", "object_count", "fps", "latency_ms", "device", "alert_count"}
ALERT_KEYS = {"id", "type", "rule", "severity", "description", "message", "timestamp",
              "created_at", "zone", "track_id", "has_evidence"}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # Keep the real data/zones.json untouched
    config.ZONES_PATH = tmp_path_factory.mktemp("data") / "zones.json"
    from backend import main

    with TestClient(main.app) as c:
        c.pipeline = main.pipeline
        yield c
        c.post("/api/source/stop")


def test_status_idle(client):
    client.post("/api/source/stop")
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == STATUS_KEYS
    assert body["running"] is False and body["source"] is None


def test_samples_listed(client):
    r = client.get("/api/source/samples")
    assert r.status_code == 200
    assert "vtest.avi" in r.json()


def test_source_sample_vtest(client):
    r = client.post("/api/source/sample", json={"name": "vtest.avi"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == STATUS_KEYS
    assert body["running"] is True
    assert body["source"] == "sample: vtest.avi"

    time.sleep(3)  # let a few frames through the pipeline
    s = client.get("/api/status").json()
    assert s["running"] is True and s["fps"] > 0 and s["object_count"] > 0

    stopped = client.post("/api/source/stop").json()
    assert stopped["running"] is False and stopped["source"] is None


@pytest.mark.parametrize("name", ["../README.md", "..\\backend\\main.py", "..", "C:\\Windows\\win.ini",
                                  "/etc/passwd", "nope.mp4", ""])
def test_source_sample_rejects_bad_names(client, name):
    r = client.post("/api/source/sample", json={"name": name})
    assert r.status_code == 400
    assert "detail" in r.json()


def test_source_camera_rejects_garbage(client):
    r = client.post("/api/source/camera", json={"source": "not a camera"})
    assert r.status_code == 400 and "detail" in r.json()


def test_source_upload_rejects_non_video(client):
    r = client.post("/api/source/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400 and "detail" in r.json()


def test_zones_round_trip(client):
    zones = [
        {"id": "zone-1", "name": "Door", "polygon": [[0.1, 0.1], [0.4, 0.1], [0.4, 0.5]], "restricted": True},
        {"id": "zone-2", "name": "Lobby", "polygon": [[0.5, 0.5], [0.9, 0.5], [0.9, 0.9], [0.5, 0.9]],
         "restricted": False},
    ]
    r = client.post("/api/zones", json=zones)
    assert r.status_code == 200
    assert client.get("/api/zones").json() == zones
    assert config.ZONES_PATH.is_file()

    assert client.post("/api/zones", json=[]).json() == []
    assert client.get("/api/zones").json() == []


@pytest.mark.parametrize("bad", [
    [{"id": "z", "name": "x", "polygon": [[0.1, 0.1], [0.2, 0.2]]}],             # too few points
    [{"id": "z", "name": "x", "polygon": [[10, 10], [200, 10], [200, 200]]}],     # pixels, not 0-1
    {"id": "z"},                                                                  # not a list
])
def test_zones_rejects_invalid(client, bad):
    r = client.post("/api/zones", json=bad)
    assert r.status_code == 400 and "detail" in r.json()


def test_thresholds_round_trip(client):
    defaults = client.get("/api/thresholds").json()
    assert defaults == config.DEFAULT_THRESHOLDS

    new = {"loiter_seconds": 12, "crowd_threshold": 4, "unattended_seconds": 25,
           "wrong_direction_angle_deg": 90}
    r = client.post("/api/thresholds", json=new)
    assert r.status_code == 200
    assert client.get("/api/thresholds").json() == new
    assert client.pipeline.thresholds["crowd_threshold"] == 4  # applied live

    for bad in ({"crowd_threshold": 0}, {"loiter_seconds": "x"}, {"crowd_threshold": 2.5},
                {"wrong_direction_angle_deg": 270}, {"bogus": 1}):
        r = client.post("/api/thresholds", json=bad)
        assert r.status_code == 400, bad
    assert client.get("/api/thresholds").json() == new  # rejected updates change nothing

    client.post("/api/thresholds", json=config.DEFAULT_THRESHOLDS)


def test_alerts_shape_sort_and_evidence(client):
    mgr = client.pipeline.alerts
    mgr.clear()
    now = time.time()
    low = mgr.raise_alert("loitering", "Person #3 loitering", zone="Lobby", track_id=3, now=now - 5)
    crit = mgr.raise_alert("weapon", "Knife detected", key="knife", evidence=b"\xff\xd8fakejpeg\xff\xd9",
                           now=now - 10)
    high = mgr.raise_alert("restricted_intrusion", "Person #7 entered", zone="Door", track_id=7, now=now)
    # same rule + zone + track inside the 12s cooldown is suppressed
    assert mgr.raise_alert("restricted_intrusion", "again", zone="Door", track_id=7, now=now + 1) is None

    r = client.get("/api/alerts?limit=50")
    assert r.status_code == 200
    items = r.json()
    assert [a["id"] for a in items] == [crit.id, high.id, low.id]  # severity first
    for a in items:
        assert set(a) == ALERT_KEYS
        assert a["severity"] == a["severity"].lower()
        assert a["description"] == a["message"]
        assert len(a["timestamp"]) == 8 and a["timestamp"].count(":") == 2
    weapon = items[0]
    assert weapon["type"] == "Weapon Detected" and weapon["rule"] == "weapon"
    assert weapon["severity"] == "critical" and weapon["has_evidence"] is True

    ev = client.get(f"/api/alerts/{crit.id}/evidence")
    assert ev.status_code == 200 and ev.headers["content-type"] == "image/jpeg"
    assert client.get("/api/status").json()["alert_count"] == 3
    mgr.clear()


def test_zone_cooldown_does_not_suppress_weapon(client):
    mgr = client.pipeline.alerts
    mgr.clear()
    now = time.time()
    assert mgr.raise_alert("restricted_intrusion", "1 person", zone="Door", track_id=1, key="z1", now=now)
    # another person in the same zone inside the cooldown: suppressed
    assert mgr.raise_alert("restricted_intrusion", "2 people", zone="Door", track_id=2, key="z1",
                           now=now + 1) is None
    assert mgr.raise_alert("restricted_intrusion", "1 person", zone="Gate", track_id=3, key="z2",
                           now=now + 1)
    assert mgr.raise_alert("weapon", "Knife detected", key="knife", now=now + 1)
    assert mgr.raise_alert("restricted_intrusion", "2 people", zone="Door", track_id=4, key="z1",
                           now=now + 13)
    mgr.clear()


def test_delete_alerts(client):
    mgr = client.pipeline.alerts
    mgr.raise_alert("weapon", "Knife detected", key="knife")
    assert client.get("/api/status").json()["alert_count"] >= 1
    r = client.delete("/api/alerts")
    assert r.status_code == 200 and r.json()["ok"] is True
    assert client.get("/api/alerts").json() == []
    assert client.get("/api/status").json()["alert_count"] == 0


@pytest.mark.parametrize("alert_id", ["999999", "demo-1727000000000"])
def test_evidence_404(client, alert_id):
    r = client.get(f"/api/alerts/{alert_id}/evidence")
    assert r.status_code == 404


def test_audit_accepts_any_json(client):
    r = client.post("/api/audit", json={"action": "SYSTEM_READY", "detail": "x"})
    assert r.status_code == 200 and r.json() == {"ok": True}


def test_dashboard_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "VIGIL" in r.text
