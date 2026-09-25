"""One owner per physical webcam: cameras on the same index share one capture."""
import threading
import time

import numpy as np

from backend.cameras import STATUS_ONLINE, STATUS_RECONNECTING, WebcamHub
from backend import phones


class FakeCap:
    def __init__(self, ok=True):
        self.ok = ok
        self.released = False

    def isOpened(self):
        return self.ok

    def read(self):
        time.sleep(0.01)
        if not self.ok or self.released:
            return False, None
        return True, np.zeros((48, 64, 3), np.uint8)

    def release(self):
        self.released = True


class Opener:
    def __init__(self, ok=True):
        self.ok = ok
        self.caps = []

    def __call__(self, index):
        cap = FakeCap(self.ok)
        self.caps.append(cap)
        return cap


def _wait(pred, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_second_camera_reuses_the_open_device():
    opener = Opener()
    hub = WebcamHub(opener)
    a = hub.acquire(0, "CAM-01")
    b = hub.acquire(0, "CAM-02")
    assert a is b
    assert hub.owner(0) == "CAM-01"
    assert _wait(lambda: a.latest() is not None)
    assert len(opener.caps) == 1


def test_device_released_only_when_last_camera_stops():
    opener = Opener()
    hub = WebcamHub(opener)
    hub.acquire(0, "CAM-01")
    hub.acquire(0, "CAM-02")
    assert _wait(lambda: opener.caps)
    hub.release(0, "CAM-01")
    assert not opener.caps[0].released
    hub.release(0, "CAM-02")
    assert opener.caps[0].released
    assert hub.get(0) is None


def test_open_failure_is_a_clear_error():
    hub = WebcamHub(Opener(ok=False))
    dev = hub.acquire(0, "CAM-01")
    assert _wait(lambda: dev.error)
    assert "Webcam 0 could not be opened - close other apps using the camera" in dev.error
    hub.release(0, "CAM-01")


def _camera_on_webcam(hub, tmp_path, code="CAM-01"):
    from backend.cameras import Camera, Location

    class Shared:
        weapon = None

    cam = Camera(f"cam-{code[-1]}", code, 0, Location(), Shared(), None, zone_path=tmp_path / f"zones_{code}.json")
    cam.webcams = hub
    cam._running = True
    t = threading.Thread(target=cam._capture_loop, daemon=True)
    t.start()
    return cam, t


def test_cameras_on_one_index_share_frames_and_release(tmp_path):
    opener = Opener()
    hub = WebcamHub(opener)
    cam1, t1 = _camera_on_webcam(hub, tmp_path, "CAM-01")
    cam2, t2 = _camera_on_webcam(hub, tmp_path, "CAM-02")
    assert _wait(lambda: cam1.status == STATUS_ONLINE and cam2.status == STATUS_ONLINE)
    assert len(opener.caps) == 1
    cam1._running = cam2._running = False
    t1.join(2), t2.join(2)
    assert opener.caps[0].released


def test_failed_webcam_camera_carries_the_reason(tmp_path):
    hub = WebcamHub(Opener(ok=False))
    cam, t = _camera_on_webcam(hub, tmp_path)
    assert _wait(lambda: cam.status == STATUS_RECONNECTING)
    assert "could not be opened" in cam.last_error
    cam._running = False
    t.join(2)


def test_test_source_uses_the_shared_stream(monkeypatch):


    hub = WebcamHub(Opener())
    monkeypatch.setattr(phones, "WEBCAMS", hub)
    dev = hub.acquire(0, "CAM-02")
    assert _wait(lambda: dev.latest() is not None)
    result = phones.test_source("0")
    assert result["ok"] and result["shared_with"] == "CAM-02"
    assert "Webcam 0 is already in use by CAM-02" in result["note"]
    hub.release(0, "CAM-02")
