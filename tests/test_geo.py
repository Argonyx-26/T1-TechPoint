import json
import time

import pytest

from backend import config, geo
from backend.cameras import CameraManager, Location


class FakeWeapon:
    enabled = False


class FakeShared:
    weapon = FakeWeapon()


@pytest.fixture
def manager(tmp_path):
    return CameraManager(shared=FakeShared(), store_path=tmp_path / "cameras.json",
                         zones_dir=tmp_path, primary_zone_file=tmp_path / "zones.json")


def test_wrapped_longitude_is_normalized():
    # the value actually found in data/cameras.json: the map had wrapped once round the world
    assert geo.normalize_latlng(12.976998, 437.586167) == (12.976998, pytest.approx(77.586167))
    assert geo.normalize_latlng(0, -190) == (0.0, 170.0)
    assert geo.normalize_latlng(None, 5) == (None, None)
    with pytest.raises(ValueError):
        geo.normalize_latlng(95, 10)


def test_location_restores_normalized_and_keeps_accuracy():
    loc = Location.from_dict({"lat": 12.97, "lng": 437.5, "accuracy_m": 8.0, "source": "gps"})
    assert -180 <= loc.lng < 180
    assert loc.to_dict()["accuracy_m"] == 8.0 and loc.to_dict()["source"] == "gps"


def test_site_persists_in_cameras_json(tmp_path, manager):
    manager.set_site({"lat": 19.1197, "lng": 72.8465, "label": "Andheri Station, Mumbai"})
    data = json.loads((tmp_path / "cameras.json").read_text())
    assert data["site"]["label"] == "Andheri Station, Mumbai"
    again = CameraManager(shared=FakeShared(), store_path=tmp_path / "cameras.json",
                          zones_dir=tmp_path, primary_zone_file=tmp_path / "zones.json")
    assert again.site["lat"] == 19.1197 and again.site["zoom"] == 17
    manager.set_site(None)
    assert json.loads((tmp_path / "cameras.json").read_text())["site"] is None


def test_camera_update_rejects_bad_latitude(manager):
    cam = manager.add("Gate", "vtest.avi", start=False)
    with pytest.raises(ValueError):
        manager.update(cam.id, location={"lat": 123, "lng": 10})


def test_locate_token_is_per_camera(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    t2 = geo.locate_token("cam-2")
    assert geo.token_ok("cam-2", t2)
    assert not geo.token_ok("cam-3", t2)
    assert not geo.token_ok("cam-2", "")


def test_locate_url_uses_lan_ip_and_https(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    url = geo.locate_url("CAM-02", "cam-2", ip="192.168.43.17")
    assert url.startswith(f"https://192.168.43.17:{config.LOCATE_HTTPS_PORT}/locate/CAM-02?k=")
    assert "127.0.0.1" not in url
    assert geo.qr_svg(url).startswith("<svg")


def test_geocode_sends_user_agent_rate_limits_and_caches(monkeypatch):
    calls = []

    def fetch(url, headers):
        calls.append((time.time(), url, headers))
        return json.dumps([{"display_name": "Andheri, Mumbai", "lat": "19.1197", "lon": "72.8465", "type": "station"}])

    monkeypatch.setattr(geo, "_geo_cache", {})
    a = geo.geocode("Andheri Station, Mumbai", fetch=fetch)
    b = geo.geocode("Churchgate, Mumbai", fetch=fetch)
    c = geo.geocode("andheri   station, mumbai", fetch=fetch)   # cached
    assert a[0] == {"label": "Andheri, Mumbai", "lat": 19.1197, "lng": 72.8465, "type": "station"}
    assert c == a and len(calls) == 2
    assert calls[1][0] - calls[0][0] >= 0.95                  # max 1 request / s
    assert "VIGIL" in calls[0][2]["User-Agent"]


def test_self_signed_cert_covers_lan_ip(tmp_path, monkeypatch):
    from cryptography import x509

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    crt, key = geo.ensure_cert("192.168.43.17")
    cert = x509.load_pem_x509_certificate(crt.read_bytes())
    san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert "192.168.43.17" in [str(i) for i in san.get_values_for_type(x509.IPAddress)]
    before = crt.read_bytes()
    geo.ensure_cert("192.168.43.17")
    assert crt.read_bytes() == before        # reused while the IP is unchanged
    geo.ensure_cert("10.0.0.5")
    assert crt.read_bytes() != before        # re-issued for a new hotspot IP
