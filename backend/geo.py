"""Where things are: coordinates, address search, and phone-GPS locate links.

- normalize_latlng: Leaflet lets the map wrap round the world, so a click can
  come back as lng 437.586 (= 77.586). Everything stored goes through here.
- geocode: OpenStreetMap Nominatim, proxied by the backend because a browser
  cannot set the User-Agent Nominatim's usage policy asks for. At most one
  request per second, results cached.
- locate links: http://<lan ip>:8000 is not a secure origin, and phones only
  give GPS to secure pages, so the locate page is also served over HTTPS on
  LOCATE_HTTPS_PORT with a self-signed certificate for the laptop's LAN IP
  (the phone shows a one-time "not private" warning: Advanced -> Proceed).
  Each link carries a per-camera token so only someone who scanned the QR
  code can move that camera.
"""
import datetime
import hashlib
import hmac
import ipaddress
import json
import secrets
import socket
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

from backend import config


# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------
def normalize_latlng(lat, lng) -> Tuple[Optional[float], Optional[float]]:
    """(lat, lng) with lng wrapped into [-180, 180); (None, None) if either is
    missing. Raises ValueError for a latitude off the globe."""
    if lat is None or lng is None:
        return None, None
    lat, lng = float(lat), float(lng)
    if not -90.0 <= lat <= 90.0:
        raise ValueError(f"Latitude {lat} is outside -90..90")
    lng = ((lng + 180.0) % 360.0) - 180.0
    return round(lat, 7), round(lng, 7)


def describe_accuracy(meters: Optional[float]) -> str:
    if meters is None:
        return ""
    return f"±{meters / 1000:.1f} km" if meters >= 1000 else f"±{meters:.0f} m"


# ---------------------------------------------------------------------------
# Address search (Nominatim)
# ---------------------------------------------------------------------------
_geo_lock = threading.Lock()
_geo_last = 0.0
_geo_cache: dict = {}


def geocode(query: str, limit: int = 5, fetch=None) -> List[dict]:
    """[{label, lat, lng, type}] for a place/address. Serialised to one
    Nominatim request per second; identical queries are served from cache."""
    global _geo_last
    q = " ".join(query.split())
    if not q:
        return []
    key = (q.lower(), limit)
    if key in _geo_cache:
        return _geo_cache[key]
    url = config.NOMINATIM_URL + "?" + urllib.parse.urlencode(
        {"q": q, "format": "jsonv2", "limit": limit, "addressdetails": 0})
    headers = {"User-Agent": config.NOMINATIM_USER_AGENT, "Accept-Language": "en"}
    with _geo_lock:
        wait = 1.0 - (time.time() - _geo_last)
        if wait > 0:
            time.sleep(wait)
        try:
            raw = (fetch or _http_get)(url, headers)
        finally:
            _geo_last = time.time()
    results = [
        {"label": r.get("display_name", ""), "lat": float(r["lat"]), "lng": float(r["lon"]),
         "type": r.get("type") or r.get("category") or ""}
        for r in json.loads(raw)
    ]
    _geo_cache[key] = results
    return results


def _http_get(url: str, headers: dict) -> bytes:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=8) as resp:
        return resp.read()


# ---------------------------------------------------------------------------
# LAN address + locate links
# ---------------------------------------------------------------------------
def lan_ip() -> Optional[str]:
    """This laptop's IPv4 on the network phones will join (the default-route
    interface). None when offline."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))   # sends nothing
            ip = s.getsockname()[0]
            return None if ip.startswith("127.") else ip
    except OSError:
        return None


def _secret() -> bytes:
    path = config.DATA_DIR / "locate.secret"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_hex(16))
    return path.read_text().strip().encode()


def locate_token(cam_id: str) -> str:
    return hmac.new(_secret(), cam_id.encode(), hashlib.sha256).hexdigest()[:12]


def token_ok(cam_id: str, token: str) -> bool:
    return hmac.compare_digest(locate_token(cam_id), token or "")


def locate_url(code: str, cam_id: str, ip: Optional[str] = None) -> Optional[str]:
    ip = ip or lan_ip()
    if not ip:
        return None
    return f"https://{ip}:{config.LOCATE_HTTPS_PORT}/locate/{code}?k={locate_token(cam_id)}"


def qr_svg(text: str) -> str:
    import segno

    return segno.make(text, error="m").svg_inline(scale=4, border=2, dark="#111A19", light="#ffffff")


# ---------------------------------------------------------------------------
# Self-signed certificate for the LAN IP
# ---------------------------------------------------------------------------
def ensure_cert(ip: Optional[str]) -> Tuple[Path, Path]:
    """data/tls/lan.{crt,key}, re-issued when the LAN IP changes (a phone
    hotspot hands out a new address now and then)."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    folder = config.DATA_DIR / "tls"
    crt, key = folder / "lan.crt", folder / "lan.key"
    ips = sorted({"127.0.0.1", *([ip] if ip else [])})
    stamp = folder / "ips.txt"
    if crt.exists() and key.exists() and stamp.exists() and stamp.read_text() == ",".join(ips):
        return crt, key
    folder.mkdir(parents=True, exist_ok=True)
    pkey = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "VIGIL locate")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name)
        .public_key(pkey.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=825))
        .add_extension(x509.SubjectAlternativeName(
            [x509.DNSName("localhost")] + [x509.IPAddress(ipaddress.ip_address(a)) for a in ips]), critical=False)
        .sign(pkey, hashes.SHA256())
    )
    key.write_bytes(pkey.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                       serialization.NoEncryption()))
    crt.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    stamp.write_text(",".join(ips))
    return crt, key
