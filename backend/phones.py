"""Phones as CCTV cameras: DroidCam (http://<ip>:4747/video), the IP Webcam
app (http://<ip>:8080/video) and the laptop webcam.

The free DroidCam app serves ONE client at a time, so only the backend ever
connects to a phone; the browser only shows /video_feed/{id}. Discovery
therefore skips every host that is already a camera source, and for the
others reads just the HTTP response headers before hanging up.
"""
import asyncio
import base64
import ipaddress
import socket
import threading
from typing import Iterable, List, Optional, Set

import cv2

from backend import config
from backend.cameras import WEBCAMS, open_capture, resize_to_width, resolve_source, source_label, webcam_open_error

PHONE_PORTS = {4747: "DroidCam", 8080: "IP Webcam"}


def phone_url(ip: str, port: int = 4747) -> str:
    return f"http://{ip}:{port}/video"


def _thumb(frame, source) -> dict:
    h, w = frame.shape[:2]
    _, buf = cv2.imencode(".jpg", resize_to_width(frame, 320), [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    return {"ok": True, "width": w, "height": h, "source_label": source_label(source),
            "thumb": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()}


def test_source(raw_source, timeout_s: float = None) -> dict:
    """Grab one frame within timeout_s. Returns {ok, width, height, thumb}
    (thumb = base64 JPEG) or {ok: False, error}."""
    timeout_s = timeout_s or config.CAMERA_TEST_TIMEOUT_S
    try:
        source = resolve_source(raw_source)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    if isinstance(source, int):
        shared = WEBCAMS.get(source)
        if shared is not None:
            # Already open for a camera: opening it a second time fails on
            # Windows, so test with the shared stream instead.
            owner = shared.users[0] if shared.users else "another camera"
            frame = shared.latest()
            if frame is None:
                return {"ok": False, "error": shared.error or f"Webcam {source} is already in use by {owner} and has no frame yet"}
            return dict(_thumb(frame, source), shared_with=owner,
                        note=f"Webcam {source} is already in use by {owner} - this camera will share its stream")
    result = {"ok": False, "error": webcam_open_error(source) if isinstance(source, int)
              else f"No frame from {source_label(source)} within {timeout_s:.0f}s"}

    def grab():
        cap = open_capture(source)
        try:
            ok, frame = cap.read() if cap.isOpened() else (False, None)
            if ok and frame is not None:
                result.clear()
                result.update(_thumb(frame, source))
        finally:
            cap.release()

    worker = threading.Thread(target=grab, daemon=True)
    worker.start()
    worker.join(timeout_s)
    return result


def local_subnets() -> List[ipaddress.IPv4Network]:
    """/24 networks of this machine's IPv4 addresses (loopback excluded)."""
    addresses: Set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addresses.add(info[4][0])
    except OSError:
        pass
    try:  # the address used for the default route (sends nothing)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            addresses.add(s.getsockname()[0])
    except OSError:
        pass
    nets = {ipaddress.ip_network(f"{a}/24", strict=False) for a in addresses if not a.startswith("127.")}
    return sorted(nets, key=str)


async def _probe(ip: str, port: int, timeout: float) -> Optional[dict]:
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(ip, port), timeout)
    except (OSError, asyncio.TimeoutError):
        return None
    try:
        writer.write(f"GET /video HTTP/1.1\r\nHost: {ip}:{port}\r\nConnection: close\r\n\r\n".encode())
        await writer.drain()
        head = await asyncio.wait_for(reader.read(1024), timeout)
    except (OSError, asyncio.TimeoutError):
        head = b""
    finally:
        writer.close()
    text = head.decode("latin-1", "replace").lower()
    if "multipart" in text or "image/jpeg" in text:
        state = "ready"
    elif "busy" in text:
        state = "busy"  # DroidCam already serving another client
    else:
        return None
    return {"ip": ip, "port": port, "app": PHONE_PORTS.get(port, "MJPEG"),
            "url": phone_url(ip, port), "state": state}


async def _discover(hosts: List[str], ports: Iterable[int], timeout: float) -> List[dict]:
    tasks = [_probe(ip, port, timeout) for ip in hosts for port in ports]
    found = [r for r in await asyncio.gather(*tasks) if r]
    return sorted(found, key=lambda r: (ipaddress.ip_address(r["ip"]), r["port"]))


def discover(in_use_hosts: Iterable[str] = (), ports: Iterable[int] = tuple(PHONE_PORTS),
             timeout: float = None) -> dict:
    """Scan the local /24 subnet(s) in parallel for phones serving MJPEG."""
    timeout = timeout or config.DISCOVER_TIMEOUT_S
    skip = set(in_use_hosts)
    subnets = local_subnets()
    hosts = [str(h) for net in subnets for h in net.hosts() if str(h) not in skip]
    found = asyncio.run(_discover(hosts, list(ports), timeout)) if hosts else []
    return {"subnets": [str(n) for n in subnets], "scanned": len(hosts), "skipped_in_use": sorted(skip),
            "found": found}
