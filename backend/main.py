"""FastAPI application: serves the analyst dashboard, streams annotated video
per camera, and exposes the cameras/alerts/zones/source/threshold/audit REST
API. The single-source endpoints (/video_feed, /api/source/*, /api/zones)
act on camera 1.
"""
import re
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend import config
from backend.app_state import PRIMARY_ID, app_state
from backend.audit import audit, get_audit
from backend.cameras import (
    STATUS_CONNECTING,
    STATUS_ONLINE,
    Camera,
    CameraLimitError,
)
from backend import features, geo
from backend.locate import router as locate_router, start_https_server
from backend.phones import discover, test_source
from backend.search import SearchRefused
from backend.zones import Zone

FRONTEND_DIR = config.BASE_DIR / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # No auto-started source on boot -- the dashboard opens idle; saved
    # cameras are restored STOPPED and the analyst starts what they need.
    audit("SERVER_START", f"device {config.DEVICE}, weapon model {config.WEAPON_MODEL_PATH.name}, "
                          f"{len(app_state.manager.cameras)} saved camera(s) restored stopped")
    app.state.locate_base = start_https_server()
    yield
    app_state.stop()
    audit("SERVER_STOP", "shutdown")


app = FastAPI(title="Vigil Threat Detection & Situational Awareness", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
app.include_router(locate_router)


@app.get("/", response_class=HTMLResponse)
def index():
    return (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/legacy", response_class=HTMLResponse)
def legacy_dashboard():
    """The previous VIGIL dashboard (frontend/legacy/index.html)."""
    return (FRONTEND_DIR / "legacy" / "index.html").read_text(encoding="utf-8")


@app.get("/classic", response_class=HTMLResponse)
def classic_dashboard():
    """The original dashboard (frontend/legacy/classic.html + app.js)."""
    return (FRONTEND_DIR / "legacy" / "classic.html").read_text(encoding="utf-8")


# ---------- video ----------
def _mjpeg_generator(cam_id: str):
    boundary = b"--frame"
    delay = 1.0 / config.STREAM_MAX_FPS
    last = None
    while True:
        if cam_id != PRIMARY_ID and app_state.manager.get(cam_id) is None:
            return  # camera removed: end the stream
        jpeg = app_state.latest_jpeg(cam_id)
        if jpeg is not None and jpeg is not last:
            yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            last = jpeg
        time.sleep(delay)


def _stream(cam_id: str):
    return StreamingResponse(_mjpeg_generator(cam_id), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/video_feed")
def video_feed():
    return _stream(PRIMARY_ID)


@app.get("/video_feed/{cam_id}")
def camera_video_feed(cam_id: str):
    _camera_or_404(cam_id)
    return _stream(cam_id)


@app.get("/api/cameras/{cam_id}/snapshot")
def camera_snapshot(cam_id: str):
    """Latest annotated frame as one JPEG. The camera wall polls this instead
    of holding an MJPEG stream per tile: browsers allow only 6 connections
    per host over HTTP/1.1, and 4 tile streams + the big view + the zone
    backdrop would starve every API call."""
    camera = _camera_or_404(cam_id)
    jpeg = app_state.latest_jpeg(cam_id)
    if jpeg is None:
        raise HTTPException(status_code=503, detail=camera.last_error or f"{camera.code} has no frame yet ({camera.status})")
    return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/status")
def get_status():
    return app_state.status()


@app.get("/api/dashboard")
def get_dashboard(minutes: float = 30, alerts: int = 50):
    """Everything the dashboard polls, in one request: status, alerts,
    cameras and movements. One round trip every 2s instead of four."""
    return {
        "status": app_state.status(),
        "alerts": [a.to_dict() for a in app_state.alert_manager.ranked(alerts)],
        "cameras": [c.to_dict() for c in app_state.manager.list()],
        "movements": app_state.movements.log(minutes),
    }


# ---------- alerts ----------
@app.get("/api/alerts")
def get_alerts(limit: int = 50, camera_id: Optional[str] = None):
    alerts = app_state.alert_manager.ranked(limit if camera_id is None else 500)
    if camera_id:
        alerts = [a for a in alerts if a.camera_id == camera_id][:limit]
    return [a.to_dict() for a in alerts]


@app.delete("/api/alerts")
def clear_alerts():
    """Clear the alert feed (analyst acknowledged). Cooldowns are kept, so a
    still-ongoing event doesn't instantly re-fire the moment it's cleared."""
    cleared = app_state.alert_manager.clear()
    audit("ALERTS_CLEARED", f"{cleared} alert(s) cleared from the feed", actor="operator")
    return {"ok": True, "cleared": cleared}


class AckIn(BaseModel):
    who: str = "operator"


@app.post("/api/alerts/{alert_id}/ack")
def acknowledge_alert(alert_id: int, body: Optional[AckIn] = None):
    who = (body.who if body else "operator")[:64]
    alert = app_state.alert_manager.acknowledge(alert_id, who)
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert.to_dict()


@app.get("/api/alerts/{alert_id}/evidence")
def get_evidence(alert_id: int):
    alert = app_state.alert_manager.get(alert_id)
    if alert is None or not alert.evidence_path or not Path(alert.evidence_path).exists():
        raise HTTPException(status_code=404, detail="No evidence available for this alert")
    return FileResponse(alert.evidence_path, media_type="image/jpeg")


# ---------- Ask Vigil: search, follow, backtrack ----------
class SearchIn(BaseModel):
    query: str = Field(..., max_length=200)
    minutes: float = Field(30, gt=0, le=240)
    limit: int = Field(12, ge=1, le=50)
    camera_ids: Optional[List[str]] = None


@app.post("/api/search")
def run_search(body: SearchIn):
    if not features.enabled("search"):
        raise HTTPException(status_code=400, detail="The search module is switched off (Analytics Modules)")
    try:
        results = app_state.search.search(body.query, body.minutes, body.limit, body.camera_ids)
    except SearchRefused as exc:
        audit("SEARCH_REFUSED", f"'{body.query}': {exc}", actor="operator")
        raise HTTPException(status_code=400, detail=str(exc))
    top = f"; top #{results[0]['global_id']} {results[0]['match']}%" if results else "; no results"
    audit("SEARCH_RUN", f"'{body.query}' last {body.minutes:g} min{top}", actor="operator")
    return results


@app.get("/api/search/thumb/{entry_id}")
def search_thumb(entry_id: int):
    jpeg = app_state.search.thumb(entry_id)
    if jpeg is None:
        raise HTTPException(status_code=404, detail="Thumbnail expired")
    return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@app.get("/api/search/reid-diagnostics")
def reid_diagnostics():
    return app_state.search.reid_diagnostics()


@app.get("/api/search/stats")
def search_stats():
    return app_state.search.stats()


class FollowIn(BaseModel):
    global_id: str = Field(..., max_length=16)


@app.post("/api/follow")
def follow_subject(body: FollowIn):
    app_state.search.follow(body.global_id)
    return app_state.search.follow_status()


@app.delete("/api/follow")
def unfollow_subject(body: FollowIn):
    app_state.search.unfollow(body.global_id)
    return app_state.search.follow_status()


@app.get("/api/follow")
def follow_status(since: int = 0):
    return app_state.search.follow_status(since)


@app.get("/api/subjects/{global_id}/route")
def subject_route(global_id: str):
    return app_state.search.route(global_id)


@app.get("/api/alerts/{alert_id}/backtrack")
def backtrack_alert(alert_id: int):
    alert = app_state.alert_manager.get(alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    gid = app_state.search.subject_for_alert(alert)
    if gid is None:
        return {"global_id": None, "sightings": [],
                "note": "No indexed person matches this alert (bags and weapons without a visible person can't be backtracked)"}
    audit("BACKTRACK", f"alert #{alert_id} -> subject #{gid}", actor="operator", alert_id=alert_id,
          camera_id=alert.camera_id)
    return {"global_id": gid, "sightings": app_state.search.route(gid, until=alert.timestamp)}


# ---------- analytics modules on/off ----------
class FeaturesIn(BaseModel):
    pose: Optional[bool] = None
    fighting: Optional[bool] = None
    throwing: Optional[bool] = None
    distress: Optional[bool] = None
    search: Optional[bool] = None
    reid: Optional[bool] = None
    gap_tracking: Optional[bool] = None


@app.get("/api/features")
def get_features():
    return features.all_features()


@app.post("/api/features")
def set_features(body: FeaturesIn):
    values = body.dict(exclude_none=True)
    before = features.all_features()
    after = features.set_features(**values)
    changed = {k: v for k, v in after.items() if before.get(k) != v}
    if changed:
        audit("FEATURES_CHANGED", ", ".join(f"{k} {'ON' if v else 'OFF'}" for k, v in changed.items()), actor="operator")
    return after


# ---------- movement log (blind spots between cameras) ----------
@app.get("/api/movements")
def movements(minutes: float = 30):
    return app_state.movements.log(minutes)


@app.get("/api/movements/export")
def export_movements(format: str = "csv", minutes: float = 30):
    if format != "csv":
        raise HTTPException(status_code=400, detail="format must be csv")
    audit("MOVEMENTS_EXPORTED", f"last {minutes:g} min as CSV", actor="operator")
    return Response(app_state.movements.export_csv(minutes), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="vigil-movements-{time.strftime("%Y%m%d-%H%M%S")}.csv"'})


# ---------- zones (per camera; /api/zones = camera 1) ----------
class ZoneIn(BaseModel):
    id: str
    name: str
    polygon: List[List[float]]
    restricted: bool = False
    crowd_threshold: Optional[int] = None
    loiter_seconds: Optional[float] = None
    allowed_direction: Optional[List[float]] = None


def _zone_dict(z: Zone) -> dict:
    return {
        "id": z.id,
        "name": z.name,
        "polygon": z.polygon,
        "restricted": z.restricted,
        "crowd_threshold": z.crowd_threshold,
        "loiter_seconds": z.loiter_seconds,
        "allowed_direction": z.allowed_direction,
    }


def _zone_summary(z: Zone) -> str:
    parts = [z.name]
    if z.restricted:
        parts.append("restricted")
    if z.crowd_threshold:
        parts.append(f"crowd>={z.crowd_threshold}")
    if z.loiter_seconds:
        parts.append(f"loiter {z.loiter_seconds}s")
    return " ".join(parts)


def _get_zones(cam_id: str, fmt: str) -> list:
    camera = app_state.manager.get(cam_id)
    if camera is None:
        return []
    width, height = camera.frame_size()
    zones = camera.pipeline.zone_store.list()
    if fmt == "px" and width and height:
        zones = [z.to_pixels(width, height) for z in zones]
    else:
        zones = [z.to_normalized(width, height) for z in zones]
    return [_zone_dict(z) for z in zones]


def _set_zones(cam_id: str, zones: List[ZoneIn]) -> dict:
    camera = _camera_or_404(cam_id)
    width, height = camera.frame_size()
    parsed = [
        Zone(
            id=z.id,
            name=z.name,
            polygon=[tuple(p) for p in z.polygon],
            restricted=z.restricted,
            crowd_threshold=z.crowd_threshold,
            loiter_seconds=z.loiter_seconds,
            allowed_direction=tuple(z.allowed_direction) if z.allowed_direction else None,
        ).to_normalized(width, height)
        for z in zones
    ]
    camera.pipeline.zone_store.replace_all(parsed)
    if parsed:
        audit("ZONES_SAVED", f"{camera.code}: " + ", ".join(_zone_summary(z) for z in parsed),
              actor="operator", camera_id=cam_id)
    else:
        audit("ZONES_CLEARED", f"{camera.code}: all zones removed", actor="operator", camera_id=cam_id)
    return {"ok": True, "count": len(parsed)}


@app.get("/api/zones")
def get_zones(format: str = "normalized"):
    """Camera 1's zones as normalized 0-1 polygons. `?format=px` returns frame
    pixels (used by the legacy dashboards)."""
    return _get_zones(PRIMARY_ID, format)


@app.post("/api/zones")
def set_zones(zones: List[ZoneIn]):
    """Replaces camera 1's full zone list. Polygons may be normalized (0-1)
    or, from older clients, frame pixels."""
    if app_state.primary() is None:
        raise HTTPException(status_code=400, detail="Start a video source first")
    return _set_zones(PRIMARY_ID, zones)


@app.get("/api/cameras/{cam_id}/zones")
def get_camera_zones(cam_id: str, format: str = "normalized"):
    _camera_or_404(cam_id)
    return _get_zones(cam_id, format)


@app.post("/api/cameras/{cam_id}/zones")
def set_camera_zones(cam_id: str, zones: List[ZoneIn]):
    return _set_zones(cam_id, zones)


# ---------- thresholds (apply to every camera) ----------
class ThresholdsIn(BaseModel):
    loiter_seconds: Optional[float] = None
    crowd_threshold: Optional[int] = None
    unattended_seconds: Optional[float] = None
    unattended_radius_px: Optional[float] = None
    stationary_speed_px_s: Optional[float] = None
    wrong_direction_angle_deg: Optional[float] = None
    surge_min_increase: Optional[int] = Field(None, ge=1)
    surge_window_s: Optional[float] = Field(None, gt=0)
    surge_avg_multiplier: Optional[float] = Field(None, gt=1)
    surge_min_people: Optional[int] = Field(None, ge=1)


@app.get("/api/thresholds")
def get_thresholds():
    return app_state.thresholds()


@app.post("/api/thresholds")
def set_thresholds(thresholds: ThresholdsIn):
    values = thresholds.dict(exclude_none=True)
    app_state.manager.update_thresholds(save=True, **values)
    audit("THRESHOLDS_CHANGED", ", ".join(f"{k}={v}" for k, v in values.items()), actor="operator")
    return app_state.thresholds()


# ---------- audit log ----------
class AuditIn(BaseModel):
    action: str = Field(..., min_length=1, max_length=64)
    detail: str = Field("", max_length=2000)
    camera_id: Optional[str] = None
    alert_id: Optional[int] = None


@app.post("/api/audit")
def post_audit(entry: AuditIn):
    """Operator events from the dashboard (the log itself is append-only)."""
    return audit(entry.action, entry.detail, actor="operator",
                 camera_id=entry.camera_id, alert_id=entry.alert_id) or {"ok": False}


@app.get("/api/audit")
def list_audit(limit: int = 200, action: Optional[str] = None, camera_id: Optional[str] = None):
    return get_audit().list(limit=limit, action=action, camera_id=camera_id)


@app.get("/api/audit/verify")
def verify_audit():
    return get_audit().verify()


@app.get("/api/audit/export")
def export_audit(format: str = "csv"):
    if format not in ("csv", "json"):
        raise HTTPException(status_code=400, detail="format must be csv or json")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return Response(
        get_audit().export(format),
        media_type="text/csv" if format == "csv" else "application/json",
        headers={"Content-Disposition": f'attachment; filename="vigil-audit-{stamp}.{format}"'},
    )


# ---------- cameras ----------
def _camera_or_404(cam_id: str) -> Camera:
    camera = app_state.manager.get(cam_id)
    if camera is None:
        raise HTTPException(status_code=404, detail=f"No camera {cam_id}")
    return camera


class CameraCreate(BaseModel):
    name: str = ""
    source: str
    location: Optional[dict] = None
    start: bool = True


class CameraPatch(BaseModel):
    name: Optional[str] = None
    location: Optional[dict] = None


@app.get("/api/cameras")
def list_cameras():
    return [c.to_dict() for c in app_state.manager.list()]


@app.post("/api/cameras")
def add_camera(body: CameraCreate):
    try:
        camera = app_state.manager.add(body.name, body.source, body.location, start=body.start)
    except (CameraLimitError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return camera.to_dict()


class SourceTest(BaseModel):
    source: str


@app.post("/api/cameras/test")
def test_camera_source(body: SourceTest):
    """Grab one frame (within CAMERA_TEST_TIMEOUT_S) and return a thumbnail.
    Refuses a phone that is already a camera: DroidCam serves one client."""
    host = _stream_host(body.source)
    if host and host in _hosts_in_use():
        return {"ok": False, "error": f"{host} is already connected as a camera (DroidCam allows one client)"}
    return test_source(body.source)


@app.get("/api/cameras/discover")
def discover_phones():
    """Scan the local /24 in parallel for DroidCam (4747) / IP Webcam (8080)."""
    result = discover(in_use_hosts=_hosts_in_use())
    audit("CAMERA_DISCOVERY", f"{len(result['found'])} phone(s) found on {', '.join(result['subnets']) or 'no network'}",
          actor="operator")
    return result


def _stream_host(source: str) -> Optional[str]:
    m = re.match(r"^[a-z]+://([^/:]+)", str(source).strip(), re.I)
    return m.group(1) if m else None


def _hosts_in_use() -> set:
    return {h for h in (_stream_host(c.source) for c in app_state.manager.list() if isinstance(c.source, str)) if h}


# ---------- site plan + camera links ----------
SITEPLAN_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".svg": "image/svg+xml",
                  ".webp": "image/webp"}


def _siteplan_file() -> Optional[Path]:
    for ext in SITEPLAN_TYPES:
        path = config.SITEPLAN_BASENAME.with_suffix(ext)
        if path.exists():
            return path
    return None


@app.get("/api/siteplan")
def get_siteplan():
    path = _siteplan_file()
    if path is None:
        raise HTTPException(status_code=404, detail="No site plan uploaded (the dashboard uses its built-in plan)")
    return FileResponse(path, media_type=SITEPLAN_TYPES[path.suffix], headers={"Cache-Control": "no-store"})


@app.post("/api/siteplan")
async def upload_siteplan(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SITEPLAN_TYPES:
        raise HTTPException(status_code=400, detail=f"Site plan must be one of {', '.join(SITEPLAN_TYPES)}")
    old = _siteplan_file()
    if old:
        old.unlink()
    dest = config.SITEPLAN_BASENAME.with_suffix(ext)
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    audit("SITEPLAN_UPLOADED", f"{file.filename} ({dest.stat().st_size} bytes)", actor="operator")
    return {"ok": True}


@app.delete("/api/siteplan")
def delete_siteplan():
    path = _siteplan_file()
    if path:
        path.unlink()
        audit("SITEPLAN_REMOVED", "back to the built-in plan", actor="operator")
    return {"ok": True}


class LinkIn(BaseModel):
    a: str
    b: str
    seconds: float = Field(..., gt=0, le=3600)


@app.get("/api/links")
def list_links():
    return app_state.manager.links


@app.post("/api/links")
def save_link(body: LinkIn):
    try:
        return app_state.manager.set_link(body.a, body.b, body.seconds)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"No camera {exc.args[0]}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.delete("/api/links/{link_id}")
def delete_link(link_id: str):
    try:
        app_state.manager.remove_link(link_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="No such link")
    return {"ok": True}


@app.patch("/api/cameras/{cam_id}")
def patch_camera(cam_id: str, body: CameraPatch):
    _camera_or_404(cam_id)
    try:
        return app_state.manager.update(cam_id, name=body.name, location=body.location).to_dict()
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Bad location: {exc}")


# ---------- location: site, address search, phone-GPS locate links ----------
@app.get("/api/site")
def get_site():
    return {"site": app_state.manager.site}


class SiteIn(BaseModel):
    lat: float
    lng: float
    label: str = ""
    zoom: Optional[int] = None
    accuracy_m: Optional[float] = None
    source: str = ""


@app.put("/api/site")
def put_site(body: SiteIn):
    try:
        return {"site": app_state.manager.set_site(body.model_dump())}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.delete("/api/site")
def delete_site():
    return {"site": app_state.manager.set_site(None)}


@app.get("/api/geocode")
def geocode(q: str, limit: int = 5):
    """Address / place search via OpenStreetMap Nominatim (proxied: proper
    User-Agent, max 1 request/s, cached)."""
    try:
        return {"results": geo.geocode(q, limit=max(1, min(limit, 10))),
                "attribution": "Search by OpenStreetMap Nominatim, data (c) OpenStreetMap contributors (ODbL)"}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Address search unavailable (needs internet): {exc}")


@app.get("/api/cameras/{cam_id}/locate")
def camera_locate_link(cam_id: str):
    """The phone-GPS link for this camera, on the laptop's LAN IP (never
    127.0.0.1), plus its QR code as inline SVG."""
    camera = _camera_or_404(cam_id)
    ip = geo.lan_ip()
    url = geo.locate_url(camera.code, camera.id, ip)
    if url is None:
        raise HTTPException(status_code=503, detail="No network: connect the laptop to the phone's Wi-Fi/hotspot first")
    return {"url": url, "lan_ip": ip, "https": bool(getattr(app.state, "locate_base", None)),
            "qr_svg": geo.qr_svg(url)}


@app.delete("/api/cameras/{cam_id}")
def delete_camera(cam_id: str):
    _camera_or_404(cam_id)
    app_state.manager.remove(cam_id)
    return {"ok": True}


@app.post("/api/cameras/{cam_id}/start")
def start_camera(cam_id: str):
    camera = _camera_or_404(cam_id)
    camera.start()
    audit("CAMERA_STARTED", f"{camera.title}", actor="operator", camera_id=cam_id)
    return camera.to_dict()


@app.post("/api/cameras/{cam_id}/stop")
def stop_camera(cam_id: str):
    camera = _camera_or_404(cam_id)
    camera.stop()
    audit("CAMERA_STOPPED", f"{camera.title}", actor="operator", camera_id=cam_id)
    return camera.to_dict()


class SourceIn(BaseModel):
    source: str


def _wait_online(camera: Camera, source) -> dict:
    deadline = time.time() + config.SOURCE_CONNECT_TIMEOUT_S
    while time.time() < deadline and camera.status == STATUS_CONNECTING:
        time.sleep(0.1)
    if camera.status != STATUS_ONLINE:
        reason = camera.last_error or f"Could not open video source: {source}"
        camera.stop()
        raise HTTPException(status_code=400, detail=reason)
    return camera.to_dict()


@app.post("/api/cameras/{cam_id}/source")
def set_camera_source(cam_id: str, body: SourceIn):
    """Re-point one camera (the dashboard's focused camera) at a new source."""
    _camera_or_404(cam_id)
    try:
        camera = app_state.manager.set_source(cam_id, body.source)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _wait_online(camera, body.source)


@app.post("/api/cameras/demo")
def start_camera_demo():
    """Fills the free camera slots with the demo cameras (real footage, real
    analysis): Main Gate, Lobby, Parking, Corridor B. Webcam 0 becomes Main
    Gate when one is attached."""
    existing = {c.name for c in app_state.manager.list()}
    added, skipped = [], []
    for preset in config.DEMO_CAMERAS:
        if preset["name"] in existing:
            skipped.append(preset["name"])
            continue
        if len(app_state.manager.cameras) >= config.MAX_CAMERAS:
            skipped.append(preset["name"])
            continue
        source = preset["source"]
        if preset.get("webcam_first") and test_source(0, timeout_s=2.0).get("ok"):
            source = "0"
        camera = app_state.manager.add(preset["name"], source, preset["location"])
        added.append(camera.to_dict())
    audit("DEMO_STARTED", f"added {', '.join(c['name'] for c in added) or 'nothing'}"
                          + (f"; skipped {', '.join(skipped)}" if skipped else ""), actor="operator")
    return {"added": added, "skipped": skipped}


@app.post("/api/cameras/upload")
async def create_camera_from_upload(file: UploadFile = File(...), name: str = "", location: str = ""):
    """LOAD FOOTAGE on an empty slot: create a camera that plays the file."""
    dest = _save_upload(file)
    try:
        camera = app_state.manager.add(name or Path(file.filename or "footage").stem, str(dest),
                                       {"label": location} if location else None)
    except (CameraLimitError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return camera.to_dict()


def _save_upload(file: UploadFile) -> Path:
    suffix = Path(file.filename or "").suffix or ".mp4"
    dest = config.UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    return dest


@app.post("/api/cameras/{cam_id}/upload")
async def upload_camera_footage(cam_id: str, file: UploadFile = File(...)):
    _camera_or_404(cam_id)
    dest = _save_upload(file)
    return app_state.manager.set_source(cam_id, str(dest)).to_dict()


# ---------- single-source flow (camera 1) ----------
@app.get("/api/samples")
def list_samples():
    if not config.SAMPLE_DATA_DIR.exists():
        return []
    names = []
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        names += [p.name for p in config.SAMPLE_DATA_DIR.glob(ext)]
    return sorted(names)


def _switch_primary(source) -> dict:
    """Point camera 1 at a source and wait briefly for its first frame, so the
    dashboard gets a clear error instead of a silent black feed."""
    try:
        camera = app_state.switch_source(source)
    except (CameraLimitError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    deadline = time.time() + config.SOURCE_CONNECT_TIMEOUT_S
    while time.time() < deadline and camera.status == STATUS_CONNECTING:
        time.sleep(0.1)
    if camera.status != STATUS_ONLINE:
        reason = camera.last_error or f"Could not open video source: {source}"
        camera.stop()
        raise HTTPException(status_code=400, detail=reason)
    return app_state.status()


class WebcamIn(BaseModel):
    index: int = 0


@app.post("/api/source/webcam")
def switch_webcam(body: WebcamIn):
    return _switch_primary(body.index)


class CameraIn(BaseModel):
    source: str


@app.post("/api/source/camera")
def switch_camera(body: CameraIn):
    """A device index ("0", "1", ...) or a network stream URL, e.g. DroidCam
    over Wi-Fi (http://<phone-ip>:4747/video)."""
    return _switch_primary(body.source.strip())


class SampleIn(BaseModel):
    name: str


@app.post("/api/source/sample")
def switch_sample(body: SampleIn):
    path = config.SAMPLE_DATA_DIR / body.name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Sample not found")
    return _switch_primary(str(path))


@app.post("/api/source/upload")
async def upload_source(file: UploadFile = File(...)):
    return _switch_primary(str(_save_upload(file)))
