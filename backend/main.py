"""FastAPI application: serves the analyst dashboard, streams annotated video
per camera, and exposes the cameras/alerts/zones/source/threshold/audit REST
API. The single-source endpoints (/video_feed, /api/source/*, /api/zones)
act on camera 1.
"""
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
from backend.zones import Zone

FRONTEND_DIR = config.BASE_DIR / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # No auto-started source on boot -- the dashboard opens idle; saved
    # cameras are restored STOPPED and the analyst starts what they need.
    audit("SERVER_START", f"device {config.DEVICE}, weapon model {config.WEAPON_MODEL_PATH.name}, "
                          f"{len(app_state.manager.cameras)} saved camera(s) restored stopped")
    yield
    app_state.stop()
    audit("SERVER_STOP", "shutdown")


app = FastAPI(title="Vigil Threat Detection & Situational Awareness", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


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


@app.get("/api/status")
def get_status():
    return app_state.status()


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
    app_state.manager.update_thresholds(**values)
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


@app.patch("/api/cameras/{cam_id}")
def patch_camera(cam_id: str, body: CameraPatch):
    _camera_or_404(cam_id)
    return app_state.manager.update(cam_id, name=body.name, location=body.location).to_dict()


@app.delete("/api/cameras/{cam_id}")
def delete_camera(cam_id: str):
    _camera_or_404(cam_id)
    app_state.manager.remove(cam_id)
    return {"ok": True}


@app.post("/api/cameras/{cam_id}/start")
def start_camera(cam_id: str):
    camera = _camera_or_404(cam_id)
    camera.start()
    audit("CAMERA_STARTED", f"{camera.code} {camera.name}", actor="operator", camera_id=cam_id)
    return camera.to_dict()


@app.post("/api/cameras/{cam_id}/stop")
def stop_camera(cam_id: str):
    camera = _camera_or_404(cam_id)
    camera.stop()
    audit("CAMERA_STOPPED", f"{camera.code} {camera.name}", actor="operator", camera_id=cam_id)
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
        camera.stop()
        raise HTTPException(status_code=400, detail=f"Could not open video source: {source}")
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
