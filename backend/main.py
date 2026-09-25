"""FastAPI application: serves the analyst dashboard, streams the annotated
video feed, and exposes the alerts/zones/source/threshold REST API.
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
from backend.app_state import app_state
from backend.audit import audit, get_audit
from backend.zones import Zone

FRONTEND_DIR = config.BASE_DIR / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # No auto-started source on boot -- the dashboard opens idle, and the
    # analyst explicitly picks Webcam / Connect / Upload / a sample clip.
    audit("SERVER_START", f"device {config.DEVICE}, weapon model {config.WEAPON_MODEL_PATH.name}")
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


def _mjpeg_generator():
    boundary = b"--frame"
    delay = 1.0 / config.STREAM_MAX_FPS
    while True:
        jpeg = app_state.latest_jpeg()
        if jpeg is not None:
            yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(delay)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        _mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/status")
def get_status():
    return app_state.status()


@app.get("/api/alerts")
def get_alerts(limit: int = 50):
    return [a.to_dict() for a in app_state.pipeline.alert_manager.ranked(limit)]


@app.delete("/api/alerts")
def clear_alerts():
    """Clear the alert feed (analyst acknowledged). Cooldowns are kept, so a
    still-ongoing event doesn't instantly re-fire the moment it's cleared."""
    cleared = app_state.pipeline.alert_manager.clear()
    audit("ALERTS_CLEARED", f"{cleared} alert(s) cleared from the feed", actor="operator")
    return {"ok": True, "cleared": cleared}


class AckIn(BaseModel):
    who: str = "operator"


@app.post("/api/alerts/{alert_id}/ack")
def acknowledge_alert(alert_id: int, body: Optional[AckIn] = None):
    who = (body.who if body else "operator")[:64]
    alert = app_state.pipeline.alert_manager.acknowledge(alert_id, who)
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert.to_dict()


@app.get("/api/alerts/{alert_id}/evidence")
def get_evidence(alert_id: int):
    alert = app_state.pipeline.alert_manager.get(alert_id)
    if alert is None or not alert.evidence_path or not Path(alert.evidence_path).exists():
        raise HTTPException(status_code=404, detail="No evidence available for this alert")
    return FileResponse(alert.evidence_path, media_type="image/jpeg")


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


@app.get("/api/zones")
def get_zones(format: str = "normalized"):
    """Zones as normalized 0-1 polygons. `?format=px` returns frame pixels
    for the current source (used by the legacy dashboards)."""
    width, height = app_state.frame_size()
    zones = app_state.pipeline.zone_store.list()
    if format == "px" and width and height:
        zones = [z.to_pixels(width, height) for z in zones]
    else:
        zones = [z.to_normalized(width, height) for z in zones]
    return [_zone_dict(z) for z in zones]


@app.post("/api/zones")
def set_zones(zones: List[ZoneIn]):
    """Replaces the full zone list. Polygons may be normalized (0-1) or, from
    older clients, frame pixels; pixels are normalized against the current
    source when one is running."""
    width, height = app_state.frame_size()
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
    app_state.pipeline.zone_store.replace_all(parsed)
    if parsed:
        audit("ZONES_SAVED", ", ".join(_zone_summary(z) for z in parsed), actor="operator")
    else:
        audit("ZONES_CLEARED", "all zones removed", actor="operator")
    return {"ok": True, "count": len(parsed)}


def _zone_summary(z: Zone) -> str:
    parts = [z.name]
    if z.restricted:
        parts.append("restricted")
    if z.crowd_threshold:
        parts.append(f"crowd>={z.crowd_threshold}")
    if z.loiter_seconds:
        parts.append(f"loiter {z.loiter_seconds}s")
    return " ".join(parts)


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
    engine = app_state.pipeline.rule_engine
    return {
        "loiter_seconds": engine.loiter_seconds,
        "crowd_threshold": engine.crowd_threshold,
        "unattended_seconds": engine.unattended_seconds,
        "unattended_radius_px": engine.unattended_radius_px,
        "stationary_speed_px_s": engine.stationary_speed_px_s,
        "wrong_direction_angle_deg": engine.wrong_direction_angle_deg,
        "surge_min_increase": engine.surge_min_increase,
        "surge_window_s": engine.surge_window_s,
        "surge_avg_multiplier": engine.surge_avg_multiplier,
        "surge_min_people": engine.surge_min_people,
    }


@app.post("/api/thresholds")
def set_thresholds(thresholds: ThresholdsIn):
    values = thresholds.dict(exclude_none=True)
    app_state.pipeline.rule_engine.update_thresholds(**values)
    audit("THRESHOLDS_CHANGED", ", ".join(f"{k}={v}" for k, v in values.items()), actor="operator")
    return get_thresholds()


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


@app.get("/api/samples")
def list_samples():
    if not config.SAMPLE_DATA_DIR.exists():
        return []
    names = []
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        names += [p.name for p in config.SAMPLE_DATA_DIR.glob(ext)]
    return sorted(names)


class WebcamIn(BaseModel):
    index: int = 0


@app.post("/api/source/webcam")
def switch_webcam(body: WebcamIn):
    try:
        app_state.switch_source(body.index, f"webcam:{body.index}")
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return app_state.status()


class CameraIn(BaseModel):
    source: str


@app.post("/api/source/camera")
def switch_camera(body: CameraIn):
    """Accepts either a bare device index ("0", "1", ...) for a local/USB
    camera -- e.g. DroidCam's USB mode, which registers as another webcam
    index -- or a network stream URL, e.g. DroidCam's WiFi mode
    (http://<phone-ip>:4747/video)."""
    raw = body.source.strip()
    source: object = int(raw) if raw.isdigit() else raw
    try:
        app_state.switch_source(source, f"camera:{raw}")
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return app_state.status()


class SampleIn(BaseModel):
    name: str


@app.post("/api/source/sample")
def switch_sample(body: SampleIn):
    path = config.SAMPLE_DATA_DIR / body.name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Sample not found")
    app_state.switch_source(str(path), f"sample:{body.name}")
    return app_state.status()


@app.post("/api/source/upload")
async def upload_source(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix or ".mp4"
    dest = config.UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    try:
        app_state.switch_source(str(dest), f"upload:{file.filename}")
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return app_state.status()
