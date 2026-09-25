"""FastAPI app: serves the dashboard, the MJPEG stream and the REST API.

Run from the repo root:  python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""
import asyncio
import json
import logging
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend import config
from backend.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s - %(message)s")
log = logging.getLogger("backend.api")

pipeline: Pipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pipeline = Pipeline()
    pipeline.load()
    # No source is auto-started: the server boots idle and waits for the user to pick one
    yield
    pipeline.stop()


app = FastAPI(title="CCTV Threat Detection", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def index():
    page = config.FRONTEND_DIR / "index.html"
    if page.is_file():
        return FileResponse(page)
    return HTMLResponse("<h3>Backend running</h3><img src='/video_feed' style='max-width:100%'>")


if config.FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=config.FRONTEND_DIR), name="static")


async def _mjpeg():
    last_id = -1
    min_interval = 1.0 / config.STREAM_MAX_FPS
    while True:
        frame_id, jpeg = pipeline.latest_jpeg()
        if frame_id != last_id and jpeg:
            last_id = frame_id
            yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                   + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
        await asyncio.sleep(min_interval)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        _mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


# ---------------------------------------------------------------------------
# REST API - contract is whatever frontend/index.html calls (see docs/api.md).
# Every error is 400/404 {"detail": "..."}, including body validation errors.
# ---------------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def _validation_error(request, exc: RequestValidationError):
    parts = []
    for e in exc.errors():
        where = ".".join(str(p) for p in e.get("loc", ()) if p != "body")
        parts.append(f"{where}: {e['msg']}" if where else e["msg"])
    return JSONResponse({"detail": "; ".join(parts) or "invalid request"}, status_code=400)


@app.get("/api/status")
def status():
    return pipeline.status()


def _start(kind: str, value, label: str) -> dict:
    try:
        pipeline.start(kind, value, label)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(400, str(exc))
    return pipeline.status()


def _redact(url: str) -> str:
    """Hide user:password in camera URLs before showing them on the dashboard."""
    parts = urlsplit(url)
    if "@" not in parts.netloc:
        return url
    return urlunsplit(parts._replace(netloc="***@" + parts.netloc.rsplit("@", 1)[1]))


class CameraRequest(BaseModel):
    source: str | int


@app.post("/api/source/camera")
def source_camera(req: CameraRequest):
    value = str(req.source).strip()
    if value.isdigit():
        return _start("webcam", int(value), f"Webcam {int(value)}")
    if re.match(r"^(https?|rtsps?|rtmp)://\S+$", value, re.IGNORECASE):
        return _start("url", value, f"camera: {_redact(value)}")
    raise HTTPException(400, "source must be a webcam index like 0, or a stream URL "
                             "like http://PHONE-IP:4747/video or rtsp://...")


def _list_samples() -> list[str]:
    if not config.SAMPLES_DIR.is_dir():
        return []
    return sorted(p.name for p in config.SAMPLES_DIR.iterdir()
                  if p.is_file() and p.suffix.lower() in config.VIDEO_EXTENSIONS)


@app.get("/api/source/samples")
def source_samples():
    return _list_samples()


class SampleRequest(BaseModel):
    name: str


@app.post("/api/source/sample")
def source_sample(req: SampleRequest):
    name = req.name.strip()
    # A bare filename only: no directories, drive letters or ".." (path traversal)
    if not name or name in (".", "..") or Path(name).name != name or "/" in name or "\\" in name:
        raise HTTPException(400, "name must be a plain file name from /api/source/samples")
    samples = _list_samples()
    if name not in samples:
        raise HTTPException(400, f"sample '{name}' not found; available: {', '.join(samples) or 'none'}")
    return _start("file", str(config.SAMPLES_DIR / name), f"sample: {name}")


@app.post("/api/source/upload")
def source_upload(file: UploadFile = File(...)):
    original = Path(file.filename or "").name
    suffix = Path(original).suffix.lower()
    if suffix not in config.VIDEO_EXTENSIONS:
        raise HTTPException(400, f"unsupported file type '{suffix or original}'; "
                                 f"use {', '.join(sorted(config.VIDEO_EXTENSIONS))}")
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", Path(original).stem)[:60] or "video"
    config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    dest = config.UPLOADS_DIR / f"{uuid.uuid4().hex[:8]}_{stem}{suffix}"
    size = 0
    try:
        with dest.open("wb") as out:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > config.MAX_UPLOAD_BYTES:
                    raise HTTPException(400, "file too large (limit 1 GB)")
                out.write(chunk)
        if size == 0:
            raise HTTPException(400, "uploaded file is empty")
        return _start("file", str(dest), f"upload: {original}")
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise


@app.post("/api/source/stop")
def source_stop():
    pipeline.stop()
    return pipeline.status()


@app.get("/api/alerts")
def alerts(limit: int = Query(50, ge=1, le=config.ALERT_MAX)):
    return pipeline.alerts.list(limit)


@app.delete("/api/alerts")
def clear_alerts():
    pipeline.alerts.clear()
    return {"ok": True, "alert_count": 0}


@app.get("/api/alerts/{alert_id}/evidence")
def alert_evidence(alert_id: str):
    # str, not int: the dashboard's simulated alerts use ids like "demo-123" -> 404, not 400
    jpeg = pipeline.alerts.evidence(int(alert_id)) if alert_id.isdigit() else None
    if not jpeg:
        raise HTTPException(404, "no evidence for this alert")
    return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/zones")
def get_zones():
    return pipeline.zones.as_dicts()


@app.post("/api/zones")
def set_zones(zones: Any = Body(...)):
    try:
        return pipeline.zones.replace(zones)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/thresholds")
def get_thresholds():
    return pipeline.thresholds.get()


@app.post("/api/thresholds")
def set_thresholds(values: Any = Body(...)):
    try:
        return pipeline.thresholds.update(values)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/audit")
def audit(entry: Any = Body(None)):
    log.info("audit %s", json.dumps(entry, default=str)[:500])
    return {"ok": True}
