"""FastAPI app: serves the dashboard, the MJPEG stream and the REST API.

Run from the repo root:  python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend import config
from backend.pipeline import Pipeline

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
# PROVISIONAL: source control so the skeleton can be exercised end to end.
# Request/response shapes must be replaced to match docs/api.md once it exists.
# ---------------------------------------------------------------------------

class SourceRequest(BaseModel):
    kind: str  # "webcam" | "file" | "url"
    value: str | int = 0


@app.post("/api/source/start")
def source_start(req: SourceRequest):
    if req.kind not in ("webcam", "file", "url"):
        raise HTTPException(400, "kind must be webcam, file or url")
    value = req.value
    if req.kind == "file":
        path = config.ROOT / "test_videos" / str(value)
        value = str(path if path.is_file() else value)
    try:
        pipeline.start(req.kind, value)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(400, str(exc))
    return {"running": True, "source": pipeline.source.label}


@app.post("/api/source/stop")
def source_stop():
    pipeline.stop()
    return {"running": False}
