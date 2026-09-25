"""Phone GPS -> camera position.

Scan a camera tile's QR code with the phone standing where that camera is:
/locate/CAM-02 asks for GPS and posts the fix here, and the camera is placed
("Located via phone GPS ±8 m"). The same router is served by the main app
and by a small HTTPS-only app on LOCATE_HTTPS_PORT (phones give GPS only to
secure pages); that app exposes nothing but these two routes.
"""
import threading
from typing import Optional

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from backend import config, geo
from backend.app_state import app_state
from backend.audit import audit

router = APIRouter()


def camera_by_code(code: str):
    """CAM-02, cam-2 or 2 -> the camera."""
    text = code.strip().lower()
    for cam in app_state.manager.list():
        if text in (cam.id, cam.code.lower(), str(cam.number)):
            return cam
    return None


@router.get("/locate/{code}", response_class=HTMLResponse)
def locate_page(code: str):
    cam = camera_by_code(code)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"No camera {code}")
    html = (config.BASE_DIR / "frontend" / "locate.html").read_text(encoding="utf-8")
    return html.replace("{{CODE}}", cam.code).replace("{{NAME}}", cam.name.replace("<", "&lt;"))


class GpsFix(BaseModel):
    lat: float
    lng: float
    accuracy: Optional[float] = None
    k: str = ""


@router.post("/api/locate/{code}")
def locate_post(code: str, fix: GpsFix):
    cam = camera_by_code(code)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"No camera {code}")
    if not geo.token_ok(cam.id, fix.k):
        raise HTTPException(status_code=403, detail="This link is not valid for this camera - scan the QR code on its tile again")
    try:
        lat, lng = geo.normalize_latlng(fix.lat, fix.lng)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    acc = round(fix.accuracy, 1) if fix.accuracy is not None else None
    source = "gps" if acc is not None else "manual"
    app_state.manager.update(cam.id, location={"lat": lat, "lng": lng, "accuracy_m": acc, "source": source})
    message = (f"Located via phone GPS {geo.describe_accuracy(acc)}" if acc is not None
               else "Placed from pasted coordinates")
    audit("CAMERA_LOCATED", f"{cam.code} {cam.name}: {message} ({lat:.6f}, {lng:.6f})", actor="phone", camera_id=cam.id)
    return {"ok": True, "camera": cam.code, "lat": lat, "lng": lng, "accuracy_m": acc, "message": message}


def start_https_server() -> Optional[str]:
    """Serve only the locate routes over HTTPS on LOCATE_HTTPS_PORT, on all
    interfaces, in a background thread. Returns the base URL, or None."""
    if not config.LOCATE_HTTPS_PORT:
        return None
    try:
        import uvicorn

        ip = geo.lan_ip()
        crt, key = geo.ensure_cert(ip)
        locate_app = FastAPI(title="VIGIL locate", docs_url=None, redoc_url=None, openapi_url=None)
        locate_app.include_router(router)
        server = uvicorn.Server(uvicorn.Config(
            locate_app, host="0.0.0.0", port=config.LOCATE_HTTPS_PORT, log_level="warning",
            ssl_certfile=str(crt), ssl_keyfile=str(key)))
        server.install_signal_handlers = lambda: None   # the main server owns Ctrl+C
        threading.Thread(target=server.run, name="locate-https", daemon=True).start()
        return f"https://{ip or '127.0.0.1'}:{config.LOCATE_HTTPS_PORT}"
    except Exception as exc:  # never block the main server over this
        print(f"[locate] HTTPS locate server not started: {exc}")
        return None
