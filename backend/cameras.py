"""Multi-camera control room.

N independent cameras, each with its own capture thread, ByteTrack tracker,
zones, rules state (loitering, intrusion, crowd surge), weapon 5-of-8 filter
and alert cooldowns. There is ONE general YOLO model and ONE weapon model
(run15), shared by every camera through a single lock, so GPU memory does
not grow with the number of cameras.

Capture keeps only the latest frame (stale frames are dropped for low
latency). A live stream that drops is retried every 3s with backoff up to
30s and resumes on its own; other cameras are never affected. File sources
play at their native frame rate and loop.
"""
import json
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import cv2
import numpy as np

from backend import config, features
from backend.alerts.manager import AlertManager
from backend.audit import audit
from backend.detection.object_detector import Detection
from backend.detection.weapon_detector import WeaponDetector
from backend.geo import normalize_latlng
from backend.pipeline import FramePipeline
from backend.zones import ZoneStore

Source = Union[int, str]

STATUS_STOPPED = "stopped"
STATUS_CONNECTING = "connecting"
STATUS_ONLINE = "online"
STATUS_RECONNECTING = "reconnecting"
STATUS_OFFLINE = "offline"


class CameraLimitError(Exception):
    pass


# ---------------------------------------------------------------------------
# Shared models
# ---------------------------------------------------------------------------
class SharedModels:
    """One general detector + one weapon model for all cameras. Inference is
    serialized by a lock: the GPU runs one frame at a time anyway, and the
    lock keeps ultralytics' predictor state from being shared across threads."""

    def __init__(self):
        self._lock = threading.Lock()
        self._general = None
        self._pose = None
        self.weapon = WeaponDetector()

    def _ensure_general(self):
        if self._general is None:
            from ultralytics import YOLO

            self._general = YOLO(config.GENERAL_MODEL_PATH)
            audit("MODEL_LOADED", f"general detector: {config.GENERAL_MODEL_PATH}")

    def detect(self, frame):
        """Raw detections as an ultralytics Boxes array (numpy) + class names."""
        with self._lock:
            self._ensure_general()
            low = config.THROW_LOW_CONF if features.enabled("throwing") else config.DETECTOR_CONF_THRESHOLD
            result = self._general.predict(
                frame,
                conf=min(low, config.DETECTOR_CONF_THRESHOLD),
                device=config.DEVICE,
                half=config.USE_HALF_PRECISION,
                verbose=False,
            )[0]
        return result.boxes.cpu().numpy(), result.names

    def weapons(self, frame):
        with self._lock:
            return self.weapon.infer(frame)

    def pose_people(self, frame):
        """(17, 3) keypoint arrays for every person in the frame; used to
        check weapon candidates against faces / torsos (weapon_verify.py)."""
        with self._lock:
            if self._pose is None:
                from ultralytics import YOLO

                self._pose = YOLO(config.POSE_MODEL)
                audit("MODEL_LOADED", f"weapon pose check: {config.POSE_MODEL}")
            result = self._pose.predict(frame, conf=config.POSE_CONF, device=config.DEVICE,
                                        half=config.USE_HALF_PRECISION, verbose=False)[0]
        if result.keypoints is None or len(result.keypoints) == 0:
            return []
        return list(result.keypoints.data.cpu().numpy())


class CameraObjectDetector:
    """Per-camera ByteTrack over the shared model's raw detections, so track
    ids never leak between cameras."""

    def __init__(self, shared: SharedModels, frame_rate: int):
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import IterableSimpleNamespace, yaml_load
        from ultralytics.utils.checks import check_yaml

        self.shared = shared
        cfg = IterableSimpleNamespace(**yaml_load(check_yaml(config.TRACKER_CONFIG)))
        self.tracker = BYTETracker(args=cfg, frame_rate=frame_rate)

    def infer(self, frame) -> List[Detection]:
        boxes, names = self.shared.detect(frame)
        if len(boxes) == 0:
            return []
        # Below the normal threshold only throwable classes survive (throw detection).
        keep = [
            i for i in range(len(boxes))
            if boxes.conf[i] >= config.DETECTOR_CONF_THRESHOLD or names.get(int(boxes.cls[i])) in config.THROW_CLASSES
        ]
        if len(keep) != len(boxes):
            boxes = boxes[keep]
            if len(boxes) == 0:
                return []
        tracks = self.tracker.update(boxes, frame)
        detections = []
        for x1, y1, x2, y2, track_id, score, cls, _ in tracks:
            detections.append(Detection(
                track_id=int(track_id),
                cls_name=names.get(int(cls), str(int(cls))),
                conf=float(score),
                bbox=(float(x1), float(y1), float(x2), float(y2)),
            ))
        return detections


class CameraWeaponDetector:
    """Runs the shared weapon model on every Nth frame of one camera. On the
    frames in between it returns None, and the pipeline leaves that camera's
    5-of-8 filter untouched (the filter counts weapon evaluations)."""

    def __init__(self, shared: SharedModels, every_n: int):
        self.shared = shared
        self.every_n = max(1, int(every_n))
        self._n = 0

    @property
    def enabled(self) -> bool:
        return self.shared.weapon.enabled

    def infer(self, frame):
        self._n += 1
        if (self._n - 1) % self.every_n:
            return None
        return self.shared.weapons(frame)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
def resolve_source(raw: Source) -> Source:
    """webcam index, DroidCam / IP Webcam URL, RTSP URL, a file path, or the
    name of a bundled sample clip."""
    if isinstance(raw, int):
        return raw
    text = str(raw).strip()
    if text.isdigit():
        return int(text)
    if re.match(r"^(https?|rtsp|rtmp)://", text, re.I):
        return text
    sample = config.SAMPLE_DATA_DIR / text
    if sample.exists():
        return str(sample)
    if Path(text).exists():
        return str(Path(text))
    raise ValueError(f"Unknown source '{text}': use a webcam index, a stream URL, or a sample clip name")


def is_file_source(source: Source) -> bool:
    return isinstance(source, str) and not re.match(r"^[a-z]+://", source, re.I)


def source_label(source: Source) -> str:
    if isinstance(source, int):
        return f"webcam:{source}"
    if is_file_source(source):
        return f"file:{Path(source).name}"
    m = re.match(r"^[a-z]+://([^/:]+)(?::(\d+))?", source, re.I)
    return f"{m.group(1)}:{m.group(2) or ''}".rstrip(":") if m else source


def open_capture(source: Source) -> cv2.VideoCapture:
    if isinstance(source, int) and sys.platform == "win32":
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
    if isinstance(source, str) and not is_file_source(source):
        # Network stream (DroidCam, IP Webcam, RTSP): bounded open/read waits,
        # so a locked phone is noticed within seconds instead of hanging read().
        ms = int(config.STREAM_TIMEOUT_S * 1000)
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG,
                               [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, ms, cv2.CAP_PROP_READ_TIMEOUT_MSEC, ms])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    return cv2.VideoCapture(source)


# ---------------------------------------------------------------------------
# Webcams: one owner per physical device
# ---------------------------------------------------------------------------
def webcam_open_error(index: int) -> str:
    return f"Webcam {index} could not be opened - close other apps using the camera (Zoom, Teams, Camera app, a browser tab)"


class _Webcam:
    """One cv2.VideoCapture on one device index, read by a single thread.
    Every camera on that index reads the latest frame from here, so the device
    is opened exactly once however many tiles / cameras show it."""

    def __init__(self, index: int, opener: Callable):
        self.index = index
        self.users: List[str] = []   # camera codes, first one is the owner
        self.error = ""
        self._opener = opener
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0
        self._running = True
        self._thread = threading.Thread(target=self._loop, name=f"webcam-{index}", daemon=True)
        self._thread.start()

    def _loop(self):
        backoff = config.RECONNECT_INITIAL_S
        while self._running:
            cap = self._opener(self.index)
            ok, frame = cap.read() if cap.isOpened() else (False, None)
            if not ok:
                cap.release()
                with self._cond:
                    self.error = webcam_open_error(self.index)
                    self._cond.notify_all()
                end = time.time() + backoff
                while self._running and time.time() < end:
                    time.sleep(0.1)
                backoff = min(backoff * 2, config.RECONNECT_MAX_S)
                continue
            backoff = config.RECONNECT_INITIAL_S
            with self._cond:
                self.error = ""
            try:
                while self._running and ok:
                    with self._cond:
                        self._frame, self._seq = frame, self._seq + 1
                        self._cond.notify_all()
                    ok, frame = cap.read()
            finally:
                cap.release()   # the device is free again the moment this thread stops
            if self._running:
                with self._cond:
                    self.error = f"Webcam {self.index} stopped sending frames - reconnecting"
                    self._cond.notify_all()

    def wait_frame(self, after_seq: int, timeout: float):
        """(frame, seq) newer than after_seq, or (None, after_seq) on timeout / error."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq > after_seq or not self._running or self.error, timeout)
            if self._seq > after_seq:
                return self._frame, self._seq
            return None, after_seq

    def latest(self):
        with self._cond:
            return self._frame

    def stop(self):
        self._running = False
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=5.0)


class WebcamHub:
    def __init__(self, opener: Callable = None):
        self._lock = threading.Lock()
        self._devices: Dict[int, _Webcam] = {}
        self._opener = opener or open_capture

    def acquire(self, index: int, user: str) -> _Webcam:
        with self._lock:
            dev = self._devices.get(index)
            if dev is None:
                dev = self._devices[index] = _Webcam(index, self._opener)
            if user not in dev.users:
                dev.users.append(user)
            return dev

    def release(self, index: int, user: str):
        with self._lock:
            dev = self._devices.get(index)
            if dev is None:
                return
            if user in dev.users:
                dev.users.remove(user)
            if dev.users:
                return
            del self._devices[index]
        dev.stop()

    def owner(self, index: int) -> Optional[str]:
        with self._lock:
            dev = self._devices.get(index)
            return dev.users[0] if dev and dev.users else None

    def get(self, index: int) -> Optional[_Webcam]:
        with self._lock:
            return self._devices.get(index)


WEBCAMS = WebcamHub()


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
@dataclass
class Location:
    label: str = ""
    lat: Optional[float] = None
    lng: Optional[float] = None
    x: Optional[float] = None   # site-plan position, normalized 0-1
    y: Optional[float] = None
    heading: Optional[float] = None  # view direction on the site plan, degrees
    accuracy_m: Optional[float] = None  # GPS / browser fix accuracy; None = placed by hand
    source: str = ""                    # how lat/lng were set: gps, map, drag, search, manual

    def to_dict(self) -> dict:
        return {"label": self.label, "lat": self.lat, "lng": self.lng, "x": self.x, "y": self.y,
                "heading": self.heading, "accuracy_m": self.accuracy_m, "source": self.source}

    @classmethod
    def from_dict(cls, d: Optional[dict], default_label: str = "") -> "Location":
        d = d or {}
        lat, lng = normalize_latlng(d.get("lat"), d.get("lng"))
        return cls(
            label=d.get("label") or default_label,
            lat=lat, lng=lng, x=d.get("x"), y=d.get("y"), heading=d.get("heading"),
            accuracy_m=d.get("accuracy_m"), source=d.get("source") or "",
        )


class Camera:
    def __init__(self, cam_id: str, name: str, source: Source, location: Location,
                 shared: SharedModels, alert_manager: AlertManager, zone_path: Path,
                 on_frame: Optional[Callable] = None):
        self.id = cam_id
        self.name = name
        self.source = source
        self.location = location
        self.shared = shared
        self.status = STATUS_STOPPED
        self.offline_since: Optional[float] = None
        self.last_error = ""
        self.webcams = WEBCAMS
        self.on_frame = on_frame  # hook for later analytics (search indexer, behavior)

        self.pipeline = FramePipeline(
            object_detector=None,
            weapon_detector=None,
            zone_store=ZoneStore(zone_path),
            alert_manager=alert_manager,
            lazy=True,
        )
        self.pipeline.camera = self

        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_seq = 0
        self._latest_jpeg: Optional[bytes] = None
        self._frame_shape = (0, 0)
        self._fps = 0.0
        self._proc_ms = 0.0
        self._running = False
        self._threads: List[threading.Thread] = []

    @property
    def number(self) -> int:
        return int(self.id.split("-")[1])

    @property
    def code(self) -> str:
        return f"CAM-{self.number:02d}"

    @property
    def title(self) -> str:
        """"CAM-01" when the name is just the code, else "CAM-02 Main Gate"."""
        name = self.name.strip()
        return self.code if not name or name.upper() == self.code else f"{self.code} {name}"

    @property
    def place(self) -> str:
        return self.location.label or self.name

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        if self._running:
            return
        # Fresh tracker/weapon state every start: a new source is a new scene.
        self.pipeline.object_detector = CameraObjectDetector(self.shared, config.CAMERA_MAX_FPS)
        self.pipeline.weapon_detector = CameraWeaponDetector(self.shared, config.WEAPON_EVERY_N_FRAMES)
        self.pipeline.pose_fn = self.shared.pose_people
        self._running = True
        self.status = STATUS_CONNECTING
        self._threads = [
            threading.Thread(target=self._capture_loop, name=f"{self.id}-capture", daemon=True),
            threading.Thread(target=self._process_loop, name=f"{self.id}-process", daemon=True),
        ]
        for t in self._threads:
            t.start()

    def stop(self):
        self._running = False
        for t in self._threads:
            t.join(timeout=3.0)
        self._threads = []
        self.status = STATUS_STOPPED
        self.offline_since = None
        with self._lock:
            self._latest_frame = None
            self._latest_jpeg = None
            self._fps = 0.0

    def set_source(self, source: Source):
        was_running = self._running
        self.stop()
        self.source = source
        if was_running:
            self.start()

    # -- capture: latest frame only, reconnect with backoff -----------------
    def _capture_loop(self):
        if isinstance(self.source, int):
            return self._webcam_loop(self.source)
        backoff = config.RECONNECT_INITIAL_S
        ever_online = False
        while self._running:
            cap = open_capture(self.source)
            ok, frame = (cap.read() if cap.isOpened() else (False, None))
            if not ok:
                cap.release()
                self.last_error = f"No video from {source_label(self.source)}"
                self._mark_down(ever_online)
                self._sleep(backoff)
                backoff = min(backoff * 2, config.RECONNECT_MAX_S)
                continue

            backoff = config.RECONNECT_INITIAL_S
            self.last_error = ""
            self._mark_up(ever_online)
            ever_online = True
            is_file = is_file_source(self.source)
            file_fps = (cap.get(cv2.CAP_PROP_FPS) or 25.0) if is_file else 0.0
            next_t = time.time()
            while self._running and ok:
                self._publish(frame)
                if is_file:
                    # Play files in real time, looping, instead of as fast as decode allows.
                    next_t += 1.0 / file_fps
                    delay = next_t - time.time()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        next_t = time.time()
                ok, frame = cap.read()
                if not ok and is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
            cap.release()
            if self._running:
                self._mark_down(ever_online)

    def _webcam_loop(self, index: int):
        """Read a shared webcam (WEBCAMS): never opens the device itself."""
        hub = self.webcams
        dev = hub.acquire(index, self.code)
        owner = dev.users[0]
        if owner != self.code:
            audit("WEBCAM_SHARED", f"{self.code} shares webcam {index} with {owner}", camera_id=self.id)
        ever_online, seq, last_frame_t = False, 0, time.time()
        try:
            while self._running:
                frame, new_seq = dev.wait_frame(seq, 0.5)   # short, so stop() is noticed quickly
                if frame is None:
                    stalled = time.time() - last_frame_t > config.STREAM_TIMEOUT_S
                    if self._running and (dev.error or stalled):
                        self.last_error = dev.error or f"Webcam {index} sent no frame for {config.STREAM_TIMEOUT_S:.0f}s"
                        self._mark_down(ever_online)
                        self._sleep(0.5)
                    continue
                last_frame_t = time.time()
                if self.status != STATUS_ONLINE:
                    self.last_error = ""
                    self._mark_up(ever_online)
                    ever_online = True
                seq = new_seq
                self._publish(frame)
        finally:
            hub.release(index, self.code)

    def _sleep(self, seconds: float):
        end = time.time() + seconds
        while self._running and time.time() < end:
            time.sleep(0.1)

    def _mark_up(self, was_online_before: bool):
        down_for = time.time() - self.offline_since if self.offline_since else 0
        self.status = STATUS_ONLINE
        self.offline_since = None
        if was_online_before:
            audit("CAMERA_RECONNECTED", f"{self.title} back after {down_for:.0f}s", camera_id=self.id)
        else:
            audit("CAMERA_ONLINE", f"{self.title} ({source_label(self.source)})", camera_id=self.id)

    def _mark_down(self, was_online: bool):
        now = time.time()
        if self.offline_since is None:
            self.offline_since = now
            self.status = STATUS_RECONNECTING
            audit("CAMERA_OFFLINE" if was_online else "CAMERA_UNREACHABLE",
                  f"{self.title} ({source_label(self.source)}); retrying", camera_id=self.id)
        elif self.status == STATUS_RECONNECTING and now - self.offline_since >= config.OFFLINE_AFTER_S:
            self.status = STATUS_OFFLINE

    def _publish(self, frame):
        with self._lock:
            self._latest_frame = frame
            self._latest_seq += 1

    # -- processing -----------------------------------------------------------
    def _process_loop(self):
        min_dt = 1.0 / config.CAMERA_MAX_FPS
        seen_seq = 0
        last = time.time()
        while self._running:
            with self._lock:
                frame, seq = self._latest_frame, self._latest_seq
            if frame is None or seq == seen_seq:
                time.sleep(0.01)
                continue
            seen_seq = seq
            t0 = time.time()
            small = resize_to_width(frame, config.PROCESS_WIDTH)
            try:
                annotated = self.pipeline.process(small)
            except Exception as exc:  # keep the camera alive if one frame errors
                print(f"[{self.id}] frame error: {exc}")
                annotated = small
            if self.on_frame:
                try:
                    self.on_frame(self, small, annotated)
                except Exception as exc:
                    print(f"[{self.id}] analytics hook error: {exc}")
            draw_camera_label(annotated, self._overlay_text(), self.code)
            ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), config.STREAM_JPEG_QUALITY])
            now = time.time()
            with self._lock:
                if ok:
                    self._latest_jpeg = buf.tobytes()
                self._frame_shape = annotated.shape[:2]
                dt = now - last
                self._fps = 0.8 * self._fps + 0.2 * (1.0 / dt) if dt > 0 else self._fps
                self._proc_ms = 0.8 * self._proc_ms + 0.2 * (now - t0) * 1000.0
            last = now
            spare = min_dt - (time.time() - t0)
            if spare > 0:
                time.sleep(spare)

    # -- read side ------------------------------------------------------------
    def _overlay_text(self) -> str:
        parts = [p for p in dict.fromkeys((self.name.strip(), self.place.strip())) if p and p.upper() != self.code]
        return " | ".join(parts)

    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def latest_frame(self):
        """Latest annotated frame as an image (evidence for alerts raised
        outside the pipeline, e.g. blind-spot tracking)."""
        jpeg = self.latest_jpeg()
        if jpeg is None:
            return None
        return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)

    def frame_size(self):
        with self._lock:
            h, w = self._frame_shape
        return w, h

    def to_dict(self) -> dict:
        w, h = self.frame_size()
        counts = dict(self.pipeline.last_object_counts) if self.status == STATUS_ONLINE else {}
        return {
            "id": self.id,
            "code": self.code,
            "name": self.name,
            "source_label": source_label(self.source),
            "status": self.status,
            "error": self.last_error if self.status != STATUS_ONLINE else "",
            "offline_since": self.offline_since,
            "fps": round(self._fps, 1) if self.status == STATUS_ONLINE else 0.0,
            "inference_ms": round(self._proc_ms, 1),
            "resolution": f"{w}x{h}" if w else None,
            "frame_width": w,
            "frame_height": h,
            "object_count": sum(counts.values()),
            "object_counts": counts,
            "active_alerts": self.pipeline.alert_manager.active_count(camera_id=self.id),
            "critical_alerts": self.pipeline.alert_manager.active_count(camera_id=self.id, band="CRITICAL"),
            "zone_counts": dict(self.pipeline.rule_engine.zone_counts) if self.status == STATUS_ONLINE else {},
            "location": self.location.to_dict(),
            "weapon_detector_enabled": self.shared.weapon.enabled,
            **self.pipeline.alert_manager.camera_summary(self.id),
        }

    def persist_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "source": self.source, "location": self.location.to_dict()}


def resize_to_width(frame, width: int):
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    return cv2.resize(frame, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)


def draw_camera_label(frame, text: str, code: str):
    """Camera name + location in the top-right corner of the stream."""
    label = f"{code}  {text}" if text else code
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (tw, th), base = cv2.getTextSize(label, font, scale, thick)
    x2 = frame.shape[1] - 6
    x1 = max(0, x2 - tw - 10)
    cv2.rectangle(frame, (x1, 6), (x2, 6 + th + base + 8), (17, 26, 25), -1)
    cv2.putText(frame, label, (x1 + 5, 6 + th + 4), font, scale, (148, 215, 248), thick, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------
class CameraManager:
    def __init__(self, shared: Optional[SharedModels] = None, alert_manager: Optional[AlertManager] = None,
                 store_path: Path = None, zones_dir: Path = None, primary_zone_file: Path = None):
        self.shared = shared or SharedModels()
        self.alert_manager = alert_manager or AlertManager()
        self.store_path = Path(store_path or config.CAMERAS_FILE)
        self.zones_dir = Path(zones_dir or config.DATA_DIR)
        # Camera 1 keeps the single-source dashboard's zones.json.
        self.primary_zone_file = Path(primary_zone_file or config.BASE_DIR / "zones.json")
        self._lock = threading.RLock()
        self.cameras: Dict[str, Camera] = {}
        self.links: List[dict] = []
        self.thresholds: Dict[str, float] = {}
        self.site: Optional[dict] = None   # {lat, lng, label, zoom}: where the map opens
        self.frame_hooks: List[Callable] = []
        self._restore()

    # -- persistence ------------------------------------------------------------
    def _restore(self):
        if not self.store_path.exists():
            return
        try:
            data = json.loads(self.store_path.read_text())
        except (OSError, ValueError):
            return
        for c in data.get("cameras", []):
            if not isinstance(c, dict) or "id" not in c:
                continue
            try:
                self._create(c["id"], c["name"], c["source"], Location.from_dict(c.get("location"), c["name"]))
            except Exception as exc:
                print(f"[cameras] could not restore {c}: {exc}")
        site = data.get("site")
        if isinstance(site, dict) and site.get("lat") is not None:
            try:
                self.site = self._clean_site(site)
            except (TypeError, ValueError):
                self.site = None
        self.links = [
            {**l, "id": self.link_id(l["a"], l["b"])} for l in data.get("links", [])
            if l.get("a") in self.cameras and l.get("b") in self.cameras
        ]

    def save(self):
        with self._lock:
            data = {"cameras": [c.persist_dict() for c in self.cameras.values()], "links": self.links,
                    "site": self.site}
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.store_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.store_path)

    # -- CRUD ---------------------------------------------------------------------
    def _zone_path(self, cam_id: str) -> Path:
        if cam_id == "cam-1":
            return self.primary_zone_file
        return self.zones_dir / f"zones_{cam_id}.json"

    def _next_id(self) -> str:
        n = 1
        while f"cam-{n}" in self.cameras:
            n += 1
        return f"cam-{n}"

    def _create(self, cam_id: str, name: str, source: Source, location: Location) -> Camera:
        camera = Camera(cam_id, name, source, location, self.shared, self.alert_manager,
                        self._zone_path(cam_id), on_frame=self._on_frame)
        if self.thresholds:
            camera.pipeline.rule_engine.update_thresholds(**self.thresholds)
        self.cameras[cam_id] = camera
        return camera

    def add(self, name: str, source: Source, location: Optional[dict] = None,
            start: bool = True, cam_id: Optional[str] = None) -> Camera:
        resolved = resolve_source(source)
        with self._lock:
            if cam_id in self.cameras:
                raise ValueError(f"Camera {cam_id} already exists")
            if len(self.cameras) >= config.MAX_CAMERAS:
                raise CameraLimitError(
                    f"Camera limit reached ({config.MAX_CAMERAS}). Remove a camera first "
                    f"(MAX_CAMERAS in backend/config.py).")
            cam_id = cam_id or self._next_id()
            name = (name or "").strip() or f"CAM-{int(cam_id.split('-')[1]):02d}"
            camera = self._create(cam_id, name, resolved, Location.from_dict(location, name))
        self.save()
        audit("CAMERA_ADDED", f"{camera.title} ({source_label(resolved)}) at {camera.place}",
              actor="operator", camera_id=cam_id)
        if start:
            camera.start()
        return camera

    @staticmethod
    def _clean_site(site: dict) -> dict:
        lat, lng = normalize_latlng(site.get("lat"), site.get("lng"))
        if lat is None:
            raise ValueError("Site needs lat and lng")
        zoom = site.get("zoom")
        return {"lat": lat, "lng": lng, "label": str(site.get("label") or "")[:200],
                "zoom": int(zoom) if zoom is not None else 17,
                "accuracy_m": site.get("accuracy_m"), "source": site.get("source") or ""}

    def set_site(self, site: Optional[dict]) -> Optional[dict]:
        self.site = self._clean_site(site) if site else None
        self.save()
        audit("SITE_SET", f"{self.site['label'] or 'site'} at {self.site['lat']:.5f}, {self.site['lng']:.5f}"
              if self.site else "site location cleared", actor="operator")
        return self.site

    def get(self, cam_id: str) -> Optional[Camera]:
        return self.cameras.get(cam_id)

    def update(self, cam_id: str, name: Optional[str] = None, location: Optional[dict] = None) -> Camera:
        camera = self.cameras[cam_id]
        if name:
            camera.name = name.strip()
        if location is not None:
            merged = {**camera.location.to_dict(), **location}
            camera.location = Location.from_dict(merged, camera.name)
        self.save()
        audit("CAMERA_UPDATED", f"{camera.title} at {camera.place}", actor="operator", camera_id=cam_id)
        return camera

    def set_source(self, cam_id: str, source: Source, start: bool = True) -> Camera:
        camera = self.cameras[cam_id]
        resolved = resolve_source(source)
        camera.stop()
        camera.source = resolved
        self.save()
        audit("SOURCE_CHANGED", f"{camera.code} -> {source_label(resolved)}", actor="operator", camera_id=cam_id)
        if start:
            camera.start()
        return camera

    def remove(self, cam_id: str):
        with self._lock:
            camera = self.cameras.pop(cam_id, None)
            if camera is None:
                raise KeyError(cam_id)
            self.links = [l for l in self.links if cam_id not in (l.get("a"), l.get("b"))]
        camera.stop()
        # A camera's zones belong to it: a new camera reusing this id starts clean.
        camera.pipeline.zone_store.replace_all([])
        self.save()
        audit("CAMERA_REMOVED", f"{camera.title}", actor="operator", camera_id=cam_id)

    def stop_all(self):
        for camera in list(self.cameras.values()):
            camera.stop()

    def list(self) -> List[Camera]:
        return sorted(self.cameras.values(), key=lambda c: c.number)

    # -- links (expected walking time between cameras; Phase 7 uses them) ---------
    @staticmethod
    def link_id(a: str, b: str) -> str:
        return "~".join(sorted((a, b)))

    def set_link(self, a: str, b: str, seconds: float) -> dict:
        if a == b:
            raise ValueError("A link needs two different cameras")
        for cam_id in (a, b):
            if cam_id not in self.cameras:
                raise KeyError(cam_id)
        if seconds <= 0:
            raise ValueError("Walking time must be positive")
        link = {"id": self.link_id(a, b), "a": a, "b": b, "seconds": float(seconds)}
        with self._lock:
            self.links = [l for l in self.links if self.link_id(l["a"], l["b"]) != link["id"]] + [link]
        self.save()
        audit("LINK_SAVED", f"{self.cameras[a].code} <-> {self.cameras[b].code}: {seconds:g}s walk", actor="operator")
        return link

    def remove_link(self, link_id: str):
        with self._lock:
            before = len(self.links)
            self.links = [l for l in self.links if self.link_id(l["a"], l["b"]) != link_id]
            removed = before != len(self.links)
        if not removed:
            raise KeyError(link_id)
        self.save()
        audit("LINK_REMOVED", link_id, actor="operator")

    def expected_seconds(self, a: str, b: str) -> Optional[float]:
        lid = self.link_id(a, b)
        for l in self.links:
            if self.link_id(l["a"], l["b"]) == lid:
                return float(l["seconds"])
        return None

    def update_thresholds(self, **values):
        self.thresholds.update(values)
        for camera in self.cameras.values():
            camera.pipeline.rule_engine.update_thresholds(**values)

    # -- analytics hooks -----------------------------------------------------------
    def _on_frame(self, camera: Camera, frame, annotated):
        for hook in self.frame_hooks:
            hook(camera, frame, annotated)
