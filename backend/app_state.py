"""Shared state the FastAPI endpoints read from.

The system is multi-camera (backend/cameras.py). The original single-source
API keeps working by mapping onto camera 1 ("cam-1"): /video_feed,
/api/source/*, /api/zones and /api/status all act on it.
"""
import time
from typing import Optional, Union

from backend import config
from backend.analytics.rules import RuleEngine
from backend.cameras import STATUS_ONLINE, Camera, CameraManager
from backend.behavior import BehaviourEngine
from backend.movements import MovementTracker
from backend.search import SearchIndex

PRIMARY_ID = "cam-1"


class AppState:
    def __init__(self):
        # Restores saved cameras in STOPPED state: the server always starts idle.
        self.manager = CameraManager()
        # Ask Vigil: indexes every camera through a per-frame hook.
        self.search = SearchIndex(self.manager)
        self.manager.frame_hooks.append(self.search.on_frame)
        # Blind-spot tracking over the search index's sightings.
        self.movements = MovementTracker(self.search, self.manager, self.manager.alert_manager)
        # Pose behaviours (fighting, throwing, distress) on the same GPU lock as detection.
        self.behaviour = BehaviourEngine(self.manager.shared._lock, self.manager.alert_manager,
                                         embedder=self.search.embedder)
        self.manager.frame_hooks.append(self.behaviour.on_frame)
        self._start_time = time.time()

    @property
    def alert_manager(self):
        return self.manager.alert_manager

    def primary(self) -> Optional[Camera]:
        return self.manager.get(PRIMARY_ID)

    def switch_source(self, source: Union[int, str], label: str = "") -> Camera:
        """Single-source flow: (re)point camera 1 at a new source and start it."""
        if self.primary() is None:
            return self.manager.add("CAM-01", source, cam_id=PRIMARY_ID)
        return self.manager.set_source(PRIMARY_ID, source)

    def stop(self):
        self.manager.stop_all()

    def frame_size(self, cam_id: str = PRIMARY_ID):
        camera = self.manager.get(cam_id)
        return camera.frame_size() if camera else (0, 0)

    def latest_jpeg(self, cam_id: str = PRIMARY_ID) -> Optional[bytes]:
        camera = self.manager.get(cam_id)
        return camera.latest_jpeg() if camera else None

    def thresholds(self) -> dict:
        camera = self.primary() or next(iter(self.manager.cameras.values()), None)
        engine = camera.pipeline.rule_engine if camera else RuleEngine()
        if camera is None and self.manager.thresholds:
            engine.update_thresholds(**self.manager.thresholds)
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

    def status(self) -> dict:
        camera = self.primary()
        cam = camera.to_dict() if camera else {}
        cameras = self.manager.list()
        online = [c for c in cameras if c.status == STATUS_ONLINE]
        counts = cam.get("object_counts", {})
        return {
            "running": bool(camera and camera.status == STATUS_ONLINE),
            "source": cam.get("source_label", "none") if camera else "none",
            "camera_status": cam.get("status", "none"),
            "fps": cam.get("fps", 0.0),
            "frame_width": cam.get("frame_width", 0),
            "frame_height": cam.get("frame_height", 0),
            "weapon_detector_enabled": self.manager.shared.weapon.enabled,
            "device": config.DEVICE,
            "objects": counts,
            "object_counts": counts,
            "object_count": sum(counts.values()),
            "alert_count": self.alert_manager.active_count(),
            "zone_counts": dict(camera.pipeline.rule_engine.zone_counts) if camera else {},
            "inference_ms": cam.get("inference_ms", 0.0),
            "capture_ms": 0.0,
            "uptime_seconds": round(time.time() - self._start_time),
            "total_alerts": self.alert_manager.total_count(),
            "cameras_online": len(online),
            "cameras_total": len(cameras),
            "total_objects": sum(c.to_dict()["object_count"] for c in online),
        }


app_state = AppState()
