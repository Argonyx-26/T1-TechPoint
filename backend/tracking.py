"""Per-track motion history, sitting between raw per-frame detections and the
behavior rules engine (which needs to reason about a track over time: how
long it has dwelled somewhere, which way it is moving, whether it has gone
stationary).
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Tuple

from backend.detection.object_detector import Detection
from backend.geometry import bbox_centroid, euclidean

STALE_SECONDS = 3.0        # drop a track if unseen for this long
HISTORY_MAXLEN = 600       # ~20s at 30fps of (timestamp, centroid) samples


@dataclass
class TrackState:
    track_id: int
    cls_name: str
    bbox: Tuple[float, float, float, float]
    conf: float
    first_seen: float
    last_seen: float
    history: Deque[Tuple[float, Tuple[float, float]]] = field(
        default_factory=lambda: deque(maxlen=HISTORY_MAXLEN)
    )

    @property
    def centroid(self) -> Tuple[float, float]:
        return bbox_centroid(self.bbox)

    def speed_px_s(self) -> float:
        """Speed estimated from roughly the last second of history."""
        if len(self.history) < 2:
            return 0.0
        t_end, p_end = self.history[-1]
        for t, p in reversed(self.history):
            dt = t_end - t
            if dt >= 1.0:
                return euclidean(p, p_end) / dt
        t0, p0 = self.history[0]
        dt = t_end - t0
        return euclidean(p0, p_end) / dt if dt > 0 else 0.0

    def movement_vector(self, window_seconds: float = 1.5) -> Tuple[float, float]:
        """Net displacement vector over the trailing window (direction + magnitude)."""
        if len(self.history) < 2:
            return (0.0, 0.0)
        t_end, p_end = self.history[-1]
        for t, p in reversed(self.history):
            if t_end - t >= window_seconds:
                return (p_end[0] - p[0], p_end[1] - p[1])
        t0, p0 = self.history[0]
        return (p_end[0] - p0[0], p_end[1] - p0[1])


class TrackManager:
    """Ingests one frame of detections at a time, keeps rolling per-id history."""

    def __init__(self):
        self.tracks: Dict[int, TrackState] = {}
        self._next_synthetic_id = -1

    def update(self, detections: List[Detection], timestamp: float) -> Dict[int, TrackState]:
        seen_ids = set()
        for det in detections:
            track_id = det.track_id
            if track_id < 0:
                # No stable id from the tracker (e.g. still warming up): give it
                # an ephemeral id so it can still be drawn/scored this frame,
                # just without history-dependent rules (loitering, direction).
                track_id = self._next_synthetic_id
                self._next_synthetic_id -= 1

            state = self.tracks.get(track_id)
            if state is None:
                state = TrackState(
                    track_id=track_id,
                    cls_name=det.cls_name,
                    bbox=det.bbox,
                    conf=det.conf,
                    first_seen=timestamp,
                    last_seen=timestamp,
                )
                self.tracks[track_id] = state
            else:
                state.cls_name = det.cls_name
                state.bbox = det.bbox
                state.conf = det.conf
                state.last_seen = timestamp

            state.history.append((timestamp, bbox_centroid(det.bbox)))
            seen_ids.add(track_id)

        stale_ids = [
            tid for tid, st in self.tracks.items() if timestamp - st.last_seen > STALE_SECONDS
        ]
        for tid in stale_ids:
            del self.tracks[tid]

        return {tid: st for tid, st in self.tracks.items() if tid in seen_ids}
