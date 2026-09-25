"""Detection zones: normalized (0-1) polygons drawn in the dashboard, persisted to JSON."""
import json
import logging
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from backend import config

log = logging.getLogger(__name__)


@dataclass
class Zone:
    id: str
    name: str
    polygon: list  # [[x, y], ...] normalized 0-1
    restricted: bool = False

    def pixel_polygon(self, width: int, height: int) -> list[tuple[float, float]]:
        return [(x * width, y * height) for x, y in self.polygon]


def validate_zones(raw) -> list[Zone]:
    """Parse the full zone list sent by the dashboard. Raises ValueError with a readable message."""
    if not isinstance(raw, list):
        raise ValueError("body must be a JSON list of zones")
    zones, seen = [], set()
    for i, z in enumerate(raw):
        if not isinstance(z, dict):
            raise ValueError(f"zone {i} must be an object")
        zid = str(z.get("id") or f"zone-{i + 1}")
        if zid in seen:
            raise ValueError(f"duplicate zone id {zid}")
        seen.add(zid)
        poly = z.get("polygon")
        if not isinstance(poly, list) or len(poly) < 3:
            raise ValueError(f"zone {zid}: polygon needs at least 3 points")
        points = []
        for p in poly:
            if (not isinstance(p, (list, tuple)) or len(p) != 2
                    or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in p)):
                raise ValueError(f"zone {zid}: each point must be [x, y]")
            x, y = float(p[0]), float(p[1])
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(f"zone {zid}: points must be normalized to 0-1")
            points.append([x, y])
        name = str(z.get("name") or f"Zone {i + 1}")
        zones.append(Zone(zid, name, points, bool(z.get("restricted", False))))
    return zones


def point_in_polygon(pt: tuple[float, float], poly: list[tuple[float, float]]) -> bool:
    """Ray casting; poly in the same coordinate space as pt."""
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside
    return inside


class ZoneStore:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or config.ZONES_PATH)
        self._lock = threading.Lock()
        self._zones: list[Zone] = []
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            self._zones = validate_zones(json.loads(self.path.read_text(encoding="utf-8")))
        except (ValueError, json.JSONDecodeError) as exc:
            log.warning("Ignoring unreadable zones file %s: %s", self.path, exc)
            self._zones = []

    def all(self) -> list[Zone]:
        with self._lock:
            return list(self._zones)

    def as_dicts(self) -> list[dict]:
        return [asdict(z) for z in self.all()]

    def replace(self, raw) -> list[dict]:
        zones = validate_zones(raw)
        with self._lock:
            self._zones = zones
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps([asdict(z) for z in zones], indent=2), encoding="utf-8")
            tmp.replace(self.path)
        return self.as_dicts()
