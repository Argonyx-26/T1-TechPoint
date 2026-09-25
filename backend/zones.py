"""Zone definitions and a small JSON-backed store so zone config drawn in the
dashboard survives a server restart.

A single polygon can serve several rule roles at once (e.g. a doorway zone
can be both `restricted` and carry an `allowed_direction` for wrong-way
detection) -- this mirrors how an analyst would actually set one up: draw a
shape, then flip on whichever checks apply to it.
"""
import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from backend import config

ZONES_FILE = config.BASE_DIR / "zones.json"


@dataclass
class Zone:
    id: str
    name: str
    polygon: List[Tuple[float, float]]
    restricted: bool = False
    crowd_threshold: Optional[int] = None
    loiter_seconds: Optional[float] = None
    allowed_direction: Optional[Tuple[float, float]] = None


class ZoneStore:
    def __init__(self, path: Path = ZONES_FILE):
        self._path = path
        self._lock = threading.Lock()
        self._zones: List[Zone] = []
        self._load()

    def _load(self):
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text())
            self._zones = [Zone(**z) for z in raw]
        except (json.JSONDecodeError, TypeError):
            self._zones = []

    def _save(self):
        self._path.write_text(json.dumps([asdict(z) for z in self._zones], indent=2))

    def list(self) -> List[Zone]:
        with self._lock:
            return list(self._zones)

    def replace_all(self, zones: List[Zone]):
        with self._lock:
            self._zones = zones
            self._save()
