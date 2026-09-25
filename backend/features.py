"""Runtime on/off switches for every analytics module, so any of them can be
turned off if FPS drops. Defaults come from config; /api/features flips them
while the server runs (and logs each change to the audit log). Changes are
kept in data/features.json, so a module switched off for FPS stays off after
a restart."""
import json
import threading

from backend import config

_lock = threading.Lock()
_features = dict(config.FEATURE_DEFAULTS)


def _path():
    return config.DATA_DIR / "features.json"


def load():
    """Saved switches on top of the defaults; unknown names are ignored."""
    try:
        saved = json.loads(_path().read_text())
    except (OSError, ValueError):
        return
    with _lock:
        for name, value in saved.items():
            if name in _features:
                _features[name] = bool(value)


load()


def enabled(name: str) -> bool:
    with _lock:
        return bool(_features.get(name, False))


def all_features() -> dict:
    with _lock:
        return dict(_features)


def set_features(**values) -> dict:
    """Only known module names are accepted."""
    with _lock:
        for name, value in values.items():
            if name in _features and value is not None:
                _features[name] = bool(value)
        snapshot = dict(_features)
    try:
        _path().parent.mkdir(parents=True, exist_ok=True)
        _path().write_text(json.dumps(snapshot, indent=1))
    except OSError:
        pass
    return snapshot
