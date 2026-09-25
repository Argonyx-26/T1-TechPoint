"""Runtime on/off switches for every analytics module, so any of them can be
turned off if FPS drops. Defaults come from config; /api/features flips them
while the server runs (and logs each change to the audit log)."""
import threading

from backend import config

_lock = threading.Lock()
_features = dict(config.FEATURE_DEFAULTS)


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
        return dict(_features)
