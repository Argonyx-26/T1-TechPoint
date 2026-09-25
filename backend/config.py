"""Central configuration for models, thresholds, and storage paths."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Models -----------------------------------------------------------------
# General detector: pretrained YOLOv8, COCO classes, auto-downloaded by
# ultralytics on first use. No custom training required.
GENERAL_MODEL_PATH = os.environ.get("ARGONYX_GENERAL_MODEL", "yolov8n.pt")

# Weapon detector: a separate pretrained/fine-tuned YOLO model (gun/knife
# classes). Drop a compatible .pt file at this path to enable it; the system
# runs fully without it (weapon alerts are simply never raised).
# Active model: run15 (run10 fine-tuned on horizontal-phone frames; fixes the
# phone -> pistol/knife false alarm). models/weapon.pt is still run10 -- point
# this back at it (or set ARGONYX_WEAPON_MODEL) to revert.
WEAPON_MODEL_PATH = Path(
    os.environ.get("ARGONYX_WEAPON_MODEL", BASE_DIR / "models" / "weapon_run15.pt")
)
WEAPON_IMGSZ = 960  # higher than the 640 default -- gives small/distant objects more pixels to be detected from
# Only these weapon-model classes count as an actual threat. The other
# classes in the Sohas dataset (smartphone, monedero, billete, tarjeta) are
# deliberately-included "this is NOT a weapon" negatives, not threats.
WEAPON_CONF_SMOOTH_WINDOW = 5   # average the on-screen confidence over this many recent frames
WEAPON_BOX_PERSIST_SECONDS = 4.0   # keep the box drawn this long after the last real detection, so it doesn't flicker out for one dropped frame
WEAPON_MIN_CONF = 0.25       # floor for what the model reports at all (feeds display smoothing only)
WEAPON_ALERT_MIN_CONF = 0.45   # stricter bar that actually raises a CRITICAL alert
WEAPON_ALERT_MIN_CONF = 0.55   # raised from 0.45 -- fewer false CRITICALs, requires a clearer hold
WEAPON_THREAT_CLASSES = {"pistol", "knife"}
# A real weapon never fills most of a CCTV frame; boxes larger than this
# fraction of the frame are the model latching onto the whole scene (e.g. a
# phone-filmed monitor) and are dropped before display/alerting.
WEAPON_MAX_BOX_FRACTION = 0.30

def _resolve_device() -> str:
    """`ARGONYX_DEVICE` always wins; otherwise auto-pick a CUDA GPU if torch
    can see one, falling back to CPU. Checked at import time once."""
    override = os.environ.get("ARGONYX_DEVICE")
    if override:
        return override
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


DEVICE = _resolve_device()
# Half-precision (fp16) inference: real speedup on a CUDA GPU, unsupported on CPU.
USE_HALF_PRECISION = DEVICE.startswith("cuda")
DETECTOR_CONF_THRESHOLD = 0.35
TRACKER_CONFIG = "bytetrack.yaml"

# COCO class names this system reasons about explicitly.
PERSON_CLASS = "person"
ITEM_CLASSES = {"backpack", "handbag", "suitcase"}

# --- Rule thresholds (all runtime-adjustable via /api/thresholds) -----------
LOITER_SECONDS = 8.0                 # time stationary-in-zone before flagging
CROWD_COUNT_THRESHOLD = 5            # person count in a zone before flagging
SURGE_MIN_INCREASE = 4               # zone count rising by this much...
SURGE_WINDOW_S = 10.0                # ...within this many seconds = surge
SURGE_AVG_WINDOW_S = 60.0            # rolling-average window for the relative surge check
SURGE_AVG_MULTIPLIER = 1.5           # count above this x its rolling average = surge...
SURGE_MIN_PEOPLE = 3                 # ...but only with at least this many people present
SURGE_SMOOTH_S = 1.0                 # median-smooth zone counts over this window (kills 1-frame detector dropouts)
CROWD_WARN_FRACTION = 0.7            # overlay turns amber above this fraction of a zone's crowd threshold
UNATTENDED_SECONDS = 10.0            # time an item sits alone before flagging
UNATTENDED_RADIUS_PX = 120           # "near" radius for item<->person association
STATIONARY_SPEED_PX_S = 40.0         # centroid speed below this = "stationary"
WRONG_DIRECTION_ANGLE_DEG = 120.0    # angle vs. allowed vector to call it wrong-way
MIN_TRACK_HISTORY = 5                # frames of history before rules trust a track

ALERT_COOLDOWN_SECONDS = 12.0        # suppress duplicate alerts within this window

# --- Severity -----------------------------------------------------------
SEVERITY_WEIGHTS = {
    "WEAPON": 100,
    "UNATTENDED_OBJECT": 80,
    "RESTRICTED_ZONE_INTRUSION": 75,
    "CROWD_SURGE": 70,       # sudden rise -> HIGH
    "CROWD_THRESHOLD": 55,   # absolute count over the zone limit -> MEDIUM
    "WRONG_DIRECTION": 45,
    "LOITERING": 30,
}
CO_OCCURRENCE_BONUS = 10   # added per extra distinct alert type in same zone/window
CO_OCCURRENCE_CAP = 30     # max total bonus from co-occurrence
CO_OCCURRENCE_WINDOW_SECONDS = 5.0

SEVERITY_BANDS = (
    (85, "CRITICAL"),
    (65, "HIGH"),
    (40, "MEDIUM"),
    (0, "LOW"),
)

# --- Storage ------------------------------------------------------------
EVIDENCE_DIR = BASE_DIR / "evidence"
UPLOAD_DIR = BASE_DIR / "uploads"
SAMPLE_DATA_DIR = BASE_DIR / "sample_data"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

STREAM_JPEG_QUALITY = 80
STREAM_MAX_FPS = int(os.environ.get("ARGONYX_STREAM_MAX_FPS", "60"))
