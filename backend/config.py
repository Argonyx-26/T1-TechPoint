"""Central settings: paths, model choices, thresholds."""
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent

# Models
GENERAL_MODEL_PATH = ROOT / "pretrained" / "yolov8n.pt"  # COCO, falls back to auto-download
WEAPON_MODEL_PATH = ROOT / "pretrained" / "weapon_model" / "best.pt"  # pretrained pre-event, disclosed
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
HALF = DEVICE.startswith("cuda")  # fp16 only helps (and only works) on GPU
PRECISION = 16 if HALF else 32  # ultralytics 8.4 "quantize" arg (replaces deprecated half=)
INFER_IMGSZ = 640
TRACKER_CFG = "bytetrack.yaml"

# Weapon detector (run10: pistol, smartphone, knife, monedero, billete, tarjeta).
# The other four classes are confusors that soak up phones/wallets; only these alert.
WEAPON_THREAT_CLASSES = {"pistol", "knife"}
WEAPON_DISPLAY_CONF = 0.25  # draw a box only
WEAPON_ALERT_CONF = 0.5     # counts toward confirmation
WEAPON_WINDOW = 8           # frames of weapon inference kept
WEAPON_MIN_HITS = 5         # hits in the window needed to confirm
WEAPON_EVERY_N_FRAMES = 1   # raise on slow CPU machines; the window then spans N x 8 frames

# Two-tier confidence: low floor keeps boxes from flickering, higher bar counts toward alerts
DISPLAY_CONF = 0.25
ALERT_CONF = 0.5

# COCO classes the general detector tracks and draws, by name, with a fixed BGR color each.
# Anything else COCO knows (chairs, TVs, ...) is ignored. Order = order in the counts panel.
SECURITY_CLASSES = {
    "person": (0, 200, 0),
    "backpack": (255, 140, 0),
    "handbag": (200, 0, 200),
    "suitcase": (140, 70, 20),
    "bottle": (255, 255, 0),
    "cell phone": (0, 255, 255),
    "laptop": (255, 0, 150),
    "umbrella": (120, 200, 255),
    "knife": (0, 100, 255),
    "scissors": (80, 160, 255),
    "bicycle": (180, 255, 120),
    "car": (255, 200, 120),
    "motorcycle": (160, 120, 255),
}
IN_RESTRICTED_COLOR = (0, 0, 255)  # persons inside a restricted zone

# Tracks not seen for this long are dropped from the track manager
TRACK_STALE_SECONDS = 2.0
TRACK_HISTORY_LEN = 60

# Streaming
JPEG_QUALITY = 80
STREAM_MAX_FPS = 30
IDLE_FRAME_SIZE = (1280, 720)

# Paths the frontend is served from (read-only, owned by the frontend teammate)
FRONTEND_DIR = ROOT / "frontend"

# Video sources
SAMPLES_DIR = ROOT / "test_videos"
UPLOADS_DIR = ROOT / "uploads"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
MAX_UPLOAD_BYTES = 1024 * 1024 * 1024  # 1 GB

# Zones persist across restarts
DATA_DIR = ROOT / "data"
ZONES_PATH = DATA_DIR / "zones.json"

# Rule thresholds: defaults, editable live via /api/thresholds
DEFAULT_THRESHOLDS = {
    "loiter_seconds": 30.0,
    "crowd_threshold": 10,
    "unattended_seconds": 20.0,
    "wrong_direction_angle_deg": 45.0,
}

# Alerts
ALERT_MAX = 200
ALERT_COOLDOWN_SECONDS = 12.0
# rule -> (human-readable type, severity). Weapon alerts far outrank behavioural ones.
ALERT_RULES = {
    "weapon": ("Weapon Detected", "critical"),
    "restricted_intrusion": ("Restricted Zone Intrusion", "high"),
    "crowd_surge": ("Crowd Surge", "medium"),
    "wrong_direction": ("Wrong Direction", "medium"),
    "loitering": ("Loitering", "low"),
}
SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}
