"""Central settings: paths, model choices, thresholds."""
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent

# Models
GENERAL_MODEL_PATH = ROOT / "pretrained" / "yolov8n.pt"  # COCO, falls back to auto-download
WEAPON_MODEL_PATH = ROOT / "pretrained" / "weapon_model" / "best.pt"  # pretrained pre-event, disclosed
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
INFER_IMGSZ = 640
TRACKER_CFG = "bytetrack.yaml"

# Two-tier confidence: low floor keeps boxes from flickering, higher bar counts toward alerts
DISPLAY_CONF = 0.25
ALERT_CONF = 0.5

# COCO classes the general detector tracks (0 = person). None tracks everything.
TRACKED_CLASSES = None

# Tracks not seen for this long are dropped from the track manager
TRACK_STALE_SECONDS = 2.0
TRACK_HISTORY_LEN = 60

# Streaming
JPEG_QUALITY = 80
STREAM_MAX_FPS = 30
IDLE_FRAME_SIZE = (1280, 720)

# Paths the frontend is served from (read-only, owned by the frontend teammate)
FRONTEND_DIR = ROOT / "frontend"
