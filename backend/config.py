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
# Pose check on weapon candidates (analytics/weapon_verify.py): drop a box
# that mostly covers a face, or sits on a torso with no wrist near it.
WEAPON_POSE_VERIFY = True
WEAPON_FACE_OVERLAP = 0.5     # share of the weapon box inside the head region
WEAPON_FACE_PAD = 0.35        # head region = face keypoints padded by this x face size
WEAPON_WRIST_REACH = 0.5      # "near a hand" = within this x shoulder width of the box
# The 5-of-8 filter only counts hits near the current box: a knife lowered out
# of view plus a flicker on a face elsewhere is not one sustained weapon.
WEAPON_TRACK_DIST = 1.0       # centre distance, as a multiple of the larger box side...
WEAPON_STRONG_CONF = 0.70     # fast path: a weapon this confident...
WEAPON_STRONG_HITS = 2         # ...seen this many times in the window, same spot, confirms at once
                               # (a gun shown briefly: m2-res_480p has 0.74 + 0.76 at the end of its
                               # first 4.6 s play, but never 5 of 8 frames, so it took minutes of loops)
WEAPON_TRACK_MIN_FRAC = 0.2   # ...but never less than this x the frame diagonal: a small gun being swung moves more than its own size between frames
# Raw frame + raw detections + pose for every fired weapon alert.
WEAPON_DEBUG_DIR = BASE_DIR / "debug" / "confirmed_alerts"

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
    "BLIND_SPOT_GAP": 50,     # MEDIUM (HIGH when the subject was in a serious alert)
    "FIGHT": 70,              # HIGH
    "HANDS_RAISED": 70,       # HIGH, CRITICAL with a confirmed weapon on the same camera
    "PERSON_DOWN": 70,        # HIGH
    "CROWD_PANIC": 70,        # HIGH
    "OBJECT_THROWN": 50,      # MEDIUM, HIGH toward a person or into a restricted zone
    "SUBJECT_MISSING": 50,
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

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Audit log (append-only, hash-chained; see backend/audit.py) -------------
AUDIT_ENABLED = True
AUDIT_DB = Path(os.environ.get("ARGONYX_AUDIT_DB", DATA_DIR / "audit.db"))

# --- Multi-camera ---------------------------------------------------------
MAX_CAMERAS = 4
CAMERA_MAX_FPS = 12            # processing rate per camera (capture keeps only the latest frame)
PROCESS_WIDTH = 640            # frames are downscaled to this width before analysis
WEAPON_EVERY_N_FRAMES = 1      # run the weapon model on every Nth processed frame per camera
RECONNECT_INITIAL_S = 3.0      # first retry after a stream drops...
RECONNECT_MAX_S = 30.0         # ...doubling up to this
OFFLINE_AFTER_S = 30.0         # "reconnecting" becomes "offline since HH:MM" after this long
CAMERAS_FILE = DATA_DIR / "cameras.json"

# --- Location (backend/geo.py) ----------------------------------------------
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
# Nominatim's usage policy: identify the application; max 1 request/s (enforced in geo.py).
NOMINATIM_USER_AGENT = "VIGIL-control-room/1.0 (argonyx-threat-detection; " +     os.environ.get("ARGONYX_NOMINATIM_CONTACT", "local demo") + ")"
LOCATE_HTTPS_PORT = int(os.environ.get("ARGONYX_LOCATE_PORT", "8443"))   # phone GPS needs HTTPS; 0 disables
APPROX_LOCATION_M = 200.0     # a browser fix worse than this is IP / Wi-Fi guesswork: never trusted silently
SITEPLAN_BASENAME = DATA_DIR / "siteplan"   # uploaded floor plan: data/siteplan.<ext>
STREAM_TIMEOUT_S = 5.0         # network streams: open/read timeout before a reconnect
CAMERA_TEST_TIMEOUT_S = 3.0    # "TEST" in the add-camera dialog must grab a frame within this
DISCOVER_TIMEOUT_S = 1.0       # per-host connect/read timeout when scanning the /24 for phones
SOURCE_CONNECT_TIMEOUT_S = 5.0
# "START 4-CAMERA DEMO": sample clips (webcam 0 replaces the first when present).
DEMO_CAMERAS = [
    {"name": "Main Gate", "source": "vtest.avi", "location": {"label": "Main Gate"}, "webcam_first": True},
    {"name": "Lobby", "source": "confusers.mp4", "location": {"label": "Lobby"}},
    {"name": "Parking", "source": "vtest.avi", "location": {"label": "Parking"}},
    {"name": "Corridor B", "source": "confusers.mp4", "location": {"label": "Corridor B"}},
]  # single-source flow: wait this long for a first frame before reporting failure

# --- Analytics modules: each can be switched off (/api/features) if FPS drops --
FEATURE_DEFAULTS = {
    "pose": True,          # yolov8n-pose, needed by fighting/throwing/distress
    "fighting": True,
    "throwing": False,     # OFF by default: on clean UMN footage (overlays removed) it caught 0/17 real throws
                           # and only raised stray alerts; the detector cannot see small objects in flight
    "distress": True,      # hands raised, person down, crowd panic
    "search": True,        # Ask Vigil indexing + natural-language search
    "reid": True,          # cross-camera re-identification (global subject ids)
    "gap_tracking": True,  # blind-spot / missing-subject alerts between linked cameras
}

# --- Ask Vigil (OpenCLIP search + re-ID) ---------------------------------------
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
CLIP_CACHE_DIR = BASE_DIR / "models" / "clip"   # weights cached here: works offline
SEARCH_CLASSES = {"person", "backpack", "handbag", "suitcase"}
SEARCH_SAMPLE_S = 1.0          # at most one crop per track per second...
SEARCH_KEEP_BEST_S = 5.0       # ...keeping the sharpest one per track per 5s
SEARCH_MAX_ENTRIES = 8000      # ring buffer size
SEARCH_RETENTION_MIN = 30      # entries and sightings expire after this
SEARCH_QUEUE_MAX = 256         # pending crops; new ones are dropped when full (never block a camera)
SEARCH_BATCH = 16
SEARCH_CROP_PAD = 0.03         # tight crops: less background, better re-ID (0.886 -> 0.905 measured)
SIGHTING_CLOSE_S = 3.0         # a sighting closes when its track has been lost this long
# Cosine similarity of MEAN embeddings for "same person". Spec default was
# 0.82; measured on our clips (2026-09-25): same person across two phone
# sessions 0.886-0.905 (single crops only ~0.81), different people across
# clips p95 0.71 / p99 0.868; 0.82 merged 2.5% of stranger pairs.
REID_THRESHOLD = 0.86
REID_WARMUP_S = 5.0            # a new sighting is sampled every second for this long...
REID_REVISIT_EMBS = 4          # ...and re-ID is re-run on its mean for its first N crops
REID_MARGIN = 0.03             # the best candidate must beat the runner-up (another subject) by this
REID_WINDOW_S = 300.0          # only match people who left within the last 5 minutes
REID_ACTIVE_S = 1.0            # a subject visible elsewhere within this is not a candidate
REID_MIN_GAP_S = 0.5

# --- Behaviour (pose): fighting, throwing, distress --------------------------------
POSE_MODEL = "yolov8n-pose.pt"   # pretrained, auto-downloaded by ultralytics
POSE_FPS = 6                     # per camera, only while people are in view
POSE_CONF = 0.35
POSE_KP_CONF = 0.3               # keypoints below this confidence are ignored
POSE_MATCH_IOU = 0.3             # pose person <-> existing track
BEHAVIOUR_REARM_S = 10.0         # a behaviour re-alerts only after it was absent this long
HANDS_RAISED_S = 2.0             # both wrists above the nose this long
HANDS_RAISED_WEAPON_S = 30.0     # ...CRITICAL if a weapon was confirmed on the camera this recently
UPRIGHT_ASPECT = 1.4             # box h/w for "upright"
DOWN_ASPECT = 1.2                # box w/h for "horizontal"
DOWN_LOW_CONF = 0.2              # people lying on the floor score low as 'person': the person-down rule (only) sees them from here
DOWN_STRONG_ASPECT = 1.5         # this wide is lying whatever the skeleton says
DOWN_TORSO_DEG = 55.0            # shoulders->hips this far from vertical = lying
DOWN_HIP_ANKLE_FRAC = 0.25       # hips within this fraction of the box width of ankle height
UPRIGHT_LOOKBACK_S = 10.0        # must have been upright this recently (falls, not sleepers)
DOWN_BOX_MIN_WIDTH = 0.10        # box-only (no skeleton) candidates narrower than this x frame width are fragments, not bodies
DOWN_MIN_LENGTH = 0.6            # a lying body is about as long as it stood tall: box width >= this x that height
DOWN_NEAR_UPRIGHT = 1.0          # ...on this track, or anyone upright within this x their height of the spot
PERSON_DOWN_S = 3.0
FIGHT_WINDOW_S = 2.0
# Learned fight classifier (backend/fight.py, tools/train_fight_classifier.py)
FIGHT_CLF_MODEL = BASE_DIR / "models" / "fight_clf.npz"
FIGHT_CLF_FPS = 4.0              # people-region CLIP embeddings per second, per camera with 2+ people
FIGHT_CLF_WINDOW_S = 2.0         # scored over this window
FIGHT_CLF_PAD = 0.2              # padding around the union of people boxes
FIGHT_CLF_THRESHOLD = 0.6        # window probability that counts as fighting: held-out normal CCTV peaks at 0.31, a real armed-robbery struggle reaches 0.70-0.77
FIGHT_CLF_PEOPLE_LOOKBACK_S = 5.0 # 1-2 boxes count as a pair if 2+ people were here this recently
FIGHT_CLF_SUSTAIN = 2            # consecutive positive windows (0.5 s apart) before an alert
FIGHT_MIN_RATIO = 0.6            # positive in >= 60% of pose frames over the window
FIGHT_GAP_FRAC = 0.5             # box gap < 0.5 x person width
FIGHT_DEPTH_FRAC = 0.25          # feet within this x person height of each other (same ground depth)
FIGHT_SIZE_RATIO = 1.4           # and similar apparent size
FIGHT_WRIST_SPEED = 1.2          # wrist/elbow speed, person heights per second
FIGHT_ENERGY = 0.15              # mean upper-body speed relative to the body, heights/s (both people)
FIGHT_EXTEND_COS = 0.5           # arm direction vs direction to the other person
THROW_CLASSES = {"bottle", "cup", "cell phone", "book", "sports ball", "backpack", "handbag"}
THROW_LOW_CONF = 0.2             # these small classes are tracked from this confidence (throwing only)
THROW_SPEED = 0.6                # object speed, frame diagonals per second
THROW_NEAR_WRIST = 0.35          # object within this x person height of a wrist = "in hand"
THROW_MIN_FRAMES = 3             # moving away for at least this many observations
WRIST_SPIKE_SPEED = 2.0          # person heights per second
# Flying object (throw detection without recognising the object): the
# detector barely sees a thrown bag/bottle mid-flight on CCTV (UMN: 0-0.25
# conf), so a small blob that moves in three consecutive analysed frames,
# outside every person, starting next to someone and travelling fast in a
# steady direction counts as a throw.
FLY_ENABLED = False              # measured: loose settings 8/17 UMN throws but 11 false alerts on vtest,
                                 # strict settings 0/17; not good enough to ship
FLY_DIFF_THRESHOLD = 22          # grey-level change for a moving pixel
FLY_MIN_AREA = 4                 # px, at a 320x240-equivalent scale
FLY_MAX_PERSON_FRAC = 0.35       # blob area <= this x the median person box area
FLY_START_NEAR = 1.0             # first sighting within this x person height of a person box
FLY_MIN_OBS = 4                  # linked sightings before it can count
FLY_MIN_TRAVEL = 1.5             # total travel >= this x the thrower's height
FLY_MIN_SPEED = 0.3              # frame diagonals per second
FLY_PERSON_MARGIN = 0.3          # blobs within this x box size of any person are limbs / box lag, not objects
FLY_STEADY_COS = 0.6             # successive steps point the same way
FLY_MAX_GAP_S = 0.6              # a track not seen this long is dropped
PANIC_MIN_PEOPLE = 3
PANIC_SPEED_RATIO = 2.5          # speed vs the person's own 10s baseline
PANIC_BASELINE_S = 10.0
PANIC_MIN_SPEED = 0.12           # frame diagonals per second
PANIC_AWAY_COS = 0.5
PANIC_SUSTAIN_S = 1.0
# Crowd dispersal (second panic trigger): runners blur and lose their tracks,
# so on real footage (UMN) the per-runner rule sees 1-4 runners while the
# crowd count collapses. Fires when both happen within a few seconds:
PANIC_DISPERSE_MIN = 5           # crowd of at least this many (median over the baseline window)...
PANIC_DISPERSE_FRAC = 0.5        # ...drops to at most this fraction of it...
PANIC_DISPERSE_WINDOW_S = 4.0    # ...within this many seconds...
PANIC_MOTION_RATIO = 2.0         # ...while frame motion is at least this x its recent median,
PANIC_DISPERSE_RUNNERS = 1       # ...or at least this many people are running (>= PANIC_MIN_SPEED): whole-frame
                                 # motion hardly moves for a dozen small runners (UMN 44 s: 1.7x), running does
SCENE_CUT_FRAC = 0.4             # more than this share of pixels changing at once = a cut / new source, not motion

# --- Blind-spot tracking between linked cameras ---------------------------------
GAP_ALERT_S = 60.0             # gap alert when longer than max(2 x link walking time, this)
MISSING_ALERT_S = 120.0        # left a linked camera and not seen anywhere for this long
ESCALATE_WINDOW_S = 900.0      # HIGH if the subject was in a HIGH/CRITICAL alert this recently

STREAM_JPEG_QUALITY = 80
STREAM_MAX_FPS = int(os.environ.get("ARGONYX_STREAM_MAX_FPS", "60"))
