# Argonyx - Intelligent Threat Detection & Situational Awareness

Analyzes a live or uploaded video feed to detect anomalous behavior, flag
potential security threats, prioritize alerts by severity, and give an
analyst a single dashboard to act on it.

## Core capabilities

1. **General object/person detection** - pretrained YOLOv8 (COCO), works
   out of the box on any footage: detects people and objects, and tracks
   them frame-to-frame.
2. **Behavior-based anomaly rules** on top of detection output:
   - Loitering (person stationary in one zone too long)
   - Crowd surge (person count exceeds a threshold in a zone)
   - Unattended object (an item appears, its owner leaves, it remains)
   - Restricted zone intrusion (a person enters a marked no-go area)
   - Wrong-direction movement (entry/exit lane misuse)
3. **Weapon detection** - a separate, pluggable pretrained YOLO model
   (gun/knife), run alongside the general model. Any hit is immediately
   flagged critical severity, no corroboration needed.
4. **Severity scoring + prioritization** - weapon > unattended object >
   restricted intrusion > crowd surge > wrong-direction > loitering, with a
   bonus when multiple alert types co-occur in the same zone/window.
5. **Analyst dashboard** - live feed with bounding-box + zone overlays, a
   ranked alert sidebar, click-through evidence (the frame + what
   triggered it), and an upload-your-own-footage control.

## Architecture

```
backend/
  config.py            thresholds, model paths, storage paths (all in one place)
  geometry.py           point-in-polygon / vector math, no external deps
  tracking.py           per-track motion history (speed, dwell, direction)
  zones.py               zone data model + JSON-backed store
  detection/
    object_detector.py   YOLOv8 general detector + tracker wrapper
    weapon_detector.py    pluggable weapon model wrapper (degrades gracefully)
  analytics/
    rules.py              the 5 behavior rules, pure logic over tracks + zones
    severity.py            weight + co-occurrence scoring
  alerts/
    manager.py             dedup/cooldown, ranking, evidence capture
  pipeline.py            wires detector -> tracker -> rules -> alerts -> overlay
  video_source.py         swappable webcam/file capture
  app_state.py             background processing thread + shared frame buffer
  main.py                  FastAPI app: dashboard, video stream, REST API
frontend/
  index.html / style.css / app.js   the dashboard (vanilla JS, no build step)
tests/                   unit tests for geometry/rules/severity (no ML needed)
```

Each rule is pure geometry/time logic over `TrackState` objects, so the
whole rules engine is unit-tested without needing a GPU or any model
weights (see `tests/`).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The general detector (`yolov8n.pt`) is downloaded automatically by
`ultralytics` on first use - no manual setup required. It's a COCO-trained
model: it knows "person", "backpack", "bottle", etc., but **has no concept
of weapons or of specific objects like a "can"** - see the note below.

### GPU acceleration (CUDA)

`backend/config.py` auto-detects a CUDA GPU via `torch.cuda.is_available()`
and switches to it (with fp16 inference) with no configuration needed --
*if* the `torch` in your virtualenv was built with CUDA support. The
`torch`/`torchvision` that plain `pip install -r requirements.txt` pulls in
from PyPI is often the **CPU-only** build. To actually use an RTX 3050 (or
any CUDA GPU), reinstall the CUDA build over it, matching your installed
CUDA version (check with `nvidia-smi`; cu121 works for most recent drivers):

```bash
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify it took:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Once that prints `True` and your GPU's name, just start the server as usual
-- `/api/status`'s `fps` field will jump substantially, and the dashboard's
top bar shows the active device isn't reported directly but you'll see it
in the fps. To force a specific device regardless of auto-detection, set
`ARGONYX_DEVICE=cuda:0` (or `cpu`) as an environment variable before
starting uvicorn.

`STREAM_MAX_FPS` (env var `ARGONYX_STREAM_MAX_FPS`, default `60`) caps how
fast the annotated stream is re-encoded and served -- raise or lower it to
match what your GPU can actually sustain; the real bottleneck at high fps
is usually the YOLO inference step itself, not the streaming.

For more accuracy at some speed cost, swap the general model to a larger
COCO-trained variant via `ARGONYX_GENERAL_MODEL=yolov8s.pt` (or
`yolov8m.pt`) -- ultralytics downloads it automatically the same way.

### Connecting a phone camera (e.g. DroidCam)

The dashboard has a "Camera index or DroidCam URL" field next to "Use
Webcam". Two ways DroidCam exposes your phone's camera:

- **USB mode**: DroidCam installs a virtual webcam device on your PC,
  usually at index `1` or `2` (index `0` is normally your laptop's built-in
  cam). Try `1`, then `2`, in that field and hit Connect.
- **WiFi mode**: the DroidCam app shows an IP address; use
  `http://<that-ip>:4747/video` in the field. This works for any IP-camera
  app that exposes an MJPEG/HTTP stream, not just DroidCam.

This hits `POST /api/source/camera`, which accepts a bare index or any URL
`cv2.VideoCapture` understands.

### Weapon detection (optional) -- and its real limits

The weapon detector needs its own weights (COCO has no gun/knife classes,
and no "can" class either -- accurately detecting either requires a model
actually trained on labeled images of it, not a threshold tweak on the
general detector).

**A word of warning before you go looking for a shortcut here.** We tested
a randomly-found "pretrained weapon detection" model from GitHub against
real, weapon-free footage: it produced 95 false-positive detections across
~160 sampled frames, and even after adding sustained-detection smoothing
(`backend/analytics/weapon_filter.py`), it still fired a false CRITICAL
alert roughly every 9 seconds on video with no weapons in it. Most small,
hobbyist weapon-detection models floating around online have this problem
-- they're trained on a few hundred staged photos and don't generalize.
Don't drop a random `.pt` file into `models/weapon.pt` and trust it; use
`tools/validate_weapon_model.py` (below) on **any** model before trusting
it, including one you find pretrained.

**The real path to a usable weapon detector, in order:**

1. **Get a real, sizeable labeled dataset.** [Roboflow Universe](https://universe.roboflow.com)
   has several public knife/gun detection datasets with thousands of
   annotated images (search "knife detection" or "weapon detection", check
   image count and label quality, prefer ones with images that look like
   your actual deployment -- varied backgrounds/lighting, not staged studio
   shots). Make a free account, open a dataset, and use its "Download
   Dataset" button with format **YOLOv8** -- this gives you a zip with a
   `data.yaml` plus `train/valid/test` image+label folders.

2. **Train.** Extract that zip anywhere in the project, then:
   ```bash
   python tools/train_weapon_model.py --data path/to/data.yaml
   ```
   This fine-tunes `yolov8n.pt` on your dataset using your GPU (auto-detected;
   pass `--device 0` to force it). Expect roughly 20-90 minutes on an RTX
   3050 depending on dataset size and epoch count -- run `--help` for all
   options (epochs, image size, batch size). It prints the path to the best
   checkpoint when done (`runs/weapon_training/run1/weights/best.pt`).

3. **Validate -- do not skip this.** Point it at footage from your own
   camera setup that has **no** weapons in it:
   ```bash
   python tools/validate_weapon_model.py \
     --weights runs/weapon_training/run1/weights/best.pt \
     --footage path/to/your_own_weapon_free_footage.mp4 \
     --positive-footage path/to/footage_that_has_the_weapon.mp4
   ```
   This runs the exact same sustained-detection logic the live dashboard
   uses, so the false-positive count it reports is what you'd actually see
   in the alert sidebar. The optional `--positive-footage` checks the model
   actually detects the weapon at all (a model can pass the false-positive
   check trivially by never detecting anything). Only copy the weights to
   `models/weapon.pt` once both checks look reasonable -- and even then,
   keep watching real usage for a while before fully trusting it.

4. **If validation fails** (false positives on your footage), the fix is
   more/better training data -- specifically, add the exact frames that
   triggered false positives as labeled-empty "background" images in your
   next training run (this is called hard-negative mining) and retrain.

Once `models/weapon.pt` exists and passes validation, the system picks it
up automatically on the next start -- no code changes needed. The same
approach applies to detecting a specific object class like "can": find or
build a dataset that has that class, fine-tune, validate the same way.

### Sample footage

Drop any `.mp4`/`.avi`/`.mov`/`.mkv` into `sample_data/` and it becomes
selectable from the dashboard's "Sample clips" dropdown; the server also
auto-plays the first one found on startup if no webcam is available.

## Running

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open **`http://127.0.0.1:8000`** on the laptop, not `http://localhost:8000`:
Windows resolves `localhost` to IPv6 `::1` first, and every request then
pays ~0.2 s falling back to IPv4.

`--host 0.0.0.0` also makes the server reachable from phones on the same
Wi-Fi / hotspot (needed for the phone-GPS locate links, `http://<laptop LAN
ip>:8000/locate/CAM-02`). Use `--host 127.0.0.1` to keep it laptop-only.
Skip `--reload` for demos: it restarts the server (and every camera) on
each file save.

**Keeping the cameras fast with the dashboard open.** The browser renders on
the same CPU the detector needs. The dashboard therefore makes one batched
`GET /api/dashboard` call every 2 s (status, alerts, cameras, movements),
verifies the audit chain on load and once a minute, refreshes camera tiles
at 2 frames/s only while they are on screen (the big view streams), and
uses no blur or looping animations. Measured with 4 cameras (webcam x2 +
two crowd clips) on an RTX 3050: old dashboard 7.9 -> 4.0 fps when opened,
now 7.9 -> 7.8. If fps is still short, the analytics modules panel shows
per-camera fps; search (CLIP) and pose are the optional costs.

### Behaviour detection (pose): fighting, throwing, distress

yolov8n-pose runs per camera at up to 6 fps, only while people are in view.
Every rule must hold over time (never one frame) and is worded "possible":

| Alert | Fires when | Band |
|---|---|---|
| Possible fight | two people close, at the same depth, fast arm strikes toward each other in >= 60% of pose frames over 2 s | HIGH |
| Object thrown | a tracked bottle / phone / bag (or an unknown blob) leaves a fast-moving hand | MEDIUM, HIGH toward a person or a restricted zone |
| Hands raised | both wrists above the nose for 2 s | HIGH; CRITICAL if a weapon was confirmed on that camera in the last 30 s |
| Person down | horizontal for 3 s after being seen upright within 10 s (a fall, not a sleeper) | HIGH |
| Possible panic | 3+ people suddenly running (2.5x their own speed) away from one point | HIGH, origin marked |

Body language only: no face or emotion analysis. Replayed over vtest,
confusers, the gate/lobby simulations, the phone capture and the knife CCTV
clip (about 6 minutes of footage) these raised no behaviour alerts. The
**Analytics Modules** panel switches each module (pose, fighting, throwing,
distress, search, re-ID, gap tracking) on or off live and shows per-camera
fps; the switches are kept in `data/features.json` across restarts.

### Putting cameras on the map (accurately)

A laptop's "Use my location" over a phone hotspot is IP / Wi-Fi based and can
be kilometres off, so the dashboard never trusts a fix worse than 200 m: it
shows the accuracy circle and says "Approximate location (±X km) - search
your address, drag the pin, or use phone GPS". Better ways, best first:

1. **Phone GPS.** Each camera tile has a **GPS** button (also in the location
   editor) showing a QR code for `https://<laptop LAN ip>:8443/locate/CAM-02?k=...`.
   Stand next to the camera, scan it with a phone on the same Wi-Fi / hotspot,
   tap *Advanced -> Proceed* on the certificate warning, allow location. The
   camera is placed once GPS is within 20 m ("Located via phone GPS ±8 m")
   and the dashboard flies there. Phones only give GPS to HTTPS pages, so the
   server also listens on port 8443 with a self-signed certificate for the
   LAN IP (`data/tls/`, re-issued when the IP changes); that port serves only
   the locate page and its POST, nothing else. Change it with
   `ARGONYX_LOCATE_PORT` (0 turns it off). Windows may ask once to allow
   Python through the firewall: allow it on private networks.
2. **Search.** STREET MAP -> type an address or place ("Andheri Station,
   Mumbai") -> pick a result. Uses OpenStreetMap Nominatim through the
   backend (proper User-Agent, max 1 request/s, cached; set
   `ARGONYX_NOMINATIM_CONTACT` to add your contact to the User-Agent).
3. **Drag.** Every pin on the street map is draggable; it saves on drop.

**SET SITE HERE** stores the site in `data/cameras.json`; the map then opens
on the placed cameras (or the site when none are placed yet), never on a
guessed city. A new alert flies the map to its camera and opens the popup;
NAVIGATE opens Google Maps directions to that exact point. The layer button
(top right) switches to **Satellite** (Esri World Imagery).

From the dashboard you can:

- Switch between webcam, an uploaded file, or a bundled sample clip.
- Draw a zone directly on the video (click points, "Finish Polygon"), then
  mark it restricted, give it a crowd-count threshold, and/or a loiter
  time limit.
- Watch alerts stream into the ranked sidebar in real time; click one to
  see the evidence frame and details.

## Tests

```bash
pytest tests/
```

These exercise the rules engine and severity scoring against synthetic
track data - no camera, model weights, or GPU required.

## Runtime tuning

All thresholds (`loiter_seconds`, `crowd_threshold`,
`unattended_seconds`, `unattended_radius_px`, `stationary_speed_px_s`,
`wrong_direction_angle_deg`) are readable/writable at runtime via
`GET`/`POST /api/thresholds`, and severity weights live in
`backend/config.py`.
