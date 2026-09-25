"""Benchmark the pose behaviours (fight, throw, hands raised, person down,
panic) on real CCTV / research footage instead of live acting.

Each clip runs through the same detector + ByteTrack + yolov8n-pose +
behaviour rules as a live camera, on a simulated clock (so sustain windows
behave as in real time however fast the GPU is). A manifest says which
behaviour each clip should raise ("expect") or that it must stay quiet
(expect: []), and the tool prints per-clip hits and misses plus totals.

Manifest (JSON list):
    {"clip": "caviar/Fight_Chase.mpg", "expect": ["FIGHT"]}
    {"clip": "urfall_cam0/fall-01", "expect": ["PERSON_DOWN"], "hold_last_s": 5}
    {"clip": "caviar/Walk1.mpg", "expect": []}
    {"clip": "repo:sample_data/vtest.avi", "expect": []}      (relative to this repo)
A clip is a video file or a folder of numbered images (fps from "fps",
default 30). hold_last_s repeats the final frame (the person stays down
after a short fall clip ends).

Usage:
    python tools/eval_behaviour.py --root ../behaviour_datasets --manifest tools/behaviour_benchmark.json
    python tools/eval_behaviour.py --root ../behaviour_datasets --manifest ... --only Fight --save-events out.json
"""
import argparse
import collections
import glob
import json
import os
import sys
import types
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import behavior, config  # noqa: E402
from backend.cameras import CameraObjectDetector, SharedModels, resize_to_width  # noqa: E402
from backend.tracking import TrackManager  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
PROC_FPS = 8.0   # frames per second a live camera analyses (CAMERA_MAX_FPS-ish under load)


def frames(path: Path, fps_hint: float, hold_last_s: float):
    """(timestamp, frame) at the clip's own timeline."""
    if path.is_dir():
        files = sorted(glob.glob(str(path / "**" / "*.png"), recursive=True)
                       + glob.glob(str(path / "**" / "*.jpg"), recursive=True))
        fps = fps_hint or 30.0
        reader = (cv2.imread(f) for f in files)
    else:
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or fps_hint or 25.0

        def gen():
            while True:
                ok, f = cap.read()
                if not ok:
                    return
                yield f
        reader = gen()
    i, last = 0, None
    for f in reader:
        if f is None:
            continue
        yield i / fps, f, fps
        last = f
        i += 1
    if last is not None and hold_last_s:
        for k in range(int(hold_last_s * fps)):
            yield (i + k) / fps, last, fps


class _Alerts:
    def __init__(self):
        self.events = []

    def ingest_event(self, rule, message, ts, **kw):
        self.events.append({"t": round(ts - CLOCK0, 1), "rule": rule, "band": kw.get("band"), "message": message})

    def ranked(self, n):
        return []


CLOCK0 = 1_000_000.0


def run_clip(shared, path: Path, fps_hint=None, hold_last_s=0.0, upscale_to=None):
    clock = {"t": CLOCK0}
    behavior.time = types.SimpleNamespace(time=lambda: clock["t"])
    det = CameraObjectDetector(shared, int(PROC_FPS))
    tm = TrackManager()
    alerts = _Alerts()
    engine = behavior.BehaviourEngine(shared._lock, alerts)
    cam = types.SimpleNamespace(id="bench", pipeline=types.SimpleNamespace(
        last_tracks={}, zone_store=types.SimpleNamespace(list=lambda: [])))
    next_t, n, seconds = 0.0, 0, 0.0
    for ts, frame, fps in frames(path, fps_hint, hold_last_s):
        seconds = ts
        if ts + 1e-6 < next_t:
            continue
        next_t = ts + 1.0 / PROC_FPS
        clock["t"] = CLOCK0 + ts
        if upscale_to and frame.shape[1] < upscale_to:
            s = upscale_to / frame.shape[1]
            frame = cv2.resize(frame, (upscale_to, int(frame.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
        small = resize_to_width(frame, config.PROCESS_WIDTH)
        tracks = tm.update(det.infer(small), clock["t"])
        cam.pipeline.last_tracks = tracks
        engine.on_frame(cam, small, small.copy())
        n += 1
    return alerts.events, n, seconds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="Folder the manifest's clip paths are relative to")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--only", default="", help="Substring filter on clip paths")
    ap.add_argument("--upscale-to", type=int, default=0, help="Upscale narrower frames to this width first (small CCTV)")
    ap.add_argument("--save-events", default="")
    args = ap.parse_args()

    items = [m for m in json.load(open(args.manifest)) if args.only in m["clip"]]
    shared = SharedModels()
    tp = collections.Counter(); fn = collections.Counter(); fp = collections.Counter()
    quiet_clips = quiet_ok = 0
    saved = []
    for m in items:
        clip = m["clip"]
        path = (REPO / clip[5:]) if clip.startswith("repo:") else Path(args.root) / clip
        if not path.exists():
            print(f"  missing: {m['clip']}")
            continue
        events, n, secs = run_clip(shared, path, m.get("fps"), m.get("hold_last_s", 0.0),
                                   args.upscale_to or m.get("upscale_to"))
        fired = collections.Counter(e["rule"] for e in events)
        expect = set(m["expect"])
        for rule in expect:
            (tp if fired[rule] else fn)[rule] += 1
        extra = {r: c for r, c in fired.items() if r not in expect}
        for r in extra:
            fp[r] += 1
        if not expect:
            quiet_clips += 1
            quiet_ok += not fired
        verdict = "OK " if all(fired[r] for r in expect) and not extra else "BAD"
        print(f"{verdict} {m['clip']:42s} {secs:5.1f}s expect {sorted(expect) or '-'}  fired {dict(fired) or '-'}")
        for e in events[:6]:
            print(f"        t={e['t']:6.1f} {e['rule']:14s} {e['band'] or ''} {e['message']}")
        saved.append({"clip": m["clip"], "expect": sorted(expect), "events": events})
    print("\nper behaviour: detected / expected   (false alarms on other clips)")
    for rule in sorted(set(tp) | set(fn) | set(fp)):
        print(f"  {rule:14s} {tp[rule]}/{tp[rule] + fn[rule]}   false-alarm clips: {fp[rule]}")
    print(f"  quiet clips with no alert: {quiet_ok}/{quiet_clips}")
    if args.save_events:
        json.dump(saved, open(args.save_events, "w"), indent=1)


if __name__ == "__main__":
    main()
