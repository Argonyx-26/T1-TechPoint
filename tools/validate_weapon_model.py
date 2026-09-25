"""Validate a candidate weapon-detection model BEFORE trusting it.

This runs the exact same sustained-detection logic the live app uses
(backend.analytics.weapon_filter.WeaponTemporalFilter) against real
footage, so the numbers you get here are what you'd actually see in the
dashboard -- not a lab number that doesn't transfer.

Required: a video with NO weapons in it (false-positive check). This is
the single most important test -- a model that hallucinates weapons on
ordinary footage is worse than no model at all for a CRITICAL,
no-corroboration-needed alert.

Optional: a video that DOES contain the weapon, to sanity-check the model
actually detects it at all (recall), not just that it stays quiet.

Usage:
    python tools/validate_weapon_model.py \
        --weights runs/weapon_training/run1/weights/best.pt \
        --footage path/to/ordinary_footage_with_no_weapons.mp4 \
        --positive-footage path/to/footage_that_has_the_weapon.mp4
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from backend import config
from backend.analytics.weapon_filter import WeaponTemporalFilter


def run_footage(model, video_path, conf, window, min_hits, label, imgsz=640):
    """Returns (frames, duration_s, sustained_events, frames_with_hit).

    Only config.WEAPON_THREAT_CLASSES count, same as the live app -- the
    model's confusor classes (smartphone, wallet, ...) are not alerts."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    wf = WeaponTemporalFilter(window=window, min_hits=min_hits)
    last_fired: dict = {}  # class -> last-alerted timestamp, mirrors AlertManager's cooldown
    raw_hits = 0
    frames_with_hit = 0
    sustained_events = []  # confirmed AND past cooldown -- an actual dashboard alert
    frame_i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1
        timestamp = frame_i / fps if fps else float(frame_i)
        results = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)
        boxes = results[0].boxes
        classes_this_frame = set()
        if boxes is not None and len(boxes) > 0:
            for b in boxes:
                cls = model.names[int(b.cls[0])]
                if cls in config.WEAPON_THREAT_CLASSES:
                    raw_hits += 1
                    classes_this_frame.add(cls)
        frames_with_hit += bool(classes_this_frame)
        for cls in wf.update(classes_this_frame):
            prev = last_fired.get(cls)
            if prev is not None and (timestamp - prev) < config.ALERT_COOLDOWN_SECONDS:
                continue  # same cooldown window AlertManager applies -- not a new alert
            last_fired[cls] = timestamp
            sustained_events.append((frame_i, cls))
    cap.release()

    duration_s = frame_i / fps if fps else None
    print(f"\n--- {label}: {video_path} ---")
    duration_note = f" (~{duration_s:.0f}s at {fps:.1f}fps)" if duration_s else ""
    print(f"frames: {frame_i}{duration_note}")
    print(f"raw per-frame detections (conf>={conf}): {raw_hits}")
    print(f"dashboard-equivalent alerts (smoothed + {config.ALERT_COOLDOWN_SECONDS:.0f}s cooldown, same as live app): {len(sustained_events)}")
    for f, cls in sustained_events[:20]:
        print(f"  frame {f}: {cls}")
    if len(sustained_events) > 20:
        print(f"  ... and {len(sustained_events) - 20} more")
    return frame_i, duration_s, sustained_events, frames_with_hit


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, help="Path to the candidate .pt file")
    parser.add_argument("--footage", required=True, help="Path to a video with NO weapons in it")
    parser.add_argument("--positive-footage", default=None, help="Optional: a video that DOES contain the weapon")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--window", type=int, default=8, help="Must match backend config.py's smoothing window")
    parser.add_argument("--min-hits", type=int, default=5, help="Must match backend config.py's smoothing threshold")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    print(f"Model classes: {model.names}")

    _, duration_s, sustained, _ = run_footage(
        model, args.footage, args.conf, args.window, args.min_hits,
        "WEAPON-FREE FOOTAGE (false-positive check)",
    )

    print("\n" + "=" * 72)
    if not sustained:
        print("PASS: zero false alerts on this weapon-free clip.")
        print("(Still test on more/longer weapon-free footage before fully trusting it --")
        print("one clean clip is a good sign, not a guarantee.)")
    else:
        rate = f"~1 every {duration_s / len(sustained):.0f}s" if duration_s else f"{len(sustained)} total"
        print(f"FAIL: {len(sustained)} false CRITICAL alert(s) on weapon-free footage ({rate}).")
        print("Do not deploy this. Needs more/better training data, more epochs, or")
        print("hard-negative mining (add this exact footage's frames as labeled-empty")
        print("background images in the next training run).")
    print("=" * 72)

    if args.positive_footage:
        _, _, pos_sustained, _ = run_footage(
            model, args.positive_footage, args.conf, args.window, args.min_hits,
            "FOOTAGE WITH THE WEAPON (recall check)",
        )
        print("\n" + "=" * 72)
        if pos_sustained:
            print(f"PASS: {len(pos_sustained)} sustained detection(s) on footage that has the weapon.")
        else:
            print("FAIL: no sustained detection on footage that DOES contain the weapon --")
            print("this model is missing real weapons, not just over-triggering on nothing.")
        print("=" * 72)


if __name__ == "__main__":
    main()
