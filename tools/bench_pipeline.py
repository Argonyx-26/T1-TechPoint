"""Run clips through the real per-frame pipeline as fast as possible (no playback pacing)
and report processing FPS and confirmed weapon events.

    python -m tools.bench_pipeline test_videos/vtest.avi test_videos/confusers.mp4
    python -m tools.bench_pipeline --no-weapon test_videos/vtest.avi
"""
import argparse
import time

import cv2

from backend.pipeline import Pipeline


def run_clip(pipe: Pipeline, path: str, max_frames: int | None) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    pipe.reset_state()
    frames = raw_hits = events = confirmed_frames = 0
    prev_confirmed: set = set()
    t0 = time.perf_counter()
    while max_frames is None or frames < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        pipe.process_frame(frame)
        frames += 1
        raw_hits += sum(d.conf >= 0.5 for d in pipe.weapon_dets)
        if pipe.confirmed_weapons:
            confirmed_frames += 1
        events += len(pipe.confirmed_weapons - prev_confirmed)  # rising edges only
        prev_confirmed = set(pipe.confirmed_weapons)
    elapsed = time.perf_counter() - t0
    cap.release()
    return {"frames": frames, "fps": frames / elapsed, "raw_hits_ge_0.5": raw_hits,
            "confirmed_frames": confirmed_frames, "confirmed_events": events}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--no-weapon", action="store_true", help="general detector + tracking only")
    ap.add_argument("--max-frames", type=int)
    args = ap.parse_args()

    pipe = Pipeline(use_weapon_model=not args.no_weapon)
    pipe.load()
    label = "without weapon model" if args.no_weapon else "with weapon model"
    for clip in args.clips:
        r = run_clip(pipe, clip, args.max_frames)
        print(f"{clip} [{label}]: " + ", ".join(
            f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))


if __name__ == "__main__":
    main()
