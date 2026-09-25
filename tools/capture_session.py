"""Record a named webcam capture session for weapon-model fine-tuning or
held-out validation.

Writes <out-dir>/<session>.mp4 plus <out-dir>/<session>.json, a sidecar that
records which frame ranges were tagged "phone" vs "other" (bottles, pens,
wallets, remotes, chargers). label_capture.py uses the tag to decide whether
a frame should carry a smartphone box or be an empty-label background frame,
so press the key BEFORE switching what you're holding.

Keys in the preview window:
    p  = now holding PHONES            (default for phones_* sessions)
    o  = now holding OTHER objects     (bottles/pens/wallets/remotes/chargers)
    w  = now holding the WEAPON        (default for weapons_* sessions)
    q  = stop early

Usage:
    python tools/capture_session.py --session phones_train --minutes 5
    python tools/capture_session.py --session weapons_test --minutes 2
"""
import argparse
import json
import time
from pathlib import Path

import cv2

TAG_KEYS = {ord("p"): "phone", ord("o"): "other", ord("w"): "weapon"}
TAG_COLORS = {"phone": (0, 200, 0), "other": (255, 178, 60), "weapon": (0, 0, 255)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True, help="e.g. phones_train, weapons_train, phones_test, weapons_test")
    parser.add_argument("--minutes", type=float, required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--out-dir", default="data/captures")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--tag", choices=["phone", "other", "weapon"], default=None,
                        help="What you start out holding (default: weapon for weapons_* sessions, else phone)")
    parser.add_argument("--countdown", type=int, default=5, help="Seconds of preview before recording starts")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / f"{args.session}.mp4"
    meta_path = out_dir / f"{args.session}.json"
    if video_path.exists():
        raise SystemExit(f"{video_path} already exists -- delete it first if you want to re-record this session.")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open webcam {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Webcam opened but returned no frame")
    h, w = frame.shape[:2]

    tag = args.tag or ("weapon" if args.session.startswith("weapon") else "phone")
    window = f"capture: {args.session}"

    # Countdown so you can get into position; nothing is recorded yet.
    start = time.time()
    while time.time() - start < args.countdown:
        ok, frame = cap.read()
        if not ok:
            break
        remaining = args.countdown - (time.time() - start)
        cv2.putText(frame, f"Starting in {remaining:.0f}s", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            raise SystemExit("Cancelled before recording started.")
        if key in TAG_KEYS:
            tag = TAG_KEYS[key]

    # Frames stream straight to disk (a 5 min buffer would cost >1GB RAM).
    # The container fps is fixed up afterwards from the MEASURED rate, since
    # CAP_PROP_FPS often claims 30 when a webcam delivers 15 in dim light --
    # the validation cooldown math relies on the file's timeline being real.
    tmp_path = out_dir / f"{args.session}.raw.mp4"
    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    segments = [{"start_frame": 0, "tag": tag}]
    duration_s = args.minutes * 60
    rec_start = time.time()
    frame_i = 0
    while True:
        elapsed = time.time() - rec_start
        if elapsed >= duration_s:
            break
        ok, frame = cap.read()
        if not ok:
            print("Webcam stopped delivering frames; ending early.")
            break
        writer.write(frame)
        frame_i += 1

        preview = frame.copy()
        color = TAG_COLORS[tag]
        cv2.circle(preview, (30, 35), 12, (0, 0, 255), -1)
        cv2.putText(
            preview, f"REC {args.session}  {elapsed:5.0f}/{duration_s:.0f}s  holding: {tag.upper()}",
            (55, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
        )
        cv2.putText(preview, "p=phone  o=other objects  w=weapon  q=stop", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(window, preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key in TAG_KEYS and TAG_KEYS[key] != tag:
            tag = TAG_KEYS[key]
            segments.append({"start_frame": frame_i, "tag": tag})

    elapsed = time.time() - rec_start
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    fps = frame_i / elapsed if elapsed > 0 else 30.0
    if abs(fps - 30.0) < 1.0:
        tmp_path.rename(video_path)
    else:
        src = cv2.VideoCapture(str(tmp_path))
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        while True:
            ok, frame = src.read()
            if not ok:
                break
            writer.write(frame)
        src.release()
        writer.release()
        tmp_path.unlink()

    meta = {"session": args.session, "frames": frame_i, "fps": round(fps, 2), "size": [w, h], "segments": segments}
    meta_path.write_text(json.dumps(meta, indent=2))
    tag_counts = {}
    bounds = [s["start_frame"] for s in segments[1:]] + [frame_i]
    for seg, end in zip(segments, bounds):
        tag_counts[seg["tag"]] = tag_counts.get(seg["tag"], 0) + end - seg["start_frame"]
    print(f"Saved {video_path} ({frame_i} frames, {elapsed:.0f}s, {fps:.1f} fps)")
    print(f"Frames per tag: {tag_counts}")


if __name__ == "__main__":
    main()
