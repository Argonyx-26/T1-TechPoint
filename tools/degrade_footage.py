"""Synthetically degrade clear footage to approximate real CCTV/security-camera
quality: low resolution, heavy compression artifacts, sensor noise.

Why this exists: if you can't physically reposition a camera far from the
subject, you can still close part of the domain gap between "clear webcam
footage" and "real CCTV footage" by simulating the image degradation a real
security camera and its video pipeline introduce. This does NOT replicate a
true change in camera distance/perspective (the subject won't get smaller
in the frame) -- it only degrades per-pixel quality: downscale+upscale
(loses fine detail, like a lower-resolution sensor), heavy JPEG
recompression (blocky artifacts), and added noise (grain, like a real
camera's sensor in imperfect lighting). Combine with genuine distance
footage when you can get it -- this is a partial substitute, not a
replacement.

Usage:
    python tools/degrade_footage.py \
        --input path/to/clear_video.mp4 \
        --output path/to/degraded_video.mp4 \
        --scale 0.35 \
        --jpeg-quality 25 \
        --noise 8
"""
import argparse

import cv2
import numpy as np


def degrade_frame(frame, scale: float, jpeg_quality: int, noise: float) -> "np.ndarray":
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_LINEAR)
    upscaled = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    ok, encoded = cv2.imencode(".jpg", upscaled, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if ok:
        upscaled = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    if noise > 0:
        gaussian_noise = np.random.normal(0, noise, upscaled.shape).astype(np.float32)
        upscaled = np.clip(upscaled.astype(np.float32) + gaussian_noise, 0, 255).astype(np.uint8)

    return upscaled


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Clear source video")
    parser.add_argument("--output", required=True, help="Where to write the degraded video (.mp4)")
    parser.add_argument("--scale", type=float, default=0.35, help="Downscale factor before upscaling back (lower = blurrier, more detail lost)")
    parser.add_argument("--jpeg-quality", type=int, default=25, help="JPEG re-encode quality 1-100 (lower = blockier artifacts)")
    parser.add_argument("--noise", type=float, default=8.0, help="Gaussian noise std-dev added per pixel (0 = none, higher = grainier)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.input}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    frame_i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1
        writer.write(degrade_frame(frame, args.scale, args.jpeg_quality, args.noise))

    cap.release()
    writer.release()
    print(f"Degraded {frame_i} frame(s). Wrote: {args.output}")
    print("\nSanity-check a few frames before mining -- it should look noticeably")
    print("grainier/blockier than the source, but the object should still be")
    print("recognizable (if it's unrecognizable even to you, back off --scale/--noise).")


if __name__ == "__main__":
    main()
