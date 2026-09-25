"""Remove another detector's drawings from benchmark footage.

The UMN videos are demo recordings of UMN's own detector: an "Abnormal Crowd
Activity" / "THROWN OBJECT" banner appears exactly when the event happens,
and thrown objects get a red trajectory line. Left in, those overlays can
trigger our rules by themselves (a banner popping up is sudden motion; a red
line appearing beside a hand looks like an object leaving it), so benchmark
numbers on the raw videos are not trustworthy.

This writes cleaned copies:
  - the banner corner is covered by the SAME constant grey patch in every
    frame, so it never contributes motion
  - saturated red trajectory lines are inpainted, only in frames where the
    banner is showing (people in red clothes elsewhere are left alone)

Usage:
    python tools/clean_overlays.py ../behaviour_datasets/umn/Thrown-Object-All.avi ../behaviour_datasets/umn_clean/Thrown-Object-All.avi
"""
import sys

import cv2
import numpy as np

BANNER = (0, 0, 172, 28)        # x1, y1, x2, y2 in the native 320x240 frame


def red_mask(frame):
    b, g, r = (frame[:, :, k].astype(int) for k in range(3))
    return (r > 150) & (g < 90) & (b < 90)


def clean(src, dst):
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sx, sy = w / 320.0, h / 240.0
    x1, y1, x2, y2 = int(BANNER[0] * sx), int(BANNER[1] * sy), int(BANNER[2] * sx), int(BANNER[3] * sy)
    out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    n = drawn = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        banner_on = red_mask(frame[y1:y2, x1:x2]).sum() > 60
        if banner_on:
            lines = red_mask(frame)
            lines[y1:y2, x1:x2] = False
            if lines.sum():
                drawn += 1
                m = cv2.dilate(lines.astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
                frame = cv2.inpaint(frame, m, 3, cv2.INPAINT_TELEA)
        frame[y1:y2, x1:x2] = 128
        out.write(frame)
    out.release()
    print(f"{src} -> {dst}: {n} frames, trajectory lines removed in {drawn}")


if __name__ == "__main__":
    clean(sys.argv[1], sys.argv[2])
