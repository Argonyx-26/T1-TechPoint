"""Weapon detection on real CCTV with labelled weapons (frame level).

Dataset: "Real-time gun detection in CCTV: An open problem" (Salazar-Gonzalez
et al., Neural Networks 2020, CC BY-NC 4.0), Mock Attack: 5149 frames of
1920x1080 university CCTV (Cam1/5/7), Handgun / Short_rifle / Knife boxes,
3615 frames without a weapon.

For each frame: did the detector put a weapon box (>= --conf) on a labelled
weapon (per class: recall), and does it put one anywhere on a weapon-free
frame (false-alarm rate). --mode picks how the frame reaches the model:
  pipeline  shrunk to PROCESS_WIDTH first, as the live pipeline does
  full      the full-resolution frame
Detections are class-agnostic hits: any pistol/knife box on a rifle counts,
since the alert is "weapon detected".

Usage:
    python tools/eval_weapons_cctv.py --root ../weapon_datasets/mock/Images --mode pipeline
    python tools/eval_weapons_cctv.py --root ... --mode full --conf 0.45 --save per_frame.json
"""
import argparse
import collections
import glob
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from backend import config  # noqa: E402
from backend.cameras import resize_to_width  # noqa: E402


def load_boxes(xml_path):
    root = ET.parse(xml_path).getroot()
    out = []
    for o in root.findall("object"):
        b = o.find("bndbox")
        out.append((o.find("name").text, tuple(float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))))
    return out


def hit(det, gt, margin=0.5):
    """Detection centre inside the labelled box (grown by margin x its size)."""
    cx, cy = (det[0] + det[2]) / 2, (det[1] + det[3]) / 2
    w, h = gt[2] - gt[0], gt[3] - gt[1]
    return gt[0] - margin * w <= cx <= gt[2] + margin * w and gt[1] - margin * h <= cy <= gt[3] + margin * h


def make_detector(kind, model_path=None):
    if kind == "run15":
        from backend.detection.weapon_detector import WeaponDetector
        wd = WeaponDetector(model_path=model_path)
        return lambda img, conf: [(d.cls_name, d.conf, d.bbox) for d in wd.infer(img) if d.conf >= conf]
    if kind.startswith("yoloworld"):
        # open-vocabulary detector described in words (option B: no training)
        from ultralytics import YOLOWorld
        m = YOLOWorld(model_path or "yolov8s-worldv2.pt")
        m.set_classes(WORLD_CLASSES)
        return lambda img, conf: [
            (m.names[int(b.cls)], float(b.conf), tuple(float(v) for v in b.xyxy[0].tolist()))
            for b in m.predict(img, conf=conf, imgsz=1280, verbose=False)[0].boxes]
    raise ValueError(kind)


WORLD_CLASSES = ["rifle", "shotgun", "handgun", "pistol", "gun", "knife"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--mode", choices=["pipeline", "full"], default="pipeline")
    ap.add_argument("--detector", default="run15")
    ap.add_argument("--model", default=None, help="weapon weights (default: config.WEAPON_MODEL_PATH)")
    ap.add_argument("--conf", type=float, default=config.WEAPON_ALERT_MIN_CONF)
    ap.add_argument("--every", type=int, default=1, help="use every Nth frame")
    ap.add_argument("--save", default="")
    ap.add_argument("--prefix", default="", help="only frames whose file name starts with this, e.g. Cam5 (held out)")
    args = ap.parse_args()

    detect = make_detector(args.detector, args.model)
    files = [f for f in sorted(glob.glob(os.path.join(args.root, "*.xml"))) if Path(f).name.startswith(args.prefix)][:: args.every]
    per_class = collections.Counter(); per_class_hit = collections.Counter()
    neg = neg_fp = 0
    rows = []
    for x in files:
        img = cv2.imread(x[:-4] + ".jpg")
        if img is None:
            continue
        gts = load_boxes(x)
        scale = 1.0
        if args.mode == "pipeline":
            small = resize_to_width(img, config.PROCESS_WIDTH)
            scale = img.shape[1] / small.shape[1]
            img = small
        dets = [(c, p, tuple(v * scale for v in b)) for c, p, b in detect(img, args.conf)]
        if not gts:
            neg += 1
            neg_fp += bool(dets)
        for name, g in gts:
            per_class[name] += 1
            per_class_hit[name] += any(hit(d[2], g) for d in dets)
        rows.append({"frame": Path(x).stem, "gt": [n for n, _ in gts],
                     "dets": [(c, round(p, 3), [round(v) for v in b]) for c, p, b in dets]})
    print(f"mode={args.mode} detector={args.detector} conf>={args.conf}  frames={len(rows)}")
    for name in sorted(per_class):
        print(f"  {name:12s} found in {per_class_hit[name]:4d}/{per_class[name]:4d} frames = {per_class_hit[name] / per_class[name]:.1%}")
    print(f"  weapon-free frames with a false weapon box: {neg_fp}/{neg} = {neg_fp / max(neg, 1):.1%}")
    if args.save:
        json.dump(rows, open(args.save, "w"))


if __name__ == "__main__":
    main()
