"""run17 dataset: run15's list + hard negatives (things mistaken for a knife)
+ labelled CCTV weapons (Mock Attack Cam1/Cam7; Cam5 is held out for testing).

Hard negatives get empty label files ("nothing here is a weapon") and are
repeated --oversample times so a few hundred frames count against ~12k.

Usage:
    python tools/build_run17.py [--extra-clips data/captures/phones_edge.mp4 ...]
"""
import argparse
import glob
import os
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "finetune" / "run17"
MOCK = REPO.parent / "weapon_datasets" / "mock" / "Images"
CLASS = {"Handgun": 0, "Short_rifle": 0, "Knife": 2}   # model classes: 0 pistol, 2 knife


def write(img, stem, lines):
    cv2.imwrite(str(OUT / "images" / f"{stem}.jpg"), img)
    (OUT / "labels" / f"{stem}.txt").write_text("\n".join(lines))
    return str(OUT / "images" / f"{stem}.jpg")


def clip_negatives(path, every):
    cap, i, out = cv2.VideoCapture(str(path)), -1, []
    while True:
        ok, f = cap.read(); i += 1
        if not ok:
            break
        if i % every == 0:
            out.append(write(f, f"neg_{Path(path).stem}_{i:05d}", []))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra-clips", nargs="*", default=[])
    ap.add_argument("--oversample", type=int, default=3)
    ap.add_argument("--mock-negatives", type=int, default=300)
    args = ap.parse_args()
    for d in ("images", "labels"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    random.seed(0)

    negatives = []
    for clip in ["data/captures/confusers_mouse.mp4", *args.extra_clips]:
        negatives += clip_negatives(REPO / clip, every=3)
    import json
    for j in glob.glob(str(REPO / "debug" / "confirmed_alerts" / "*.jpg")):
        # Saved alerts reviewed by eye (2026-09-26): phone edge-on / mouse on the
        # webcam, a steel bottle, UMN crowd scenes = false. The 480x270 ones are
        # the m2-res_480p rifle clip = REAL guns: never use them as negatives.
        meta = json.loads(Path(j[:-4] + ".json").read_text())
        if list(meta["frame_size"]) == [480, 270]:
            continue
        negatives.append(write(cv2.imread(j), f"neg_alert_{Path(j).stem}", []))

    positives, mock_neg = [], []
    for x in sorted(glob.glob(str(MOCK / "*.xml"))):
        stem = Path(x).stem
        if stem.startswith("Cam5"):
            continue                       # held out
        root = ET.parse(x).getroot()
        w, h = float(root.find("size/width").text), float(root.find("size/height").text)
        lines = []
        for o in root.findall("object"):
            b = o.find("bndbox")
            x1, y1, x2, y2 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
            lines.append(f"{CLASS[o.find('name').text]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                         f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        (positives if lines else mock_neg).append((x[:-4] + ".jpg", stem, lines))
    mock_neg = random.sample(mock_neg, min(args.mock_negatives, len(mock_neg)))
    cctv = []
    for jpg, stem, lines in positives + mock_neg:
        img = cv2.imread(jpg)
        img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)   # half size: keeps small guns visible at 640 training
        cctv.append(write(img, f"mock_{stem}", lines))

    base = (REPO / "data" / "finetune" / "run15_train.txt").read_text().splitlines()
    train = base + cctv + negatives * args.oversample
    (REPO / "data" / "finetune" / "run17_train.txt").write_text("\n".join(train))
    val = r"C:\Users\dhyey\Downloads\od-weapondetection\Weapons and similar handled objects\Sohas_weapon-Detection-YOLOv5\obj_train_data\images\test"
    (REPO / "data" / "finetune" / "run17_data.yaml").write_text(
        f"train: {REPO / 'data' / 'finetune' / 'run17_train.txt'}\nval: {val}\nnc: 6\n"
        "names:\n- pistol\n- smartphone\n- knife\n- monedero\n- billete\n- tarjeta\n")
    print(f"run15 base {len(base)} + CCTV weapons {len(positives)} + CCTV empty {len(mock_neg)} "
          f"+ hard negatives {len(negatives)} x{args.oversample} = {len(train)} images")


if __name__ == "__main__":
    sys.exit(main())
