"""Draw YOLO-format label boxes onto their images so you can visually verify
them -- e.g. after hard-positive mining, before trusting the auto-generated
labels enough to retrain on them.

Usage:
    python tools/preview_labels.py \
        --images-dir path/to/obj_train_data/images/train \
        --labels-dir path/to/obj_train_data/labels/train \
        --prefix hardpos \
        --output-dir path/to/preview_out \
        --class-names pistol smartphone knife monedero billete tarjeta
"""
import argparse
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--prefix", required=True, help="Only preview images whose filename starts with this (e.g. 'hardpos')")
    parser.add_argument("--output-dir", required=True, help="Where to save annotated preview copies")
    parser.add_argument("--class-names", nargs="+", required=True, help="Class names in index order, e.g. pistol smartphone knife monedero billete tarjeta")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matches = sorted(p for p in images_dir.glob(f"{args.prefix}*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
    if not matches:
        raise SystemExit(f"No images found matching prefix '{args.prefix}' in {images_dir}")

    saved = 0
    for image_path in matches:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]

        for line in label_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls_idx = int(parts[0])
            cx, cy, bw, bh = [float(v) for v in parts[1:5]]
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            cls_name = args.class_names[cls_idx] if cls_idx < len(args.class_names) else str(cls_idx)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, cls_name, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imwrite(str(output_dir / image_path.name), frame)
        saved += 1

    print(f"Saved {saved} annotated preview image(s) to {output_dir}")
    print("Open that folder and look at each one -- if the red box isn't clearly on the object,")
    print(f"delete the matching pair from {images_dir} and {labels_dir} before retraining.")


if __name__ == "__main__":
    main()
