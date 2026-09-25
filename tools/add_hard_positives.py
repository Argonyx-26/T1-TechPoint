"""Hard-positive mining: turn a model's own (correct but under-confident)
detections into stronger training data for YOUR specific domain.

Hard-negative mining (add_hard_negatives.py) fixes false positives by
teaching the model "nothing here." This is the mirror image: it fixes
low-confidence/inconsistent TRUE positives by feeding the model more
examples of the real object, in your actual camera/lighting/room, labeled
with the box it already (mostly correctly) drew.

Why this is needed even after a model already detects the object: a model
trained on someone else's dataset (different camera, lighting, distance,
compression) can land right at the edge of confident on your footage even
when it's basically right. More epochs on the original dataset won't fix
that -- it needs examples FROM your domain. This script auto-labels those
examples using the model's own boxes, which only works because the model
is already close (not systematically wrong) on this class.

Only mines ONE class at a time (--class-name) and drops every other
detected class in that frame, so a lucky/wrong box on some other object
never gets mislabeled as your target class.

CAUTION: unlike hard-negative mining, this trusts the model's own bounding
box as ground truth. Skim the saved images afterward (in --dataset-images)
and delete any where the box is visibly wrong before retraining -- garbage
boxes teach the model to be confidently wrong.

Usage:
    python tools/add_hard_positives.py \
        --weights path/to/best.pt \
        --footage path/to/video_clearly_showing_the_real_object.mp4 \
        --class-name knife \
        --dataset-images path/to/Sohas.../obj_train_data/images/train \
        --dataset-labels path/to/Sohas.../obj_train_data/labels/train \
        --min-conf 0.35
"""
import argparse
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, help="The model whose own detections will become labels")
    parser.add_argument("--footage", required=True, help="Video clearly showing the real object (in your actual conditions)")
    parser.add_argument("--class-name", required=True, help="Which class to mine, e.g. 'knife' or 'pistol' (others are ignored/dropped)")
    parser.add_argument("--dataset-images", required=True, help="Path to the dataset's images/train folder")
    parser.add_argument("--dataset-labels", required=True, help="Path to the dataset's labels/train folder")
    parser.add_argument("--min-conf", type=float, default=0.35, help="Only mine detections at or above this confidence")
    parser.add_argument("--prefix", default="hardpos", help="Filename prefix for added images (avoids collisions)")
    parser.add_argument("--max-frames", type=int, default=150, help="Cap on how many frames to add")
    parser.add_argument("--frame-stride", type=int, default=3, help="Only look at every Nth frame (avoids near-duplicate frames dominating the dataset)")
    args = parser.parse_args()

    images_dir = Path(args.dataset_images)
    labels_dir = Path(args.dataset_labels)
    if not images_dir.exists() or not labels_dir.exists():
        raise SystemExit(f"Dataset folders not found: {images_dir} / {labels_dir}")

    from ultralytics import YOLO

    model = YOLO(args.weights)
    names = model.names  # {idx: name}
    target_idx = None
    for idx, name in names.items():
        if name == args.class_name:
            target_idx = idx
            break
    if target_idx is None:
        raise SystemExit(f"Class '{args.class_name}' not found in model classes: {list(names.values())}")

    cap = cv2.VideoCapture(args.footage)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.footage}")

    footage_stem = Path(args.footage).stem
    frame_i = 0
    added = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1
        if frame_i % args.frame_stride != 0:
            continue
        if added >= args.max_frames:
            break

        results = model.predict(frame, conf=args.min_conf, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            continue

        lines = []
        for i in range(len(boxes)):
            cls_idx = int(boxes.cls[i].item())
            if cls_idx != target_idx:
                continue  # drop every other class -- only trust the target class's box
            cx, cy, w, h = [float(v) for v in boxes.xywhn[i].tolist()]
            lines.append(f"{target_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if not lines:
            continue  # target class not detected in this frame -- nothing to mine

        name = f"{args.prefix}_{footage_stem}_{frame_i:06d}"
        cv2.imwrite(str(images_dir / f"{name}.jpg"), frame)
        (labels_dir / f"{name}.txt").write_text("\n".join(lines) + "\n")
        added += 1

    cap.release()
    print(f"Scanned {frame_i} frames, added {added} hard-positive '{args.class_name}' image(s) to the dataset.")
    if added == 0:
        print(f"Nothing added -- the model never detected '{args.class_name}' at conf>={args.min_conf} in this footage.")
        print("Try lowering --min-conf, or check the footage actually shows the object clearly.")
    else:
        print(f"\nIMPORTANT: skim the {added} new image(s) in {images_dir} (filenames starting with '{args.prefix}_')")
        print("and delete any where the drawn box is visibly wrong -- this script trusts the model's own box as ground truth.")
        print("\nThen retrain from the existing checkpoint (not from scratch):")
        print(f"  python tools/train_weapon_model.py --data <your data.yaml> --model {args.weights} --epochs 30")


if __name__ == "__main__":
    main()
