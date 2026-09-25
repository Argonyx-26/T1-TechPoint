"""Hard-negative mining: turn a model's false positives into training data.

When validate_weapon_model.py reports false alerts on footage that has no
weapons, the fix isn't to just retrain longer -- it's to show the model the
exact frames it got wrong, labeled as "nothing here," so it learns to stop
firing on them. This script runs the model over footage known to have no
weapons, saves every frame where it (wrongly) detected something as a new
background training image with an empty label file, and reports how many
it added.

After running this, retrain FROM THE EXISTING CHECKPOINT (not from scratch)
so it keeps what it already learned and just corrects the false positives:

    python tools/train_weapon_model.py --data path/to/dataset_fixed.yaml \
        --model path/to/best.pt --epochs 30

Usage:
    python tools/add_hard_negatives.py \
        --weights path/to/best.pt \
        --footage path/to/weapon_free_video.mp4 \
        --dataset-images path/to/Sohas.../obj_train_data/images/train \
        --dataset-labels path/to/Sohas.../obj_train_data/labels/train
"""
import argparse
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, help="The model that produced false positives")
    parser.add_argument("--footage", required=True, help="Video with NO weapons in it (the one that triggered false positives)")
    parser.add_argument("--dataset-images", required=True, help="Path to the dataset's images/train folder")
    parser.add_argument("--dataset-labels", required=True, help="Path to the dataset's labels/train folder")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--prefix", default="hardneg", help="Filename prefix for added images (avoids collisions)")
    parser.add_argument("--max-frames", type=int, default=300, help="Cap on how many frames to add (avoid one video dominating the dataset)")
    args = parser.parse_args()

    images_dir = Path(args.dataset_images)
    labels_dir = Path(args.dataset_labels)
    if not images_dir.exists() or not labels_dir.exists():
        raise SystemExit(f"Dataset folders not found: {images_dir} / {labels_dir}")

    from ultralytics import YOLO

    model = YOLO(args.weights)
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
        results = model.predict(frame, conf=args.conf, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            continue  # nothing detected here -- not useful as a hard negative

        if added >= args.max_frames:
            continue

        name = f"{args.prefix}_{footage_stem}_{frame_i:06d}"
        cv2.imwrite(str(images_dir / f"{name}.jpg"), frame)
        (labels_dir / f"{name}.txt").write_text("")  # empty label = "no objects here"
        added += 1

    cap.release()
    print(f"Scanned {frame_i} frames, added {added} hard-negative image(s) to the dataset.")
    if added == 0:
        print("Nothing added -- the model didn't false-positive on this footage at this confidence.")
    elif added >= args.max_frames:
        print(f"Hit --max-frames cap ({args.max_frames}); there may be more false positives in this footage.")
    else:
        print("Now retrain from the existing checkpoint (not from scratch) to correct these:")
        print(f"  python tools/train_weapon_model.py --data <your data.yaml> --model {args.weights} --epochs 30")


if __name__ == "__main__":
    main()
