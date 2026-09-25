"""Merge an external YOLO-format dataset (e.g. from Roboflow Universe) into
your existing weapon-detection training set, remapping class IDs by NAME
rather than assuming index order matches.

Why this exists: two YOLO datasets almost never use the same class-to-index
numbering. If you just copy label files from one dataset into another
without remapping, a "pistol" that's index 1 in the source dataset could
silently become "smartphone" (index 1) in your dataset -- wrong, and hard to
notice until the model behaves strangely. This script reads both datasets'
data.yaml, matches classes by name, remaps every label file's class index
accordingly, and SKIPS any class present in the source but not in your
target (e.g. a "helmet" class you don't care about) -- boxes for skipped
classes are simply dropped from that image's label file, not the whole
image.

Usage:
    python tools/merge_external_dataset.py \
        --source-data path/to/downloaded_dataset/data.yaml \
        --source-images path/to/downloaded_dataset/train/images \
        --source-labels path/to/downloaded_dataset/train/labels \
        --target-images path/to/Sohas.../obj_train_data/images/train \
        --target-labels path/to/Sohas.../obj_train_data/labels/train \
        --target-names pistol smartphone knife monedero billete tarjeta \
        --prefix cctv1

After running this, retrain from your latest checkpoint the same way you
would after hard-negative mining:
    python tools/train_weapon_model.py --data <your data.yaml> \
        --model <latest best.pt> --epochs 30
"""
import argparse
import shutil
from pathlib import Path

import yaml


def load_class_names(data_yaml_path: Path):
    with open(data_yaml_path, "r") as f:
        data = yaml.safe_load(f)
    names = data["names"]
    if isinstance(names, dict):
        # {0: 'pistol', 1: 'knife', ...} -> ordered list
        return [names[i] for i in sorted(names)]
    return list(names)  # already a list, index = position


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-data", required=True, help="data.yaml of the downloaded (source) dataset")
    parser.add_argument("--source-images", required=True, help="Source dataset's images folder (e.g. train/images)")
    parser.add_argument("--source-labels", required=True, help="Source dataset's labels folder (e.g. train/labels)")
    parser.add_argument("--target-images", required=True, help="Your existing dataset's images/train folder")
    parser.add_argument("--target-labels", required=True, help="Your existing dataset's labels/train folder")
    parser.add_argument(
        "--target-names", required=True, nargs="+",
        help="Your dataset's class names IN INDEX ORDER, e.g. --target-names pistol smartphone knife monedero billete tarjeta",
    )
    parser.add_argument("--prefix", default="ext", help="Filename prefix for merged images (avoids collisions)")
    args = parser.parse_args()

    source_images = Path(args.source_images)
    source_labels = Path(args.source_labels)
    target_images = Path(args.target_images)
    target_labels = Path(args.target_labels)
    for p in (source_images, source_labels, target_images, target_labels):
        if not p.exists():
            raise SystemExit(f"Path not found: {p}")

    source_names = load_class_names(Path(args.source_data))
    target_names = args.target_names

    print(f"Source classes (by index): {list(enumerate(source_names))}")
    print(f"Target classes (by index): {list(enumerate(target_names))}")

    # Build source_idx -> target_idx map; classes not in target are dropped.
    index_map = {}
    dropped_classes = []
    for src_idx, name in enumerate(source_names):
        if name in target_names:
            index_map[src_idx] = target_names.index(name)
        else:
            dropped_classes.append(name)
    if dropped_classes:
        print(f"Classes present in source but not in target (boxes will be dropped): {dropped_classes}")
    if not index_map:
        raise SystemExit(
            "No matching class names between source and target -- check spelling "
            "in --target-names against the source dataset's data.yaml names."
        )

    label_files = sorted(source_labels.glob("*.txt"))
    if not label_files:
        raise SystemExit(f"No label files found in {source_labels}")

    copied = 0
    skipped_empty_after_remap = 0
    total_boxes_kept = 0
    total_boxes_dropped = 0

    for label_path in label_files:
        stem = label_path.stem
        image_path = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            candidate = source_images / f"{stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            continue  # label with no matching image, skip

        new_lines = []
        for line in label_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            src_cls = int(parts[0])
            if src_cls not in index_map:
                total_boxes_dropped += 1
                continue
            parts[0] = str(index_map[src_cls])
            new_lines.append(" ".join(parts))
            total_boxes_kept += 1

        if not new_lines:
            # Every box in this image was a dropped class (e.g. only a
            # helmet). Skip the image entirely rather than adding it as a
            # confusing "empty label" -- it isn't a validated background
            # frame, just an image we have no use for.
            skipped_empty_after_remap += 1
            continue

        new_name = f"{args.prefix}_{stem}"
        shutil.copy(image_path, target_images / f"{new_name}{image_path.suffix}")
        (target_labels / f"{new_name}.txt").write_text("\n".join(new_lines) + "\n")
        copied += 1

    print(f"\nMerged {copied} image(s) into the dataset.")
    print(f"  Boxes kept (remapped): {total_boxes_kept}")
    print(f"  Boxes dropped (unmatched classes): {total_boxes_dropped}")
    print(f"  Images skipped (no boxes left after dropping unmatched classes): {skipped_empty_after_remap}")
    if copied:
        print("\nNow retrain from your latest checkpoint (not from scratch):")
        print("  python tools/train_weapon_model.py --data <your data.yaml> --model <latest best.pt> --epochs 30")


if __name__ == "__main__":
    main()
