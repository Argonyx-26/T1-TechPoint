"""Apply a manual review pass done via preview_labels.py.

Workflow: after preview_labels.py saves annotated copies into a preview
folder, delete the bad ones from that preview folder (visually, in File
Explorer) and keep the good ones. This script then treats whatever's LEFT
in the preview folder as the "approved" list, and deletes any dataset
image+label pair (matching --prefix) that's no longer present there.

Usage:
    python tools/apply_label_review.py \
        --images-dir path/to/obj_train_data/images/train \
        --labels-dir path/to/obj_train_data/labels/train \
        --preview-dir path/to/hardpos_preview \
        --prefix hardpos
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--preview-dir", required=True, help="Preview folder after you've deleted the bad ones")
    parser.add_argument("--prefix", required=True, help="Only consider dataset images starting with this prefix")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    preview_dir = Path(args.preview_dir)

    approved_names = {p.name for p in preview_dir.iterdir() if p.is_file()}

    candidates = sorted(p for p in images_dir.glob(f"{args.prefix}*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
    if not candidates:
        raise SystemExit(f"No images found matching prefix '{args.prefix}' in {images_dir}")

    kept = 0
    removed = 0
    for image_path in candidates:
        if image_path.name in approved_names:
            kept += 1
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        image_path.unlink()
        if label_path.exists():
            label_path.unlink()
        removed += 1
        print(f"Removed: {image_path.name}")

    print(f"\nKept {kept} approved image(s), removed {removed} rejected image+label pair(s).")


if __name__ == "__main__":
    main()
