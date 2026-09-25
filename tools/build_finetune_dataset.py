"""Build the data.yaml for a fine-tune run: the full original Sohas train
set + the reviewed capture frames (auto + accepted only), with the new
frames oversampled by listing each path N times (ultralytics keeps
duplicate entries of a list-file dataset). Validation stays the untouched
Sohas test split so mAP is comparable with run10/run13.

Nothing is copied into the Sohas folders -- the new frames stay under
data/finetune/<session>/ and are referenced from a train list file.

Also prints the per-class box counts (original / new / effective after
oversampling) so the mix can be checked BEFORE training.

Usage:
    python tools/build_finetune_dataset.py --sessions phones_train weapons_train --name run15
"""
import argparse
import json
from pathlib import Path

import yaml

from label_capture import SOHAS_NAMES

DEFAULT_SOHAS_YAML = (
    r"C:\Users\dhyey\Downloads\od-weapondetection\Weapons and similar handled objects"
    r"\Sohas_weapon-Detection-YOLOv5\dataset_fixed.yaml"
)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def count_labels(image_paths, label_path_for):
    per_class = {n: 0 for n in SOHAS_NAMES}
    background = 0
    for img in image_paths:
        lp = label_path_for(img)
        lines = [ln for ln in lp.read_text().split("\n") if ln.strip()] if lp.exists() else []
        if not lines:
            background += 1
        for ln in lines:
            per_class[SOHAS_NAMES[int(ln.split()[0])]] += 1
    return per_class, background


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sessions", nargs="+", required=True)
    parser.add_argument("--name", required=True, help="Run name, e.g. run15 (names the yaml/list files)")
    parser.add_argument("--sohas-yaml", default=DEFAULT_SOHAS_YAML)
    parser.add_argument("--out-root", default="data/finetune")
    parser.add_argument("--oversample", type=int, default=3)
    args = parser.parse_args()

    sohas = yaml.safe_load(Path(args.sohas_yaml).read_text(encoding="utf-8-sig"))
    sohas_root = Path(sohas["path"])
    train_dir = sohas_root / sohas["train"]
    val_dir = sohas_root / sohas["val"]
    orig_images = sorted(p for p in train_dir.iterdir() if p.suffix.lower() in IMG_EXTS)

    def sohas_label(img):
        # same rule ultralytics' img2label_paths uses: last /images/ -> /labels/
        a, b = f"{Path('/images/')}", f"{Path('/labels/')}"
        return Path(b.join(str(img).rsplit(a, 1))).with_suffix(".txt")

    new_images = []
    status_totals = {}
    for session in args.sessions:
        session_dir = Path(args.out_root) / session
        manifest = json.loads((session_dir / "manifest.json").read_text())
        for name, m in manifest.items():
            status_totals[m["status"]] = status_totals.get(m["status"], 0) + 1
            if m["status"] in ("auto", "accepted"):
                new_images.append((session_dir / "images" / f"{name}.jpg").resolve())
    pending = status_totals.get("flagged", 0)

    def new_label(img):
        return img.parent.parent / "labels" / f"{img.stem}.txt"

    orig_counts, orig_bg = count_labels(orig_images, sohas_label)
    new_counts, new_bg = count_labels(new_images, new_label)

    out_root = Path(args.out_root).resolve()
    list_path = out_root / f"{args.name}_train.txt"
    lines = [str(p) for p in orig_images] + [str(p) for p in new_images] * args.oversample
    list_path.write_text("\n".join(lines) + "\n")
    yaml_path = out_root / f"{args.name}_data.yaml"
    yaml_path.write_text(yaml.safe_dump({
        "train": str(list_path),
        "val": str(val_dir),
        "nc": len(SOHAS_NAMES),
        "names": SOHAS_NAMES,
    }, sort_keys=False))

    print(f"Capture frames by status: {status_totals}")
    if pending:
        print(f"WARNING: {pending} frame(s) still 'flagged' (unreviewed) -- they are NOT included.")
    print(f"\n{'class':<12}{'original':>10}{'new':>8}{'new x' + str(args.oversample):>10}{'total':>9}")
    for n in SOHAS_NAMES:
        eff = new_counts[n] * args.oversample
        print(f"{n:<12}{orig_counts[n]:>10}{new_counts[n]:>8}{eff:>10}{orig_counts[n] + eff:>9}")
    eff_bg = new_bg * args.oversample
    print(f"{'(background)':<12}{orig_bg:>10}{new_bg:>8}{eff_bg:>10}{orig_bg + eff_bg:>9}   <- images with empty labels")
    print(f"\nimages: {len(orig_images)} original + {len(new_images)} new x{args.oversample} = {len(lines)} train entries")
    print(f"val: {val_dir}")
    print(f"wrote {yaml_path}")


if __name__ == "__main__":
    main()
