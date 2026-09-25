"""Fine-tune a YOLOv8 weapon detector on a dataset you provide.

This is a thin wrapper around ultralytics' own training loop -- it does not
invent training data or shortcut the actual work. You need a labeled
dataset first (see README.md's "Weapon detection" section for how to get
one from Roboflow Universe), exported in YOLOv8 format (a `data.yaml` plus
`train/`, `valid/`, `test/` image+label folders).

Usage:
    python tools/train_weapon_model.py --data path/to/data.yaml

Run `python tools/train_weapon_model.py --help` for all options. After
training, ALWAYS run tools/validate_weapon_model.py on footage that has no
weapons in it before trusting the result -- see that script's docstring.
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to the dataset's data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model to fine-tune (default: yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="e.g. '0' for first GPU, 'cpu'. Auto-detected if omitted.")
    parser.add_argument("--project", default="runs/weapon_training")
    parser.add_argument("--name", default="run1")
    parser.add_argument(
        "--patience", type=int, default=20,
        help="Stop early if validation mAP hasn't improved for this many epochs",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Parallel dataloader worker processes. Lower this (e.g. 2) if training "
        "crashes with a system-RAM 'Insufficient memory' error rather than a CUDA error.",
    )
    parser.add_argument(
        "--lr0", type=float, default=None,
        help="Initial learning rate. Only honored with an explicit --optimizer: "
        "ultralytics' optimizer=auto ignores lr0 and picks its own (SGD 0.01 on big datasets).",
    )
    parser.add_argument("--optimizer", default="auto", help="auto, SGD, AdamW, ... (set explicitly for --lr0 to take effect)")
    args = parser.parse_args()
    if args.lr0 is not None and args.optimizer == "auto":
        raise SystemExit("--lr0 is ignored by optimizer=auto; pass --optimizer SGD (or AdamW) too")

    if not Path(args.data).exists():
        raise SystemExit(f"Dataset config not found: {args.data}")

    device = args.device
    if device is None:
        try:
            import torch

            device = "0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    print(f"Training on device: {device}")

    from ultralytics import YOLO

    model = YOLO(args.model)
    extra = {"lr0": args.lr0} if args.lr0 is not None else {}
    results = model.train(
        optimizer=args.optimizer,
        **extra,
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
    )

    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    print("\n" + "=" * 70)
    print(f"Training complete. Best weights: {best_weights}")
    print("Do NOT copy this straight to models/weapon.pt yet.")
    print("Validate it first:")
    print(
        f"  python tools/validate_weapon_model.py --weights {best_weights} "
        f"--footage path/to/weapon_free_video.mp4"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
