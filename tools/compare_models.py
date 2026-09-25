"""Compare candidate weapon models side by side and print ONE table:

  * Sohas val mAP50 for pistol and knife (ultralytics val, imgsz 640)
  * phones_test:  frames with pistol/knife >= conf, and confirmed events
  * weapons_test: % of frames with a pistol/knife >= conf, and confirmed events
  * confusers.mp4 / vtest.avi: confirmed events

"Confirmed event" = validate_weapon_model.run_footage's dashboard-equivalent
alert: min_hits-of-window smoothing (5-of-8) + the AlertManager cooldown,
at the live app's WEAPON_IMGSZ. It never switches models; it just reports.

Usage:
    python tools/compare_models.py \
        --model run10=models/weapon.pt \
        --model run15=C:/Users/dhyey/Downloads/runs/detect/runs/weapon_training/run15/weights/best.pt \
        --candidate run15 --baseline run10
"""
import argparse
import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend import config
from build_finetune_dataset import DEFAULT_SOHAS_YAML
from validate_weapon_model import run_footage


def sohas_map50(model, data_yaml):
    metrics = model.val(data=data_yaml, imgsz=640, batch=8, workers=2, plots=False, verbose=False)
    classes = list(metrics.box.ap_class_index)
    out = {}
    for idx, name in model.names.items():
        if name in config.WEAPON_THREAT_CLASSES:
            out[name] = float(metrics.box.ap50[classes.index(idx)]) if idx in classes else float("nan")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True, help="name=path/to/weights.pt (repeatable)")
    parser.add_argument("--captures-dir", default="data/captures")
    parser.add_argument("--data", default=DEFAULT_SOHAS_YAML)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--min-hits", type=int, default=5)
    parser.add_argument("--candidate", default=None, help="Model name to judge against the win criteria")
    parser.add_argument("--baseline", default="run10")
    args = parser.parse_args()

    from ultralytics import YOLO

    clips = {
        "phones_test": Path(args.captures_dir) / "phones_test.mp4",
        "weapons_test": Path(args.captures_dir) / "weapons_test.mp4",
        "confusers": config.SAMPLE_DATA_DIR / "confusers.mp4",
        "vtest": config.SAMPLE_DATA_DIR / "vtest.avi",
    }
    missing = [k for k, p in clips.items() if not p.exists()]
    for k in missing:
        print(f"NOTE: {clips[k]} not found -- '{k}' columns will show n/a")

    rows = {}
    for spec in args.model:
        name, path = spec.split("=", 1)
        if not Path(path).exists():
            print(f"NOTE: {name} weights not found at {path} -- skipped")
            continue
        print(f"Evaluating {name} ...", flush=True)
        model = YOLO(path)
        row = {}
        with contextlib.redirect_stdout(io.StringIO()):
            row["map50"] = sohas_map50(model, args.data)
            for clip, p in clips.items():
                if clip in missing:
                    continue
                frames, _, events, hit_frames = run_footage(
                    model, p, args.conf, args.window, args.min_hits, clip, imgsz=config.WEAPON_IMGSZ,
                )
                row[clip] = {"frames": frames, "hit_frames": hit_frames, "events": len(events)}
        rows[name] = row

    def fmt(row, clip, key):
        if clip not in row:
            return "n/a"
        c = row[clip]
        if key == "pct":
            return f"{100 * c['hit_frames'] / max(1, c['frames']):.1f}%"
        return str(c[key])

    header = ["model", "mAP50 pistol", "mAP50 knife", "phones_test hit frames", "phones_test events",
              "weapons_test det %", "weapons_test events", "confusers events", "vtest events"]
    table = [header]
    for name, row in rows.items():
        table.append([
            name,
            f"{row['map50'].get('pistol', float('nan')):.3f}",
            f"{row['map50'].get('knife', float('nan')):.3f}",
            fmt(row, "phones_test", "hit_frames"),
            fmt(row, "phones_test", "events"),
            fmt(row, "weapons_test", "pct"),
            fmt(row, "weapons_test", "events"),
            fmt(row, "confusers", "events"),
            fmt(row, "vtest", "events"),
        ])
    widths = [max(len(r[i]) for r in table) for i in range(len(header))]
    print()
    for i, r in enumerate(table):
        print(" | ".join(c.ljust(w) for c, w in zip(r, widths)))
        if i == 0:
            print("-+-".join("-" * w for w in widths))

    cand, base = rows.get(args.candidate), rows.get(args.baseline)
    if cand and base:
        print(f"\nWin criteria for {args.candidate} vs {args.baseline}:")
        checks = []
        if "phones_test" in cand:
            checks.append(("phones_test confirmed events = 0", cand["phones_test"]["events"] == 0))
        else:
            checks.append(("phones_test confirmed events = 0 (NO phones_test clip)", False))
        if "weapons_test" in cand and "weapons_test" in base:
            pct = lambda r: 100 * r["weapons_test"]["hit_frames"] / max(1, r["weapons_test"]["frames"])
            checks.append((f"weapons_test detection within 5 pts ({pct(cand):.1f}% vs {pct(base):.1f}%)", pct(cand) >= pct(base) - 5))
        else:
            checks.append(("weapons_test detection within 5 pts (NO weapons_test clip)", False))
        for cls in ("pistol", "knife"):
            c, b = cand["map50"][cls], base["map50"][cls]
            checks.append((f"{cls} mAP50 down <= 2 pts ({c:.3f} vs {b:.3f})", c >= b - 0.02))
        for text, ok in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {text}")
        print(f"  => {args.candidate} {'WINS' if all(ok for _, ok in checks) else 'does NOT win'} (no model was switched)")


if __name__ == "__main__":
    main()
