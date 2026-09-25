"""Step through the frames label_capture.py FLAGGED and decide each one.

Shows the frame with its candidate boxes (class + index) and why it was
flagged. Keys:
    a  = accept   (keep the frame with exactly the boxes shown)
    d  = delete   (drop the frame from training entirely)
    s  = skip     (decide later; skipped frames stay out of training)
    b  = draw a box with the mouse (drag, then ENTER/SPACE; c cancels).
         The first box drawn replaces the candidates, later ones add to
         them. On weapon frames you then press 1 = pistol, 2 = knife.
         Then press a to accept.
    x  = clear all boxes (e.g. COCO called a pen a phone on a background
         frame -> x, then a to accept it as background)
    q  = quit     (progress is saved after every key)

A phone/weapon frame with NO box can't be accepted -- that would teach the
model "nothing here" on a frame that actually shows the object. Draw a
box (b) or delete it.

Usage:
    python tools/review_flagged.py --session phones_train
    python tools/review_flagged.py --session weapons_train --include-skipped
"""
import argparse
import json
from pathlib import Path

import cv2

from label_capture import SOHAS_NAMES, yolo_line

COLORS = {"pistol": (0, 0, 255), "knife": (0, 0, 255), "smartphone": (0, 200, 0)}


def draw(frame, label_lines, header_lines):
    h, w = frame.shape[:2]
    for line in label_lines:
        cls_idx, cx, cy, bw, bh = line.split()
        name = SOHAS_NAMES[int(cls_idx)]
        cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        x1, y1, x2, y2 = int(cx - bw / 2), int(cy - bh / 2), int(cx + bw / 2), int(cy + bh / 2)
        color = COLORS.get(name, (255, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, name, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    y = 30
    for text in header_lines:
        cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5)
        cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y += 30
    return frame


def draw_box(window, frame, tag):
    """Let the reviewer drag a box; returns a YOLO line or None if cancelled."""
    x, y, bw, bh = cv2.selectROI(window, frame, showCrosshair=False)
    if bw == 0 or bh == 0:
        return None
    if tag == "weapon":
        cls_name = None
        while cls_name is None:
            key = cv2.waitKey(0) & 0xFF
            cls_name = {ord("1"): "pistol", ord("2"): "knife"}.get(key)
    else:
        cls_name = "smartphone"
    h, w = frame.shape[:2]
    return yolo_line(SOHAS_NAMES.index(cls_name), [(x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True)
    parser.add_argument("--out-root", default="data/finetune")
    parser.add_argument("--include-skipped", action="store_true", help="Also revisit frames you skipped earlier")
    args = parser.parse_args()

    session_dir = Path(args.out_root) / args.session
    manifest_path = session_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    wanted = {"flagged", "skipped"} if args.include_skipped else {"flagged"}
    queue = [name for name, m in manifest.items() if m["status"] in wanted]
    if not queue:
        print(f"Nothing to review in {args.session}.")
        return

    window = f"review: {args.session}"
    counts = {"accepted": 0, "deleted": 0, "skipped": 0}
    for i, name in enumerate(queue):
        m = manifest[name]
        frame = cv2.imread(str(session_dir / "images" / f"{name}.jpg"))
        label_path = session_dir / "labels" / f"{name}.txt"
        lines = [ln for ln in label_path.read_text().split("\n") if ln.strip()]
        drawn = False

        while True:
            can_accept = bool(lines) or m["tag"] == "other"
            box_note = f"{len(lines)} box(es)" if lines else ("NO BOX -> accept = background" if can_accept else "NO BOX -> draw (b), d or s")
            header = [
                f"[{i + 1}/{len(queue)}] {name}  tag={m['tag']}  {box_note}",
                "boxes edited by you" if drawn else m["reason"],
                "a=accept  b=draw box" + ("  (then 1=pistol 2=knife)" if m["tag"] == "weapon" else "") + "  x=clear  d=delete  s=skip  q=quit",
            ]
            cv2.imshow(window, draw(frame.copy(), lines, header))
            key = cv2.waitKey(0) & 0xFF
            if key == ord("b"):
                line = draw_box(window, frame, m["tag"])
                if line:
                    lines = (lines if drawn else []) + [line]
                    drawn = True
                continue
            if key == ord("x"):
                lines, drawn = [], True
                continue
            if key == ord("q"):
                manifest_path.write_text(json.dumps(manifest, indent=1))
                cv2.destroyAllWindows()
                print(f"Stopped. {counts}; {len(queue) - i} left (run again to continue).")
                return
            if key == ord("a") and can_accept:
                m["status"] = "accepted"
                label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
            elif key == ord("d"):
                m["status"] = "deleted"
            elif key == ord("s"):
                m["status"] = "skipped"
            else:
                continue
            counts[m["status"]] += 1
            break
        manifest_path.write_text(json.dumps(manifest, indent=1))

    cv2.destroyAllWindows()
    print(f"Done reviewing {args.session}: {counts}")


if __name__ == "__main__":
    main()
