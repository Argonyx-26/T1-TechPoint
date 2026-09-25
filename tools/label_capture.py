"""Auto-label a capture session (from capture_session.py) into YOLO training
frames, flagging anything uncertain for review_flagged.py instead of
guessing.

Labeling policy, per frame tag from the session's .json sidecar:
  phone  -> COCO yolov8n "cell phone" boxes (conf >= --phone-conf) become
            class 1 (smartphone). No confident phone box -> FLAGGED, with
            review candidates attached: low-confidence yolov8n phones, a
            bigger COCO model's phones (--candidate-coco), and the weapon
            model's own boxes relabeled smartphone -- on a phone frame, a
            run10 "pistol" box IS the false alarm we're correcting.
  other  -> empty label (background: bottles/pens/wallets/...). If COCO sees
            a cell phone anyway -> FLAGGED (a phone may be in view).
  weapon -> the weapon model's pistol/knife boxes at conf >= --weapon-conf.
            Otherwise FLAGGED, with its lower-confidence pistol/knife
            candidates attached.
  skip   -> frame not extracted at all (e.g. the part of a clip where a
            weapon may be present but unlabeled).

Output layout (ultralytics finds labels by swapping images/ -> labels/):
    <out-root>/<session>/images/<session>_<frame>.jpg
    <out-root>/<session>/labels/<session>_<frame>.txt
    <out-root>/<session>/manifest.json   {name: {"status", "tag", "reason"}}
status is "auto" (trusted), "flagged" (needs review), later "accepted" /
"deleted" / "skipped" via review_flagged.py. Only auto + accepted frames
go into training (see build_finetune_dataset.py).

Usage:
    python tools/label_capture.py --session phones_train
    python tools/label_capture.py --session weapons_train --weapon-weights models/weapon.pt
"""
import argparse
import json
from pathlib import Path

import cv2

# Sohas dataset_fixed.yaml class order
SOHAS_NAMES = ["pistol", "smartphone", "knife", "monedero", "billete", "tarjeta"]
SMARTPHONE_IDX = SOHAS_NAMES.index("smartphone")
WEAPON_CLASSES = {"pistol", "knife"}
CANDIDATE_CONF = 0.10  # floor for boxes shown to the reviewer on flagged frames


def tag_for_frame(segments, frame_i):
    tag = segments[0]["tag"]
    for seg in segments:
        if seg["start_frame"] <= frame_i:
            tag = seg["tag"]
    return tag


def yolo_line(cls_idx, xywhn):
    cx, cy, w, h = [float(v) for v in xywhn]
    return f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def phone_boxes(coco_model, frame, min_conf):
    """(confident_lines, candidate_lines) for COCO 'cell phone' boxes."""
    res = coco_model.predict(frame, conf=CANDIDATE_CONF, verbose=False)[0]
    confident, candidates = [], []
    for i in range(len(res.boxes)):
        if res.names[int(res.boxes.cls[i])] != "cell phone":
            continue
        line = yolo_line(SMARTPHONE_IDX, res.boxes.xywhn[i].tolist())
        (confident if float(res.boxes.conf[i]) >= min_conf else candidates).append(line)
    return confident, candidates


def iou(a, b):
    """IoU of two YOLO lines (cls cx cy w h, normalized)."""
    ax, ay, aw, ah = [float(v) for v in a.split()[1:]]
    bx, by, bw, bh = [float(v) for v in b.split()[1:]]
    ix = max(0.0, min(ax + aw / 2, bx + bw / 2) - max(ax - aw / 2, bx - bw / 2))
    iy = max(0.0, min(ay + ah / 2, by + bh / 2) - max(ay - ah / 2, by - bh / 2))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def merge_candidates(*groups):
    """Concatenate candidate lists, dropping boxes that overlap one already kept."""
    kept = []
    for group in groups:
        for line in group:
            if all(iou(line, k) < 0.5 for k in kept):
                kept.append(line)
    return kept


def any_boxes_as(model, frame, cls_idx, min_conf=0.25):
    """Every box the model draws, relabeled as cls_idx (review candidates only)."""
    res = model.predict(frame, conf=min_conf, verbose=False)[0]
    return [yolo_line(cls_idx, res.boxes.xywhn[i].tolist()) for i in range(len(res.boxes))]


def weapon_boxes(weapon_model, frame, min_conf):
    """(confident_lines, candidate_lines) for pistol/knife, class ids remapped
    by name onto the Sohas order so a model with a different class order
    can't silently mislabel."""
    res = weapon_model.predict(frame, conf=CANDIDATE_CONF, verbose=False)[0]
    confident, candidates = [], []
    for i in range(len(res.boxes)):
        name = res.names[int(res.boxes.cls[i])]
        if name not in WEAPON_CLASSES:
            continue
        line = yolo_line(SOHAS_NAMES.index(name), res.boxes.xywhn[i].tolist())
        (confident if float(res.boxes.conf[i]) >= min_conf else candidates).append(line)
    return confident, candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True)
    parser.add_argument("--captures-dir", default="data/captures")
    parser.add_argument("--out-root", default="data/finetune")
    parser.add_argument("--frame-stride", type=int, default=10, help="Keep every Nth frame (~3/s at 30fps) -- neighbours are near-duplicates")
    parser.add_argument("--coco-weights", default="yolov8n.pt")
    parser.add_argument("--candidate-coco", default="yolov8x.pt", help="Bigger COCO model used ONLY for review candidates on flagged phone frames")
    parser.add_argument("--weapon-weights", default="models/weapon.pt")
    parser.add_argument("--phone-conf", type=float, default=0.35, help="COCO cell-phone confidence to trust without review")
    parser.add_argument("--weapon-conf", type=float, default=0.5, help="Weapon-model confidence to trust without review")
    args = parser.parse_args()

    video_path = Path(args.captures_dir) / f"{args.session}.mp4"
    meta_path = Path(args.captures_dir) / f"{args.session}.json"
    if not video_path.exists() or not meta_path.exists():
        raise SystemExit(f"Missing {video_path} or {meta_path} -- record it with capture_session.py first")
    segments = json.loads(meta_path.read_text())["segments"]

    session_dir = Path(args.out_root) / args.session
    images_dir = session_dir / "images"
    labels_dir = session_dir / "labels"
    manifest_path = session_dir / "manifest.json"
    if manifest_path.exists():
        raise SystemExit(f"{manifest_path} exists -- this session is already labeled (delete {session_dir} to redo it)")
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    tags_used = {tag_for_frame(segments, 0)} | {s["tag"] for s in segments}
    coco = YOLO(args.coco_weights) if tags_used & {"phone", "other"} else None
    coco_big = YOLO(args.candidate_coco) if "phone" in tags_used else None
    weapon = YOLO(args.weapon_weights)

    cap = cv2.VideoCapture(str(video_path))
    manifest = {}
    frame_i = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1
        if frame_i % args.frame_stride:
            continue
        tag = tag_for_frame(segments, frame_i)
        if tag == "skip":
            continue
        if tag == "phone":
            lines, cands = phone_boxes(coco, frame, args.phone_conf)
            status, reason = ("auto", "") if lines else ("flagged", "phone frame: COCO found no confident cell phone")
            if not lines:
                big, _ = phone_boxes(coco_big, frame, 0.25)
                lines = merge_candidates(cands, big, any_boxes_as(weapon, frame, SMARTPHONE_IDX))
        elif tag == "other":
            confident, cands = phone_boxes(coco, frame, args.phone_conf)
            if confident:
                status, reason, lines = "flagged", "background frame but COCO sees a cell phone", confident
            else:
                status, reason, lines = "auto", "", []
        else:
            lines, cands = weapon_boxes(weapon, frame, args.weapon_conf)
            status, reason = ("auto", "") if lines else ("flagged", f"weapon frame: no pistol/knife >= {args.weapon_conf}")
            if not lines:
                lines = cands

        name = f"{args.session}_{frame_i:06d}"
        cv2.imwrite(str(images_dir / f"{name}.jpg"), frame)
        (labels_dir / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        manifest[name] = {"status": status, "tag": tag, "reason": reason}
    cap.release()

    manifest_path.write_text(json.dumps(manifest, indent=1))
    print_summary(args.session, manifest, labels_dir)


def print_summary(session, manifest, labels_dir):
    by_status, by_tag, boxes = {}, {}, {}
    for name, m in manifest.items():
        by_status[m["status"]] = by_status.get(m["status"], 0) + 1
        by_tag[m["tag"]] = by_tag.get(m["tag"], 0) + 1
        if m["status"] in ("auto", "accepted"):
            for line in (labels_dir / f"{name}.txt").read_text().split("\n"):
                if line.strip():
                    cls = SOHAS_NAMES[int(line.split()[0])]
                    boxes[cls] = boxes.get(cls, 0) + 1
    print(f"[{session}] frames: {len(manifest)}  by tag: {by_tag}  by status: {by_status}")
    print(f"[{session}] boxes in trusted frames: {boxes or '{}'}")


if __name__ == "__main__":
    main()
