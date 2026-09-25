"""Fight classifier trained on real surveillance footage.

The pose rule for fighting cannot see real CCTV fights: people overlap and
occlude each other, so the pose model rarely gets two clean skeletons
(Surveillance Camera Fight Dataset, 300 clips: a fight-like pose frame in
28/150 fights vs 13/150 non-fights). This learns fighting from the footage
itself instead:

  features  CLIP (ViT-B-32, already loaded for Ask Vigil) embeddings of the
            people region, sampled at SAMPLE_FPS over a WINDOW_S window;
            window vector = [mean embedding, mean |frame-to-frame change|]
  model     L2-regularised logistic regression (numpy at runtime)
  data      Surveillance Camera Fight Dataset (Akti et al., IPTA 2019, MIT),
            150 fight + 150 no-fight clips from real CCTV
  check     5-fold cross-validation grouped by source video (clips cut from
            the same YouTube video never sit on both sides of a split)

Usage:
    python tools/train_fight_classifier.py extract --root ../behaviour_datasets/survfight
    python tools/train_fight_classifier.py mine --root ../behaviour_datasets   # hard negatives (normal crowds)
    python tools/train_fight_classifier.py cv
    python tools/train_fight_classifier.py train          # -> models/fight_clf.npz
    python tools/train_fight_classifier.py scan ../behaviour_datasets/caviar/Fight_Chase.mpg
"""
import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from backend import config  # noqa: E402

FEATURES = REPO / "data" / "fight_features.npz"
HARD_NEG = REPO / "data" / "fight_hard_negatives.npz"
# Normal crowded CCTV the live detector scores (people close together / in
# crowds) and must learn is NOT fighting. The other half of each source is
# kept out as the held-out test (tools/behaviour_benchmark.json).
HARD_NEG_CLIPS = [
    "umn/Crowd-Activity-All.avi@0-130",
    "caviar/Browse1.mpg", "caviar/Browse3.mpg", "caviar/Browse_WhileWaiting1.mpg", "caviar/Meet_Crowd.mpg",
    "caviar/Meet_WalkTogether1.mpg", "caviar/Walk1.mpg", "caviar/Walk3.mpg", "caviar/LeftBag.mpg",
]
MODEL = REPO / "models" / "fight_clf.npz"


def groups_from_videos_txt(path: Path) -> dict:
    """clip stem (fi001 / nofi001) -> source video index."""
    group, out = -1, {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("http"):
            group += 1
            continue
        m = re.match(r"^(n?o?fi\d+)\s*:", line)
        if m:
            out[m.group(1)] = group
    return out


def extract(args):
    from backend.cameras import SharedModels
    from backend.fight import clip_embeddings
    from backend.search import ClipEmbedder

    root = Path(args.root)
    groups = groups_from_videos_txt(root / "videos.txt")
    shared, emb = SharedModels(), ClipEmbedder()
    names, labels, grp, seqs = [], [], [], []
    for label, folder in ((1, "fight"), (0, "noFight")):
        for p in sorted((root / folder).glob("*.mp4")):
            seq = clip_embeddings(shared, emb, p)
            if len(seq) < 2:
                continue
            names.append(p.stem); labels.append(label); grp.append(groups.get(p.stem, -1 - len(grp)))
            seqs.append(seq)
            print(f"{p.stem}: {len(seq)} frames", flush=True)
    FEATURES.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FEATURES, names=np.array(names), labels=np.array(labels), groups=np.array(grp),
                        seqs=np.array(seqs, dtype=object), allow_pickle=True)
    print(f"saved {len(names)} clips -> {FEATURES}")


def load():
    d = np.load(FEATURES, allow_pickle=True)
    from backend.fight import window_vector
    x = np.stack([window_vector(s) for s in d["seqs"]])
    return x, d["labels"].astype(np.float32), d["groups"], d["names"]


def load_hard():
    if not HARD_NEG.exists():
        return np.zeros((0, 1024), np.float32)
    return np.load(HARD_NEG)["x"]


def mine(args):
    """Window vectors exactly as the live FightMonitor scores them, on normal
    footage (all negatives)."""
    sys.path.insert(0, str(REPO / "tools"))
    import eval_behaviour as eb
    from backend import fight
    from backend.cameras import SharedModels

    vecs = []
    orig = fight.FightClassifier.prob

    def spy(self, v, _o=orig):
        if eb.CLOCK0 + t_range[0] <= _now[0] <= eb.CLOCK0 + t_range[1]:
            vecs.append(np.asarray(v, np.float32))
        return _o(self, v)
    fight.FightClassifier.prob = spy
    orig_update = fight.FightMonitor.update

    def track_time(self, cam_id, ts, frame, boxes, _o=orig_update):
        _now[0] = ts
        return _o(self, cam_id, ts, frame, boxes)
    fight.FightMonitor.update = track_time
    shared = SharedModels()
    for spec in args.clips or HARD_NEG_CLIPS:
        clip, _, rng = spec.partition("@")
        t_range = [float(v) for v in rng.split("-")] if rng else [0, 1e9]
        _now = [0.0]
        before = len(vecs)
        eb.run_clip(shared, Path(args.root) / clip)
        print(f"{spec}: {len(vecs) - before} windows", flush=True)
    np.savez_compressed(HARD_NEG, x=np.stack(vecs) if vecs else np.zeros((0, 1024), np.float32))
    print(f"saved {len(vecs)} hard negatives -> {HARD_NEG}")


def fit(x, y, l2=1e-2, epochs=400):
    import torch

    mu, sd = x.mean(0), x.std(0) + 1e-6
    xt = torch.tensor((x - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(y)
    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=epochs, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(xt @ w + b, yt) + l2 * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return {"w": w.detach().numpy(), "b": float(b.detach()), "mu": mu, "sd": sd}


def predict(m, x):
    return 1.0 / (1.0 + np.exp(-(((x - m["mu"]) / m["sd"]) @ m["w"] + m["b"])))


def auc(y, p):
    order = np.argsort(p)
    ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    return (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


def cv(args):
    x, y, g, _ = load()
    ug = np.unique(g)
    rng = np.random.default_rng(0)
    rng.shuffle(ug)
    folds = np.array_split(ug, 5)
    for l2 in args.l2:
        p = np.zeros(len(y))
        hard = load_hard()
        for f in folds:
            test = np.isin(g, f)
            m = fit(np.concatenate([x[~test], hard]), np.concatenate([y[~test], np.zeros(len(hard), np.float32)]), l2)
            p[test] = predict(m, x[test])
        print(f"l2={l2:g}  AUC {auc(y, p):.3f}  " + "  ".join(
            f"@{t:.2f}: recall {(p[y == 1] >= t).mean():.2f} false-pos {(p[y == 0] >= t).mean():.2f}"
            for t in (0.5, 0.7, 0.8, 0.9)))


def train(args):
    x, y, _, _ = load()
    hard = load_hard()
    x, y = np.concatenate([x, hard]), np.concatenate([y, np.zeros(len(hard), np.float32)])
    m = fit(x, y, args.l2[0])
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    np.savez(MODEL, w=m["w"], b=m["b"], mu=m["mu"], sd=m["sd"], l2=args.l2[0], n=len(y))
    print(f"saved {MODEL} ({len(y)} clips, train accuracy {((predict(m, x) >= 0.5) == y).mean():.3f})")


def scan(args):
    from backend.cameras import SharedModels
    from backend.fight import clip_embeddings, window_vector, FightClassifier
    from backend.search import ClipEmbedder

    clf = FightClassifier()
    shared, emb = SharedModels(), ClipEmbedder()
    for p in args.clips:
        seq, times = clip_embeddings(shared, emb, Path(p), with_times=True)
        k = int(config.FIGHT_CLF_WINDOW_S * config.FIGHT_CLF_FPS)
        probs = [(times[i + k - 1], clf.prob(window_vector(seq[i:i + k]))) for i in range(0, max(1, len(seq) - k + 1))]
        top = max(probs, key=lambda t: t[1]) if probs else (0, 0)
        print(f"{Path(p).name:32s} max p {top[1]:.2f} at {top[0]:.1f}s   windows >= {config.FIGHT_CLF_THRESHOLD}: "
              f"{sum(pr >= config.FIGHT_CLF_THRESHOLD for _, pr in probs)}/{len(probs)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract"); e.add_argument("--root", required=True); e.set_defaults(fn=extract)
    c = sub.add_parser("cv"); c.add_argument("--l2", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1, 1.0]); c.set_defaults(fn=cv)
    t = sub.add_parser("train"); t.add_argument("--l2", type=float, nargs="+", default=[1e-2]); t.set_defaults(fn=train)
    mi = sub.add_parser("mine"); mi.add_argument("--root", required=True); mi.add_argument("clips", nargs="*")
    mi.set_defaults(fn=mine)
    s = sub.add_parser("scan"); s.add_argument("clips", nargs="+"); s.set_defaults(fn=scan)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
