"""Ask Vigil: natural-language search, cross-camera re-identification,
live follow and backtrack, across every camera.

Appearance only: clothing and carried objects, embedded with OpenCLIP
(ViT-B-32, laion2b_s34b_b79k, cached under models/clip so it works offline).
There is no face recognition, and queries about race, ethnicity or religion
are refused.

The indexer never blocks a camera: the per-frame hook only crops (at most
once per second per track) and hands work to a bounded queue that drops
when full; a single worker thread embeds in batches.

Every (camera, ByteTrack id) visit is a Sighting. Its mean embedding is
compared with people who recently left any camera; a match above
REID_THRESHOLD within REID_WINDOW_S gives it the same global id ("G12").
Sightings also feed blind-spot tracking (backend/movements.py).
"""
import itertools
import queue
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend import config, features
from backend.audit import audit
from backend.geometry import point_in_polygon

PERSON = "person"
PROMPTS = ("a CCTV photo of {q}", "a security camera image of {q}", "a photo of {q}")


class SearchRefused(ValueError):
    """A query the guardrail will not run (HTTP 400)."""


# ---------------------------------------------------------------------------
# Guardrail
# ---------------------------------------------------------------------------
_PROTECTED = re.compile(
    r"\b(race|racial|ethnic\w*|religio\w*|caste|skin\s*(tone|colou?r)|complexion|"
    r"muslim|hindu|christian|jew\w*|sikh|buddhist|islam\w*|arab|african|asian|caucasian|latin[oa]|hispanic|"
    r"indian|chinese|pakistani|nepali|bangladeshi|european|nationality|tribe|tribal)\b",
    re.I,
)
# "black person" / "white guy" etc. describe a person by skin colour, while
# "person in a black t-shirt" describes clothing and is fine.
_SKIN_COLOUR = re.compile(
    r"\b(black|white|brown|dark|fair|yellow)[- ]?(skinned\s+)?(man|men|woman|women|person|people|guy|guys|girl|boy|kid|lady|folks?)\b",
    re.I,
)
_FACE = re.compile(r"\b(face|facial|identify|identity|who is|whose|name of)\b", re.I)


def check_query(query: str) -> str:
    q = (query or "").strip()
    if not q:
        raise SearchRefused("Type what the person looks like, e.g. 'person with a backpack'")
    if len(q) > 200:
        raise SearchRefused("Query too long")
    if _PROTECTED.search(q) or _SKIN_COLOUR.search(q):
        raise SearchRefused("Ask Vigil searches appearance only (clothing, carried objects); "
                            "race, ethnicity and religion are not searchable")
    if _FACE.search(q):
        raise SearchRefused("Ask Vigil does not do face recognition or identify people; describe clothing or objects")
    return q


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------
def local_clip_weights() -> Optional[Path]:
    hits = sorted(Path(config.CLIP_CACHE_DIR).glob("models--*/snapshots/*/open_clip_model.safetensors"))
    return hits[0] if hits else None


class ClipEmbedder:
    """OpenCLIP loaded lazily from the local cache (no network at runtime)."""

    def __init__(self):
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self.device = config.DEVICE

    def _ensure(self):
        if self._model is not None:
            return
        import open_clip
        import torch

        weights = local_clip_weights()
        pretrained = str(weights) if weights else config.CLIP_PRETRAINED
        model, _, preprocess = open_clip.create_model_and_transforms(
            config.CLIP_MODEL, pretrained=pretrained, cache_dir=str(config.CLIP_CACHE_DIR))
        model.eval().to(self.device)
        if self.device.startswith("cuda"):
            model = model.half()
        self._model, self._preprocess = model, preprocess
        self._tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL)
        self._torch = torch
        audit("MODEL_LOADED", f"search: OpenCLIP {config.CLIP_MODEL} ({'local cache' if weights else pretrained})")

    def encode_images(self, crops_bgr: List[np.ndarray]) -> np.ndarray:
        from PIL import Image

        with self._lock:
            self._ensure()
            torch = self._torch
            batch = torch.stack([self._preprocess(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))) for c in crops_bgr])
            batch = batch.to(self.device)
            if self.device.startswith("cuda"):
                batch = batch.half()
            with torch.no_grad():
                emb = self._model.encode_image(batch).float()
            emb = emb / emb.norm(dim=-1, keepdim=True)
            return emb.cpu().numpy()

    def encode_text(self, texts: List[str]) -> np.ndarray:
        with self._lock:
            self._ensure()
            torch = self._torch
            tokens = self._tokenizer(texts).to(self.device)
            with torch.no_grad():
                emb = self._model.encode_text(tokens).float()
            emb = emb / emb.norm(dim=-1, keepdim=True)
            return emb.cpu().numpy()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class Entry:
    id: int
    camera_id: str
    track_id: int
    global_id: str
    cls: str
    embedding: np.ndarray
    thumb: bytes
    ts: float
    zone: Optional[str]
    bbox: Tuple[float, float, float, float]
    sharpness: float


@dataclass
class Sighting:
    """One visit of one tracked subject to one camera."""
    camera_id: str
    track_id: int
    cls: str
    first_seen: float
    last_seen: float
    bbox: Tuple[float, float, float, float]
    frame_size: Tuple[int, int] = (0, 0)
    global_id: Optional[str] = None
    reid_score: Optional[float] = None      # similarity that linked it to an earlier sighting
    reid_from: Optional[str] = None         # camera_id of that earlier sighting
    mean: Optional[np.ndarray] = None
    n_emb: int = 0
    best_entry: Optional[int] = None
    best_sharp: float = -1.0
    zones: List[str] = field(default_factory=list)
    closed: bool = False
    exit_edge: Optional[str] = None
    # sampling state
    last_sample: float = 0.0
    submitted: int = 0
    bucket_start: float = 0.0
    bucket_best: Optional[tuple] = None
    reacquired_at: float = 0.0
    reid_final: bool = False            # stop revisiting re-ID (matched, or enough crops seen)

    def add_embedding(self, emb: np.ndarray):
        self.mean = emb if self.mean is None else (self.mean * self.n_emb + emb) / (self.n_emb + 1)
        self.mean = self.mean / (np.linalg.norm(self.mean) + 1e-9)
        self.n_emb += 1


def exit_edge(bbox, frame_size) -> Optional[str]:
    """Which frame edge a subject left through (nearest edge to its last box)."""
    w, h = frame_size
    if not w or not h:
        return None
    x1, y1, x2, y2 = bbox
    gaps = {"left": x1, "right": w - x2, "top": y1, "bottom": h - y2}
    return min(gaps, key=gaps.get)


def sharpness(crop) -> float:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def match_percent(similarity: float) -> int:
    """Text-to-crop CLIP cosine: ~0.20 is noise, ~0.36 an unmistakable match
    (measured on our clips). A relative score, not a probability."""
    return int(round(100 * min(1.0, max(0.0, (similarity - 0.20) / 0.16))))


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------
class SearchIndex:
    def __init__(self, camera_manager=None, embedder: Optional[ClipEmbedder] = None, start_worker: bool = True):
        self.cameras = camera_manager
        self.embedder = embedder or ClipEmbedder()
        self._lock = threading.RLock()
        self.entries: Deque[Entry] = deque(maxlen=config.SEARCH_MAX_ENTRIES)
        self.active: Dict[Tuple[str, int], Sighting] = {}
        self.history: Deque[Sighting] = deque(maxlen=5000)   # closed sightings, newest last
        self._entry_ids = itertools.count(1)
        self._gids = itertools.count(1)
        self._items = itertools.count(1)
        self._jobs: "queue.Queue" = queue.Queue(maxsize=config.SEARCH_QUEUE_MAX)
        self.dropped_jobs = 0
        self.following: Dict[str, float] = {}               # global_id -> since
        self.follow_events: Deque[dict] = deque(maxlen=200)
        self._follow_seq = itertools.count(1)
        self.listeners = []                                  # callbacks(sighting) on close (movements)
        self._running = start_worker
        if start_worker:
            threading.Thread(target=self._worker, name="search-indexer", daemon=True).start()

    # -- camera hook (runs in each camera's processing thread) -------------------
    def on_frame(self, camera, frame, annotated):
        if not (features.enabled("search") or features.enabled("reid") or features.enabled("gap_tracking")):
            return
        now = time.time()
        h, w = frame.shape[:2]
        zones = [z.to_pixels(w, h) for z in camera.pipeline.zone_store.list()]
        tracks = getattr(camera.pipeline, "last_tracks", {}) or {}
        with self._lock:
            for track in tracks.values():
                if track.cls_name not in config.SEARCH_CLASSES or track.track_id < 0:
                    continue
                sighting = self._sighting(camera.id, track, now, (w, h))
                zone = next((z.name for z in zones if point_in_polygon(track.centroid, z.polygon)), None)
                if zone and (not sighting.zones or sighting.zones[-1] != zone):
                    sighting.zones.append(zone)
                if len(track.history) >= 2:
                    self._maybe_sample(camera.id, track, sighting, frame, zone, now)
        self._draw_follow(camera, annotated, tracks, now)

    def _sighting(self, cam_id: str, track, now: float, frame_size) -> Sighting:
        key = (cam_id, track.track_id)
        s = self.active.get(key)
        if s is not None and now - s.last_seen > config.SIGHTING_CLOSE_S:
            self._close(key, s)
            s = None
        if s is None:
            s = Sighting(camera_id=cam_id, track_id=track.track_id, cls=track.cls_name,
                         first_seen=now, last_seen=now, bbox=track.bbox, frame_size=frame_size)
            if track.cls_name != PERSON:
                s.global_id = f"B{next(self._items)}"
            self.active[key] = s
        s.last_seen = now
        s.bbox = track.bbox
        s.frame_size = frame_size
        return s

    def _maybe_sample(self, cam_id, track, s: Sighting, frame, zone, now):
        if now - s.last_sample < config.SEARCH_SAMPLE_S:
            return
        s.last_sample = now
        crop = padded_crop(frame, track.bbox, pad=config.SEARCH_CROP_PAD)
        if crop is None:
            return
        sharp = sharpness(crop)
        job = (cam_id, track.track_id, crop, sharp, now, zone, track.bbox)
        if s.submitted == 0 or now - s.first_seen <= config.REID_WARMUP_S:
            # A new subject is sampled every second at first: re-ID decides on
            # the mean of several crops (one crop is too noisy across cameras).
            self._submit(job, s)
            s.bucket_start = now
            return
        if s.bucket_best is None or sharp > s.bucket_best[3]:
            s.bucket_best = job
        if now - s.bucket_start >= config.SEARCH_KEEP_BEST_S:
            self._submit(s.bucket_best, s)
            s.bucket_best = None
            s.bucket_start = now

    def _submit(self, job, s: Sighting):
        try:
            self._jobs.put_nowait(job)
            s.submitted += 1
        except queue.Full:
            self.dropped_jobs += 1

    # -- worker ------------------------------------------------------------------------
    def _worker(self):
        while self._running:
            batch = []
            try:
                batch.append(self._jobs.get(timeout=0.5))
                while len(batch) < config.SEARCH_BATCH:
                    batch.append(self._jobs.get_nowait())
            except queue.Empty:
                pass
            if batch:
                try:
                    self.ingest(batch, self.embedder.encode_images([b[2] for b in batch]))
                except Exception as exc:
                    print(f"[search] embedding failed: {exc}")
            self.housekeeping(time.time())

    def ingest(self, batch, embeddings: np.ndarray):
        """Store embedded crops; assign global ids. Separate from the worker so
        tests can feed synthetic embeddings."""
        with self._lock:
            for (cam_id, track_id, crop, sharp, ts, zone, bbox), emb in zip(batch, embeddings):
                s = self.active.get((cam_id, track_id))
                if s is None:
                    s = next((h for h in reversed(self.history) if h.camera_id == cam_id and h.track_id == track_id), None)
                if s is None:
                    continue
                s.add_embedding(emb)
                if s.global_id is None:
                    self._assign_global(s, ts)
                elif s.cls == PERSON and not s.reid_final and s.reid_score is None:
                    self._revisit(s, ts)
                thumb = encode_thumb(crop)
                entry = Entry(id=next(self._entry_ids), camera_id=cam_id, track_id=track_id, global_id=s.global_id,
                              cls=s.cls, embedding=emb.astype(np.float32), thumb=thumb, ts=ts, zone=zone,
                              bbox=bbox, sharpness=sharp)
                self.entries.append(entry)
                if sharp > s.best_sharp:
                    s.best_sharp, s.best_entry = sharp, entry.id

    def _assign_global(self, s: Sighting, now: float):
        """Re-ID: same person as someone who recently left a camera?"""
        best, best_sim, runner_up = None, -1.0, -1.0
        if features.enabled("reid") and s.cls == PERSON and s.mean is not None:
            active_gids = {a.global_id for a in self.active.values()
                           if a is not s and a.global_id and now - a.last_seen <= config.REID_ACTIVE_S}
            for cand in list(self.history) + list(self.active.values()):
                if (cand is s or cand.cls != PERSON or cand.mean is None or not cand.global_id
                        or cand.global_id in active_gids
                        or now - cand.last_seen > config.REID_WINDOW_S
                        or now - cand.last_seen < config.REID_MIN_GAP_S):
                    continue
                sim = float(np.dot(cand.mean, s.mean))
                if sim > best_sim:
                    if best is not None and best.global_id != cand.global_id:
                        runner_up = best_sim
                    best, best_sim = cand, sim
                elif best is not None and cand.global_id != best.global_id and sim > runner_up:
                    runner_up = sim
        # Same person only when clearly above threshold AND clearly better than
        # anyone else: in a crowd, two near-equal candidates means "don't know".
        if best is not None and best_sim >= config.REID_THRESHOLD and best_sim - runner_up >= config.REID_MARGIN:
            s.global_id, s.reid_score, s.reid_from = best.global_id, best_sim, best.camera_id
            s.reid_final = True
            if s.global_id in self.following:
                # Re-acquired = the subject was last seen on a DIFFERENT camera
                # (the best-matching crop is often an older visit to this one).
                previous = max((x for x in list(self.history) + list(self.active.values())
                                if x is not s and x.global_id == s.global_id),
                               key=lambda x: x.last_seen, default=None)
                if previous is not None and previous.camera_id != s.camera_id:
                    self._follow_event("reacquired", s, best_sim)
        elif s.global_id is None:
            s.global_id = f"G{next(self._gids)}"

    def _revisit(self, s: Sighting, now: float):
        """A sighting that got a fresh id re-tries re-ID as its mean improves,
        for its first REID_REVISIT_EMBS crops."""
        if s.n_emb > config.REID_REVISIT_EMBS:
            s.reid_final = True
            return
        fresh = s.global_id
        s.global_id = None
        self._assign_global(s, now)
        if s.reid_score is None:
            s.global_id = fresh            # still nobody: keep the fresh id
            return
        s.reid_final = True
        for e in self.entries:            # its earlier crops move to the matched subject
            if e.camera_id == s.camera_id and e.track_id == s.track_id and e.global_id == fresh:
                e.global_id = s.global_id

    def housekeeping(self, now: float):
        with self._lock:
            for key, s in list(self.active.items()):
                if now - s.last_seen > config.SIGHTING_CLOSE_S:
                    self._close(key, s)
            horizon = now - config.SEARCH_RETENTION_MIN * 60
            while self.entries and self.entries[0].ts < horizon:
                self.entries.popleft()
            while self.history and self.history[0].last_seen < horizon:
                self.history.popleft()

    def _close(self, key, s: Sighting):
        self.active.pop(key, None)
        s.closed = True
        s.exit_edge = exit_edge(s.bbox, s.frame_size)
        if s.global_id is None and s.cls == PERSON:
            s.global_id = f"G{next(self._gids)}"
        self.history.append(s)
        for listener in self.listeners:
            try:
                listener(s)
            except Exception as exc:
                print(f"[search] listener failed: {exc}")

    # -- queries ---------------------------------------------------------------------------
    def all_sightings(self) -> List[Sighting]:
        with self._lock:
            return list(self.history) + list(self.active.values())

    def search(self, query: str, minutes: float = 30, limit: int = 12, camera_ids=None,
               text_embedding: Optional[np.ndarray] = None) -> List[dict]:
        q = check_query(query)
        if text_embedding is None:
            texts = [p.format(q=q) for p in PROMPTS]
            text_embedding = self.embedder.encode_text(texts).mean(axis=0)
            text_embedding = text_embedding / (np.linalg.norm(text_embedding) + 1e-9)
        since = time.time() - minutes * 60
        with self._lock:
            pool = [e for e in self.entries if e.ts >= since and (not camera_ids or e.camera_id in camera_ids)]
            sightings = self.all_sightings()
        if not pool:
            return []
        sims = np.stack([e.embedding for e in pool]) @ text_embedding
        best: Dict[str, Tuple[float, Entry]] = {}
        for sim, e in zip(sims, pool):
            if e.global_id not in best or sim > best[e.global_id][0]:
                best[e.global_id] = (float(sim), e)
        ranked = sorted(best.items(), key=lambda kv: -kv[1][0])[:limit]
        return [self._subject_summary(gid, sim, e, pool, sightings) for gid, (sim, e) in ranked]

    def _subject_summary(self, gid, sim, entry, pool, sightings) -> dict:
        mine = [s for s in sightings if s.global_id == gid]
        seen = sorted({s.camera_id for s in mine} | {e.camera_id for e in pool if e.global_id == gid})
        reid = [s.reid_score for s in mine if s.reid_score is not None]
        zones = []
        for s in sorted(mine, key=lambda s: s.first_seen):
            for z in s.zones:
                if z not in zones:
                    zones.append(z)
        return {
            "global_id": gid,
            "score": round(sim, 4),
            "match": match_percent(sim),
            "cls": entry.cls,
            "thumb_url": f"/api/search/thumb/{entry.id}",
            "best_camera_id": entry.camera_id,
            "cameras_seen": [self._camera_label(c) for c in seen],
            "camera_ids": seen,
            "first_seen": min([s.first_seen for s in mine] or [entry.ts]),
            "last_seen": max([s.last_seen for s in mine] or [entry.ts]),
            "zones_visited": zones,
            "reid_confidence": round(max(reid), 3) if reid else None,
            "following": gid in self.following,
        }

    def _camera_label(self, cam_id: str) -> dict:
        cam = self.cameras.get(cam_id) if self.cameras else None
        if cam is None:
            return {"id": cam_id, "code": cam_id, "name": cam_id, "place": cam_id}
        return {"id": cam_id, "code": cam.code, "name": cam.name, "place": cam.place}

    def thumb(self, entry_id: int) -> Optional[bytes]:
        with self._lock:
            for e in self.entries:
                if e.id == entry_id:
                    return e.thumb
        return None

    def route(self, gid: str, until: Optional[float] = None) -> List[dict]:
        """All sightings of a subject across cameras, in time order."""
        rows = []
        for s in sorted(self.all_sightings(), key=lambda s: s.first_seen):
            if s.global_id != gid or (until is not None and s.first_seen > until + 1):
                continue
            rows.append({
                "global_id": gid,
                **self._camera_label(s.camera_id),
                "camera_id": s.camera_id,
                "time_in": s.first_seen,
                "time_out": None if not s.closed else s.last_seen,
                "last_seen": s.last_seen,
                "exit_edge": s.exit_edge,
                "zones": list(s.zones),
                "thumb_url": f"/api/search/thumb/{s.best_entry}" if s.best_entry else None,
                "reid_confidence": round(s.reid_score, 3) if s.reid_score is not None else None,
            })
        return rows

    def subject_for_alert(self, alert) -> Optional[str]:
        """Global id of the person an alert is about: its track ids, or (for
        weapons, which have no track) the person box around the weapon."""
        sightings = self.all_sightings()
        for s in sightings:
            if (s.camera_id == alert.camera_id and s.track_id in (alert.track_ids or [])
                    and s.first_seen - 2 <= alert.timestamp <= s.last_seen + 2 and s.cls == PERSON):
                return s.global_id
        if alert.bbox:
            cx = (alert.bbox[0] + alert.bbox[2]) / 2
            cy = (alert.bbox[1] + alert.bbox[3]) / 2
            with self._lock:
                near = [e for e in self.entries if e.camera_id == alert.camera_id and e.cls == PERSON
                        and abs(e.ts - alert.timestamp) <= 3]
            for e in sorted(near, key=lambda e: abs(e.ts - alert.timestamp)):
                x1, y1, x2, y2 = e.bbox
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    return e.global_id
        return None

    # -- follow -----------------------------------------------------------------------------
    def follow(self, gid: str):
        with self._lock:
            self.following[gid] = time.time()
            where = max((s for s in self.all_sightings() if s.global_id == gid), key=lambda s: s.last_seen, default=None)
        audit("FOLLOW_STARTED", f"Subject #{gid}" + (f" last on {self._camera_label(where.camera_id)['code']}" if where else ""),
              actor="operator", camera_id=where.camera_id if where else None)

    def unfollow(self, gid: str):
        with self._lock:
            self.following.pop(gid, None)
        audit("FOLLOW_STOPPED", f"Subject #{gid}", actor="operator")

    def follow_status(self, since_event: int = 0) -> dict:
        subjects = []
        for gid, since in list(self.following.items()):
            live = [s for s in self.all_sightings() if s.global_id == gid]
            last = max(live, key=lambda s: s.last_seen, default=None)
            subjects.append({
                "global_id": gid,
                "since": since,
                "camera": self._camera_label(last.camera_id) if last else None,
                "last_seen": last.last_seen if last else None,
                "visible": bool(last and not last.closed and time.time() - last.last_seen < 2),
                "route": self.route(gid),
            })
        events = [e for e in self.follow_events if e["seq"] > since_event]
        return {"following": subjects, "events": events}

    def _follow_event(self, kind: str, s: Sighting, sim: float):
        label = self._camera_label(s.camera_id)
        event = {"seq": next(self._follow_seq), "ts": time.time(), "kind": kind, "global_id": s.global_id,
                 "camera_id": s.camera_id, "camera": label, "confidence": round(sim, 3),
                 "message": f"RE-ACQUIRED on {label['place']} ({label['code']})"}
        self.follow_events.append(event)
        s.reacquired_at = time.time()
        audit("FOLLOW_REACQUIRED", f"Subject #{s.global_id} {event['message']}, match {sim:.0%}",
              camera_id=s.camera_id)

    def _draw_follow(self, camera, annotated, tracks, now):
        if not self.following:
            return
        pulse = 2 + int(3 * (0.5 + 0.5 * np.sin(now * 8)))
        for track in tracks.values():
            s = self.active.get((camera.id, track.track_id))
            if s is None or s.global_id not in self.following:
                continue
            x1, y1, x2, y2 = [int(v) for v in track.bbox]
            cyan = (230, 211, 95)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), cyan, pulse)
            trail = [p for t, p in track.history if now - t <= 5.0]
            if len(trail) > 1:
                pts = np.array([[int(x), int(y)] for x, y in trail], np.int32)
                cv2.polylines(annotated, [pts], False, cyan, 2)
            conf = f" {s.reid_score:.0%}" if s.reid_score is not None else ""
            label = f"FOLLOWING #{s.global_id}{conf}"
            if now - getattr(s, "reacquired_at", 0) < 3:
                label = f"RE-ACQUIRED #{s.global_id}{conf}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), cyan, -1)
            cv2.putText(annotated, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 26, 17), 2)

    def reid_diagnostics(self, limit: int = 15) -> dict:
        """For tuning REID_THRESHOLD on real cameras: person sightings and the
        most similar cross-camera pairs (same subject or not)."""
        people = [s for s in self.all_sightings() if s.cls == PERSON and s.mean is not None]
        pairs = []
        for i, a in enumerate(people):
            for b in people[i + 1:]:
                if a.camera_id != b.camera_id:
                    pairs.append({
                        "a": f"{a.camera_id}#{a.track_id} {a.global_id}", "b": f"{b.camera_id}#{b.track_id} {b.global_id}",
                        "similarity": round(float(a.mean @ b.mean), 3), "same_subject": a.global_id == b.global_id,
                        "gap_s": round(max(a.first_seen, b.first_seen) - min(a.last_seen, b.last_seen), 1),
                    })
        pairs.sort(key=lambda p: -p["similarity"])
        return {
            "threshold": config.REID_THRESHOLD, "margin": config.REID_MARGIN,
            "sightings": [{"camera_id": s.camera_id, "track_id": s.track_id, "global_id": s.global_id,
                           "crops": s.n_emb, "reid_score": s.reid_score, "closed": s.closed,
                           "seconds": round(s.last_seen - s.first_seen, 1)} for s in people][-40:],
            "top_cross_camera_pairs": pairs[:limit],
        }

    def stats(self) -> dict:
        with self._lock:
            return {"entries": len(self.entries), "active_sightings": len(self.active),
                    "closed_sightings": len(self.history), "queued": self._jobs.qsize(),
                    "dropped": self.dropped_jobs, "following": list(self.following)}


def padded_crop(frame, bbox, pad: float = 0.1):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1, y1 = max(0, int(x1 - pad * bw)), max(0, int(y1 - pad * bh))
    x2, y2 = min(w, int(x2 + pad * bw)), min(h, int(y2 + pad * bh))
    if x2 - x1 < 12 or y2 - y1 < 24:
        return None
    return frame[y1:y2, x1:x2].copy()


def encode_thumb(crop, width: int = 160) -> bytes:
    h, w = crop.shape[:2]
    if w > width:
        crop = cv2.resize(crop, (width, max(1, int(h * width / w))), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 80])[1].tobytes()
