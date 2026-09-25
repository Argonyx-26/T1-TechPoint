from types import SimpleNamespace

import numpy as np
import pytest

from backend import config
from backend.search import SearchIndex, SearchRefused, check_query

CROP = np.zeros((80, 40, 3), np.uint8)


def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def vec(seed):
    """Deterministic random 512-d unit vector (unrelated vectors: cosine ~0)."""
    return unit(np.random.default_rng(seed).normal(size=512))


def near(base, seed, amount=0.4):
    """Same subject seen again: cosine about 1/sqrt(1 + amount^2) (~0.93)."""
    return unit(base + amount * vec(seed))


def toward(base, seed, weight):
    """A crop that matches a text query partly: cosine grows with weight."""
    return unit(weight * base + vec(seed))


@pytest.fixture
def index():
    return SearchIndex(camera_manager=None, embedder=None, start_worker=False)


def appear(index, cam, tid, t, emb, cls="person", bbox=(100, 100, 160, 260), zone=None):
    """A track is seen at time t and one embedded crop arrives for it."""
    track = SimpleNamespace(track_id=tid, cls_name=cls, bbox=bbox)
    with index._lock:
        s = index._sighting(cam, track, t, (640, 480))
        s.first_seen = min(s.first_seen, t)
        s.last_seen = t
    index.ingest([(cam, tid, CROP, 50.0, t, zone, bbox)], np.array([emb]))
    return index.active[(cam, tid)]


def leave(index, now):
    index.housekeeping(now)


# ---------------- guardrail ----------------
@pytest.mark.parametrize("q", ["person with a backpack", "person in a black t-shirt", "person wearing glasses",
                               "person holding a bottle", "man in a white shirt with a red bag", "woman with a suitcase"])
def test_appearance_queries_are_allowed(q):
    assert check_query(q) == q


@pytest.mark.parametrize("q", ["black man", "white guy with a bag", "muslim woman", "indian person", "asian person",
                               "person of a certain race", "hindu man", "face of the man in red", "identify this person",
                               "dark skinned person", ""])
def test_protected_and_face_queries_are_refused(q):
    with pytest.raises(SearchRefused):
        check_query(q)


def test_search_endpoint_style_refusal_before_any_embedding(index):
    with pytest.raises(SearchRefused):
        index.search("jewish person", text_embedding=vec(1))


# ---------------- ring buffer, grouping, expiry ----------------
def test_results_grouped_per_subject_best_match_first(index):
    now = 1000.0
    backpack = vec(10)
    a = appear(index, "cam-1", 1, now, toward(backpack, 11, 0.3))
    appear(index, "cam-1", 1, now + 2, toward(backpack, 12, 0.6))   # same subject (still tracked), better match
    appear(index, "cam-2", 7, now + 6, vec(13))                         # someone else
    import time as _t
    orig = _t.time
    _t.time = lambda: now + 10
    try:
        results = index.search("person with a backpack", minutes=30, text_embedding=backpack)
    finally:
        _t.time = orig
    assert [r["global_id"] for r in results][0] == a.global_id
    assert len({r["global_id"] for r in results}) == len(results) == 2
    assert results[0]["score"] > results[1]["score"]
    assert results[0]["thumb_url"].startswith("/api/search/thumb/")


def test_entries_expire_after_retention(index, monkeypatch):
    appear(index, "cam-1", 1, 1000.0, vec(1))
    appear(index, "cam-1", 2, 1000.0 + 20 * 60, vec(2))
    leave(index, 1000.0 + 31 * 60)
    assert [e.track_id for e in index.entries] == [2]


def test_ring_buffer_is_bounded(monkeypatch):
    monkeypatch.setattr(config, "SEARCH_MAX_ENTRIES", 5)
    idx = SearchIndex(start_worker=False)
    for i in range(12):
        appear(idx, "cam-1", i, 1000.0 + i, vec(i))
    assert len(idx.entries) == 5
    assert [e.track_id for e in idx.entries] == [7, 8, 9, 10, 11]


# ---------------- re-identification ----------------
def test_same_person_on_another_camera_keeps_the_global_id(index):
    person = vec(100)
    a = appear(index, "cam-1", 1, 1000.0, person)
    leave(index, 1004.0)                                   # lost > 3s: sighting closes
    b = appear(index, "cam-2", 5, 1020.0, near(person, 101))
    assert b.global_id == a.global_id
    assert b.reid_score >= config.REID_THRESHOLD
    assert b.reid_from == "cam-1"


def test_different_person_gets_a_new_global_id(index):
    a = appear(index, "cam-1", 1, 1000.0, vec(100))
    leave(index, 1004.0)
    b = appear(index, "cam-2", 5, 1020.0, vec(200))
    assert b.global_id != a.global_id and b.reid_score is None


def test_someone_still_visible_elsewhere_is_not_merged(index):
    person = vec(100)
    a = appear(index, "cam-1", 1, 1000.0, person)
    b = appear(index, "cam-2", 5, 1000.5, near(person, 101))   # a is still on cam-1
    assert b.global_id != a.global_id


def test_reid_window_is_five_minutes(index):
    person = vec(100)
    a = appear(index, "cam-1", 1, 1000.0, person)
    leave(index, 1004.0)
    late = appear(index, "cam-2", 5, 1000.0 + 6 * 60, near(person, 101))
    assert late.global_id != a.global_id


def test_bags_get_their_own_ids_and_are_not_reidentified(index):
    bag = appear(index, "cam-1", 3, 1000.0, vec(5), cls="backpack")
    assert bag.global_id.startswith("B")


def test_follow_reacquired_event_on_another_camera(index):
    person = vec(100)
    a = appear(index, "cam-1", 1, 1000.0, person)
    index.following[a.global_id] = 999.0
    leave(index, 1004.0)
    appear(index, "cam-2", 5, 1015.0, near(person, 101))
    events = index.follow_status()["events"]
    assert events and events[-1]["kind"] == "reacquired" and events[-1]["camera_id"] == "cam-2"
    assert "RE-ACQUIRED on cam-2" in events[-1]["message"]


def test_route_and_backtrack_subject(index):
    person = vec(100)
    a = appear(index, "cam-1", 1, 1000.0, person)
    leave(index, 1004.0)
    appear(index, "cam-2", 5, 1030.0, near(person, 101))
    route = index.route(a.global_id)
    assert [r["camera_id"] for r in route] == ["cam-1", "cam-2"]
    assert route[0]["time_out"] is not None and route[1]["time_out"] is None
    alert = SimpleNamespace(camera_id="cam-2", track_ids=[5], timestamp=1030.0, bbox=None)
    assert index.subject_for_alert(alert) == a.global_id
    weapon = SimpleNamespace(camera_id="cam-2", track_ids=[], timestamp=1030.0, bbox=(120, 150, 140, 170))
    assert index.subject_for_alert(weapon) == a.global_id


def test_ambiguous_match_between_two_people_is_not_merged(index):
    person = vec(100)
    twin = near(person, 300, amount=0.05)            # someone who looks almost the same
    appear(index, "cam-1", 1, 1000.0, person)
    appear(index, "cam-1", 2, 1000.0, twin)
    leave(index, 1004.0)
    c = appear(index, "cam-2", 5, 1020.0, near(person, 101))
    assert c.reid_score is None, "two near-equal candidates: don't guess"


def test_reid_is_revisited_as_the_mean_improves(index):
    person = vec(100)
    a = appear(index, "cam-1", 1, 1000.0, person)
    leave(index, 1004.0)
    # First crop on the new camera is a poor view (below threshold)...
    first = unit(0.9 * person + vec(401))
    b = appear(index, "cam-2", 5, 1010.0, first)
    assert b.global_id != a.global_id and b.reid_score is None
    fresh = b.global_id
    # ...better views arrive; the running mean crosses the threshold.
    for i in range(3):
        appear(index, "cam-2", 5, 1011.0 + i, near(person, 500 + i, amount=0.2))
    assert b.global_id == a.global_id and b.reid_score >= config.REID_THRESHOLD
    assert all(e.global_id == a.global_id for e in index.entries if e.camera_id == "cam-2"), fresh


def test_reacquired_is_about_the_last_camera_not_the_best_match(index):
    person = vec(100)
    a = appear(index, "cam-1", 1, 1000.0, person)          # gate, very sharp match later
    leave(index, 1004.0)
    appear(index, "cam-2", 5, 1010.0, near(person, 101))   # walks to the lobby
    leave(index, 1014.0)
    index.following[a.global_id] = 1015.0
    back = appear(index, "cam-1", 9, 1030.0, near(person, 102, amount=0.1))  # best match is the old gate visit
    assert back.global_id == a.global_id
    ev = index.follow_status()["events"]
    assert ev and ev[-1]["camera_id"] == "cam-1", "last seen on cam-2 -> re-acquired on cam-1"
