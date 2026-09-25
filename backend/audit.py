"""Append-only, tamper-evident audit log (SQLite).

Every entry stores prev_hash and hash = sha256(prev_hash + canonical JSON of
the entry), so the entries form a hash chain: editing or deleting any row
breaks every hash after it, and verify() reports the first broken id.

A chain alone can't reveal the *last* rows being deleted (what remains is
still a valid, shorter chain), so the id + hash of the newest entry is also
written to a small head file next to the database on every append; verify()
checks the chain ends exactly there.

There is deliberately no update or delete API. The dashboard's HIDE button
only hides entries on screen.
"""
import csv
import hashlib
import io
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Iterable, List, Optional

from backend import config

GENESIS_HASH = "0" * 64
FIELDS = ("id", "ts", "actor", "action", "detail", "camera_id", "alert_id", "prev_hash", "hash")


def _canonical(entry: dict) -> str:
    body = {k: entry[k] for k in FIELDS if k not in ("hash", "prev_hash")}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_hash(prev_hash: str, entry: dict) -> str:
    return hashlib.sha256((prev_hash + _canonical(entry)).encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self, path: Path = None):
        self.path = Path(path or config.AUDIT_DB)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.head_path = self.path.with_suffix(".head")
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS audit (
                id INTEGER PRIMARY KEY,
                ts REAL NOT NULL,
                actor TEXT NOT NULL,
                action TEXT NOT NULL,
                detail TEXT NOT NULL,
                camera_id TEXT,
                alert_id INTEGER,
                prev_hash TEXT NOT NULL,
                hash TEXT NOT NULL
            )"""
        )
        self._db.commit()

    def append(self, action: str, detail: str = "", actor: str = "system",
               camera_id: Optional[str] = None, alert_id: Optional[int] = None) -> dict:
        if actor not in ("operator", "system"):
            raise ValueError("actor must be 'operator' or 'system'")
        with self._lock:
            last = self._db.execute("SELECT id, hash FROM audit ORDER BY id DESC LIMIT 1").fetchone()
            entry = {
                "id": (last["id"] + 1) if last else 1,
                "ts": round(time.time(), 3),
                "actor": actor,
                "action": str(action)[:64],
                "detail": str(detail)[:2000],
                "camera_id": camera_id,
                "alert_id": alert_id,
                "prev_hash": last["hash"] if last else GENESIS_HASH,
            }
            entry["hash"] = compute_hash(entry["prev_hash"], entry)
            self._db.execute(
                f"INSERT INTO audit ({','.join(FIELDS)}) VALUES ({','.join('?' * len(FIELDS))})",
                [entry[k] for k in FIELDS],
            )
            self._db.commit()
            self.head_path.write_text(json.dumps({"id": entry["id"], "hash": entry["hash"]}))
            return entry

    def list(self, limit: int = 200, action: Optional[str] = None,
             camera_id: Optional[str] = None) -> List[dict]:
        where, args = [], []
        if action:
            where.append("action = ?")
            args.append(action)
        if camera_id:
            where.append("camera_id = ?")
            args.append(camera_id)
        sql = "SELECT * FROM audit" + (f" WHERE {' AND '.join(where)}" if where else "")
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(max(1, min(int(limit), 10000)))
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, args).fetchall()]

    def _all(self) -> Iterable[dict]:
        with self._lock:
            return [dict(r) for r in self._db.execute("SELECT * FROM audit ORDER BY id ASC").fetchall()]

    def verify(self) -> dict:
        """{ok, entries, broken_at}: broken_at is the id of the first entry
        whose link or hash doesn't check out (None when intact)."""
        rows = self._all()
        prev_hash, expected_id = GENESIS_HASH, 1
        for row in rows:
            if (row["id"] != expected_id or row["prev_hash"] != prev_hash
                    or compute_hash(row["prev_hash"], row) != row["hash"]):
                return {"ok": False, "entries": len(rows), "broken_at": row["id"]}
            prev_hash, expected_id = row["hash"], row["id"] + 1
        head = self._read_head()
        if head is not None:
            last_id = rows[-1]["id"] if rows else 0
            if head["id"] != last_id or (rows and head["hash"] != rows[-1]["hash"]):
                # The newest entries were removed (or the table was replaced).
                return {"ok": False, "entries": len(rows), "broken_at": last_id + 1}
        return {"ok": True, "entries": len(rows), "broken_at": None}

    def _read_head(self) -> Optional[dict]:
        try:
            return json.loads(self.head_path.read_text())
        except (OSError, ValueError):
            return None

    def export(self, fmt: str = "csv") -> str:
        rows = self._all()
        if fmt == "json":
            return json.dumps(rows, indent=2, ensure_ascii=False)
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(FIELDS) + ["time"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(row["ts"]))})
        return buf.getvalue()

    def close(self):
        with self._lock:
            self._db.close()


_audit: Optional[AuditLog] = None
_audit_lock = threading.Lock()


def get_audit() -> AuditLog:
    """Process-wide audit log, created on first use at config.AUDIT_DB."""
    global _audit
    with _audit_lock:
        if _audit is None:
            _audit = AuditLog()
        return _audit


def set_audit(log: Optional[AuditLog]):
    """Swap the process-wide log (tests point it at a temp database)."""
    global _audit
    with _audit_lock:
        _audit = log


def audit(action: str, detail: str = "", actor: str = "system", **kwargs) -> Optional[dict]:
    """Best-effort append: auditing must never take the video pipeline down."""
    if not config.AUDIT_ENABLED:
        return None
    try:
        return get_audit().append(action, detail, actor=actor, **kwargs)
    except Exception as exc:  # pragma: no cover - disk full, locked db, ...
        print(f"[audit] could not record {action}: {exc}")
        return None


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
