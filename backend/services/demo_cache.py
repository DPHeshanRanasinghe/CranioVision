"""
CranioVision — Redis-backed demo cache.

Why this exists
---------------
A full analysis is ~5 minutes on CPU (three models, sliding-window inference,
ANTs registration, mesh extraction). That is fine in a clinic and fatal in a
7-minute pitch. This caches everything a completed job serves — the result
JSON, every artifact file, the XAI payload and the generated PDF — keyed by the
CONTENT of the input scans, so a re-run of the same case is served in well
under a second.

Design decisions worth knowing
------------------------------
* **Content-addressed, not name-addressed.** The key is a SHA-256 over the
  actual scan bytes, not the case folder name. Renaming a folder, or dropping
  different scans into a folder that reuses a known case id, produces a MISS
  rather than another patient's cached result. (The atlas cache learned this
  the hard way — see registration.py.)
* **Fails open, always.** If Redis is down, unreachable, or the entry is
  corrupt, every call here degrades to "no cache" and the normal pipeline
  runs. A cache must never be able to break the demo it exists to protect.
* **Artifacts travel as one gzipped tar.** A warmed job is ~43 MB on disk
  (meshes dominate); a single blob keeps the entry atomic — you can never
  restore half a job.
* **No eviction.** maxmemory-policy is `noeviction` in compose: silently
  dropping a warmed case mid-demo is far worse than a loud error while
  warming.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Bump when the cached payload shape changes — old entries then simply miss
# instead of being mis-read.
SCHEMA = "v1"
NS = f"cv:demo:{SCHEMA}"
INDEX = f"{NS}:index"

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
ENABLED = os.environ.get("DEMO_CACHE", "1").lower() not in ("0", "false", "no")

_client = None
_warned = False


def _log(msg: str) -> None:
    print(f"[demo-cache] {msg}", flush=True)


def client():
    """Lazily connect. Returns None if the cache is off or unreachable."""
    global _client, _warned
    if not ENABLED:
        return None
    if _client is not None:
        return _client
    try:
        import redis  # imported lazily so the backend runs without the dep
        c = redis.Redis.from_url(REDIS_URL, socket_connect_timeout=1.5,
                                 socket_timeout=15)
        c.ping()
        _client = c
        _log(f"connected to {REDIS_URL}")
        return _client
    except Exception as e:
        if not _warned:
            _log(f"unavailable ({type(e).__name__}: {e}) — running without cache")
            _warned = True
        return None


def available() -> bool:
    return client() is not None


# ---------------------------------------------------------------------------
# KEYING
# ---------------------------------------------------------------------------

def case_key(case_dict: Dict[str, Any]) -> Optional[str]:
    """
    SHA-256 over the case's input scan bytes (+ GT mask when present).

    Returns None if any input is unreadable — a missing input means we cannot
    prove this is the same case, so we must not risk serving a stale hit.
    """
    paths: List[str] = []
    img = case_dict.get("image")
    if isinstance(img, (list, tuple)):
        paths.extend(str(p) for p in img)
    elif img:
        paths.append(str(img))
    if case_dict.get("label"):
        paths.append(str(case_dict["label"]))
    if not paths:
        return None

    h = hashlib.sha256()
    for p in sorted(paths):
        try:
            h.update(Path(p).name.encode())
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
        except Exception:
            return None
    return h.hexdigest()[:32]


def _k(key: str) -> str:
    return f"{NS}:{key}"


# ---------------------------------------------------------------------------
# PACKING
# ---------------------------------------------------------------------------

def _tar_dir(d: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=6) as tf:
        for p in sorted(d.rglob("*")):
            if p.is_file():
                tf.add(p, arcname=str(p.relative_to(d)))
    return buf.getvalue()


def _untar_to(blob: bytes, d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        for m in tf.getmembers():
            # refuse absolute paths / traversal even though we wrote it
            name = m.name.replace("\\", "/")
            if name.startswith("/") or ".." in Path(name).parts:
                continue
            tf.extract(m, path=d)


# ---------------------------------------------------------------------------
# READ / WRITE
# ---------------------------------------------------------------------------

def get(key: str) -> Optional[Dict[str, Any]]:
    c = client()
    if c is None or not key:
        return None
    try:
        raw = c.hgetall(_k(key))
        if not raw or b"meta" not in raw:
            return None
        meta = json.loads(raw[b"meta"].decode())
        return {
            "meta": meta,
            "artifacts": raw.get(b"artifacts"),
            "report": raw.get(b"report"),
        }
    except Exception as e:
        _log(f"read failed ({type(e).__name__}: {e}) — treating as miss")
        return None


def put(
    key: str,
    *,
    case_id: str,
    result_json: Dict[str, Any],
    artifacts_dir: Path,
    xai: Optional[Dict[str, Any]] = None,
    report_path: Optional[Path] = None,
) -> bool:
    """Store/refresh one case. Returns True on success."""
    c = client()
    if c is None or not key:
        return False
    try:
        meta = {
            "case_id": case_id,
            "result": result_json,
            "xai": xai or {},
            "report_name": report_path.name if report_path and report_path.exists() else None,
            "stored_at": time.time(),
            "schema": SCHEMA,
        }
        payload: Dict[bytes, Any] = {
            b"meta": json.dumps(meta).encode(),
            b"artifacts": _tar_dir(artifacts_dir),
        }
        if report_path and report_path.exists():
            payload[b"report"] = report_path.read_bytes()

        c.hset(_k(key), mapping=payload)
        c.sadd(INDEX, key)
        size = len(payload[b"artifacts"]) / 1e6
        _log(f"stored {case_id} key={key[:12]} artifacts={size:.1f}MB "
             f"xai={list((xai or {}).keys())} report={'yes' if b'report' in payload else 'no'}")
        return True
    except Exception as e:
        _log(f"write failed ({type(e).__name__}: {e}) — continuing uncached")
        return False


# ---------------------------------------------------------------------------
# ADMIN
# ---------------------------------------------------------------------------

def stats() -> Dict[str, Any]:
    c = client()
    if c is None:
        return {"enabled": False, "reachable": False, "url": REDIS_URL, "entries": []}
    try:
        info = c.info("memory")
        keys = sorted(k.decode() for k in c.smembers(INDEX))
        entries = []
        for k in keys:
            raw = c.hget(_k(k), b"meta")
            if raw is None:
                continue
            m = json.loads(raw.decode())
            entries.append({
                "key": k,
                "case_id": m.get("case_id"),
                "xai": list(m.get("xai", {}).keys()),
                "report": bool(m.get("report_name")),
                "stored_at": m.get("stored_at"),
                "artifacts_mb": round((c.hstrlen(_k(k), b"artifacts") or 0) / 1e6, 1),
            })
        return {
            "enabled": True, "reachable": True, "url": REDIS_URL,
            "used_memory_human": info.get("used_memory_human"),
            "count": len(entries), "entries": entries,
        }
    except Exception as e:
        return {"enabled": True, "reachable": False, "error": f"{type(e).__name__}: {e}"}


def flush() -> Dict[str, Any]:
    """Drop every cached case (leaves other keys in the DB alone)."""
    c = client()
    if c is None:
        return {"flushed": 0, "reachable": False}
    try:
        keys = [k.decode() for k in c.smembers(INDEX)]
        if keys:
            c.delete(*[_k(k) for k in keys])
        c.delete(INDEX)
        _log(f"flushed {len(keys)} entrie(s)")
        return {"flushed": len(keys), "reachable": True}
    except Exception as e:
        return {"flushed": 0, "reachable": False, "error": str(e)}


def drop(key: str) -> bool:
    c = client()
    if c is None:
        return False
    try:
        c.delete(_k(key))
        c.srem(INDEX, key)
        return True
    except Exception:
        return False
