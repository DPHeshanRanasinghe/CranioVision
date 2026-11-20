# Demo cache — runbook

Redis-backed cache so the finals demo serves instantly instead of recomputing.

**Measured on this machine:** cold run **521 s** → cached replay **3.4 s**.
Grad-CAM **~150 s → 0.36 s**. The replay survives a backend restart, because
the cache lives in Redis rather than process memory.

The cache is included in the CranioVision repository on both `main` and
`develop`. Run the commands below from this repository's `backend/` folder.

---

## Quick start

```bash
# 1. cache container
cd backend
docker compose up -d

# 2. backend (separate terminal)
uvicorn main:app --port 8000

# 3. warm every demo case — budget ~9 min per case, do this the night before
python scripts/warm_demo_cache.py
```

Check what's warm:

```bash
curl -s http://localhost:8000/cache/status
```

---

## Clearing the cache

| Goal | Command |
|---|---|
| Drop all cached cases, keep Redis running | `curl -X POST http://localhost:8000/cache/flush` |
| Same, from the container | `docker exec craniovision-cache redis-cli FLUSHALL` |
| Stop the cache, **keep** the data | `docker compose down` |
| Stop and **destroy** the data | `docker compose down -v` |
| Re-warm from scratch | `python scripts/warm_demo_cache.py --force` |

> **Note on `docker compose down`:** the cache is on a **named volume**
> (`craniovision-cache-data`), so a plain `down` — or a Docker restart, or a
> reboot — **keeps** your warmed cases. That is deliberate: warming is ~15
> minutes of compute, and losing it to an accidental restart on finals morning
> would hurt. Use `down -v` when you actually want it gone.
>
> If you'd rather have "delete the container = cache cleared", remove the
> `volumes:` block from `docker-compose.yml`; the cache then lives only in
> container memory and dies with it.

---

## How it works

**Key = SHA-256 of the input scan bytes**, not the folder name. So:

- same scans, renamed folder → **hit** (same patient)
- folder reused with different scans → **miss** (recomputes; never serves
  another patient's result)
- any input unreadable → **miss**

This is the same failure mode the atlas cache had — a name-keyed cache will
happily hand you the wrong patient's anatomy.

**Cached per case:** the `/result` payload, every artifact file (T1c,
predictions, MNI warps, both mesh sets, XAI heatmaps) as one gzipped tar
(~27 MB), the XAI payload, and the generated PDF.

**Written** at three points: after the pipeline finishes, after Grad-CAM, and
after a report is generated. Each refresh re-packs the entry, so a case is only
fully warm once all three have run — which is what `warm_demo_cache.py` does.

**On a hit** the job is rehydrated, artifacts unpacked to a fresh job dir, and
the pipeline stages replayed over ~1 s so the UI animates normally. Stage
messages are suffixed **"(cached)"** — visible in the progress panel, so the
replay is never passed off as a live 5-minute run.

---

## Failure behaviour

The cache **fails open**. If Redis is down, unreachable, or an entry is
corrupt, every call degrades to "no cache" and the normal pipeline runs. Set
`DEMO_CACHE=0` to disable it entirely.

Verified with Redis stopped: `available()` → `False`, `get()` → `None`,
`put()` → `False`, and jobs run normally.

`maxmemory-policy` is `noeviction` on purpose — silently evicting a warmed case
mid-demo would be worse than a loud error while warming.

---

## Demo-day checklist

- [ ] `docker compose up -d` — confirm `docker compose ps` shows **healthy**
- [ ] Backend started **after** Redis (it logs `[demo-cache] connected to …`)
- [ ] `curl /cache/status` lists every case you plan to show, each with
      `xai` populated and `report: true`
- [ ] Click through one cached case in the UI end to end
- [ ] Do **not** edit frontend source on demo day — Fast Refresh resets React
      state and loses the loaded job

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Runs take 5 min | Redis unreachable, or case not warmed | `curl /cache/status`; check `[demo-cache]` lines in backend log |
| `count: 0` after warming | Backend started before Redis | restart the backend |
| Warm case still recomputes | Scan files changed → different key | re-warm; it is doing the right thing |
| `used_memory` near 2 GB | Too many cases warmed | `POST /cache/flush`, warm only what you'll show |
