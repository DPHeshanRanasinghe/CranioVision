"""
Warm the demo cache: run every demo case once, end to end, so the finals
demo is served from Redis instead of recomputing.

Drives the real HTTP API rather than calling the pipeline directly — that way
what gets warmed is exactly what the demo will exercise (analysis, XAI, PDF),
through the same code paths.

    python scripts/warm_demo_cache.py                    # all demo cases
    python scripts/warm_demo_cache.py --cases A B        # specific cases
    python scripts/warm_demo_cache.py --skip-report      # faster, no PDFs
    python scripts/warm_demo_cache.py --force            # re-warm even if cached

Budget ~9 min per cold case on CPU. Start it and go do something else.
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
import json

DEFAULT_API = "http://localhost:8000"


def api(base, path, method="GET", body=None, timeout=1800):
    url = f"{base}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        return json.loads(raw) if raw else {}


def wait_done(base, job_id, label, poll=5, limit=2400):
    t0 = time.time()
    last = ""
    while time.time() - t0 < limit:
        st = api(base, f"/jobs/{job_id}/status")
        state = st["state"]
        prog = st.get("latest_progress") or {}
        msg = f"{prog.get('percent', 0):3d}%  {prog.get('message', '')[:58]}"
        if msg != last:
            print(f"    {msg}", flush=True)
            last = msg
        if state == "done":
            return True, time.time() - t0
        if state == "failed":
            print(f"    FAILED: {st.get('error', '')[:200]}", flush=True)
            return False, time.time() - t0
        time.sleep(poll)
    print("    TIMED OUT", flush=True)
    return False, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=DEFAULT_API)
    ap.add_argument("--cases", nargs="*", help="case ids (default: all)")
    ap.add_argument("--skip-xai", action="store_true")
    ap.add_argument("--skip-report", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="flush the cache first so every case recomputes")
    a = ap.parse_args()

    try:
        status = api(a.api, "/cache/status")
    except urllib.error.URLError as e:
        sys.exit(f"Cannot reach the backend at {a.api} ({e}).\n"
                 f"Start it first:  cd backend && uvicorn main:app --port 8000")

    if not status.get("reachable"):
        print("!! Redis is NOT reachable — nothing will be cached.")
        print("   Start it:  docker compose up -d")
        if input("   Continue anyway? [y/N] ").strip().lower() != "y":
            return

    if a.force:
        print("flushing cache:", api(a.api, "/cache/flush", "POST"))

    cases = a.cases or api(a.api, "/demo-cases")["cases"]
    already = {e["case_id"] for e in api(a.api, "/cache/status").get("entries", [])}

    print(f"\nwarming {len(cases)} case(s) via {a.api}")
    print(f"already cached: {sorted(already) or 'none'}\n")

    t_all = time.time()
    ok, skipped, failed = [], [], []

    for i, case in enumerate(cases, 1):
        if case in already and not a.force:
            print(f"[{i}/{len(cases)}] {case}  — already cached, skipping")
            skipped.append(case)
            continue

        print(f"[{i}/{len(cases)}] {case}")
        try:
            job = api(a.api, f"/demo-cases/{case}/run", "POST")["job_id"]
        except Exception as e:
            print(f"    could not start: {e}")
            failed.append(case)
            continue

        done, secs = wait_done(a.api, job, case)
        if not done:
            failed.append(case)
            continue
        print(f"    analysis {secs:.0f}s")

        if not a.skip_xai:
            t = time.time()
            try:
                api(a.api, f"/jobs/{job}/explain", "POST", {"model_name": "ensemble"})
                print(f"    xai      {time.time()-t:.0f}s")
            except Exception as e:
                print(f"    xai failed (cache still useful): {e}")

        if not a.skip_report:
            t = time.time()
            try:
                api(a.api, f"/jobs/{job}/report", "POST",
                    {"prediction_to_feature": "ensemble"})
                print(f"    report   {time.time()-t:.0f}s")
            except Exception as e:
                print(f"    report failed (cache still useful): {e}")

        ok.append(case)

    print(f"\ndone in {(time.time()-t_all)/60:.1f} min")
    print(f"  warmed  : {len(ok)}   {ok}")
    print(f"  skipped : {len(skipped)}")
    print(f"  failed  : {len(failed)}   {failed}")

    final = api(a.api, "/cache/status")
    print(f"\ncache now holds {final.get('count')} case(s), "
          f"{final.get('used_memory_human')} in Redis")
    for e in final.get("entries", []):
        print(f"  {e['case_id']:24} {e['artifacts_mb']:6.1f}MB  "
              f"xai={e['xai'] or '-'}  report={'yes' if e['report'] else 'no'}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
