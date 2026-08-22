"""
CranioVision — FastAPI backend.

Run locally:
    cd backend
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

API docs auto-generated at http://localhost:8000/docs
"""
from __future__ import annotations

import sys
from pathlib import Path

# cranovision console output uses Unicode (✓, ✗, ═, ...) — on Windows this
# crashes with UnicodeEncodeError when stdout/stderr aren't a real UTF-8
# console (e.g. redirected to a log file, or run as a background service).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# Make project root importable so we can use src/cranovision
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router

app = FastAPI(
    title="CranioVision API",
    description=(
        "Backend for the CranioVision clinical brain-tumor analysis platform. "
        "Endpoints: upload a BraTS case folder, track progress, fetch results, "
        "request lazy XAI, download the clinical PDF report."
    ),
    version="1.0.0",
)

# CORS — allow the Next.js frontend running on localhost:3000 (dev)
# and the deployed Vercel domain (set via env var in production).
import os
allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
prod_origin = os.environ.get("FRONTEND_ORIGIN")
if prod_origin:
    allowed_origins.append(prod_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    # Any localhost port, not just 3000. Next.js silently falls back to 3001
    # (or higher) when the port is taken, and /explain and /report deliberately
    # bypass the dev proxy to hit this server directly — so a shifted port made
    # their CORS preflight 400 while every proxied endpoint kept working. That
    # failure looks like "the report button is broken", which is expensive to
    # debug mid-demo.
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def health():
    """Health check + API metadata."""
    return {
        "service": "cranovision-api",
        "status": "ok",
        "version": "1.0.0",
        "docs_url": "/docs",
    }