# CranioVision — Frontend

Next.js 14 + TypeScript + Tailwind dashboard for CranioVision.

> Full first-time setup for the whole system is in the root
> [README](../README.md#running-the-system). This file covers the UI only.

## Setup

```bash
cd frontend
npm ci          # `npm install` also works; package-lock.json is committed
```

## Run

The backend must be up first, on port 8000:

```bash
npm run dev
```

Visit <http://localhost:3000>. If 3000 is taken Next will use 3001 — that is
fine, the backend accepts any `localhost` port.

## How the proxy works

`next.config.js` rewrites `/api/*` to the backend — `localhost:8000` in dev, or
`BACKEND_URL` in production. So `/api/upload` in the browser reaches
`http://localhost:8000/upload`, and the browser sees everything as same-origin.

## Layout

Single page (`app/page.tsx`) composed of:

- Header with logo + tagline
- Upload card — drag-and-drop, plus a dropdown of cases found on the server
- Progress card — progress bar and per-stage list, fed by SSE
- Model picker — Attention U-Net / SwinUNETR / nnU-Net / Ensemble
- Viewer panel — `PlotlyBrainViewer`, with an **Atlas ↔ Patient** space toggle
- Metrics panel — per-region volumes, and Dice vs ground truth when a `seg`
  volume was supplied
- Anatomy card — lobe distribution and top Harvard-Oxford regions
- Eloquent card — eloquent-cortex proximity rows and the surgical-risk alert
- Agreement banner — cross-model agreement, and the report download

## User flow

1. Drag a BraTS case folder, or pick one from the demo dropdown
   (`GET /api/demo-cases` → `POST /api/demo-cases/{id}/run`).
2. Backend returns a `job_id`; the page opens an `EventSource` on
   `/api/jobs/{id}/progress`.
3. Progress events drive the UI live.
4. On `done`, the page fetches `/api/jobs/{id}/result` and populates every card.
5. Switching model tabs re-fetches that model's meshes — including its own
   atlas and eloquent analysis, which are computed per model, not shared.
6. "+ Grad-CAM" triggers XAI lazily (~150 s the first time on a cold case).
7. "Generate Clinical Report" produces the PDF, then offers the download.

## Two viewer spaces

The toggle is not cosmetic — the two modes are different coordinate systems:

- **Atlas** — the prediction warped into MNI152 space. Comparable across
  patients; this is what the anatomical labelling is computed in.
- **Patient** — the tumour on that patient's own brain, reconstructed from
  their T1c. This is the one a surgeon actually recognises.

Both are served in **world (RAS+) coordinates**, so left is left in both. Mesh
axis labels come from `WORLD_X_AXIS_LABELS`, never from voxel index order —
using the voxel-index labels on a world-space mesh mirrors the tumour, which is
a patient-safety bug, not a rendering nit.

Camera state (rotation, zoom, pan) is preserved across the toggle and across
model switches: the mesh data is swapped atomically and the camera restored
imperatively, so the view never snaps back to default mid-inspection.

## File map

```
frontend/
├── app/
│   ├── layout.tsx              Root HTML + globals
│   ├── globals.css             Tailwind + custom CSS
│   └── page.tsx                The entire dashboard
├── components/
│   └── PlotlyBrainViewer.tsx   Plotly 3D mesh viewer (brain shell + tumour)
└── lib/
    ├── types.ts                Types mirroring the backend schemas
    └── api.ts                  Typed API client — all calls go through here
```

## Production build

```bash
npm run build && npm start
```

For a hosted deployment set `BACKEND_URL` to the backend's public URL. Note
the backend is not containerised yet, so there is no one-command deploy.
