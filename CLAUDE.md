# CLAUDE.md — CranioVision Project Context

This file briefs you (Claude Code) on the CranioVision project. Read it before
making any changes. The user (Heshan) has been working on this for months and
the working pieces are valuable — don't break them.

## TL;DR — What You're Doing

The user wants you to:
1. **Replace the broken Niivue 3D viewer with a proper rotatable 3D mesh viewer** (Three.js, marching-cubes meshes generated server-side)
2. **Fix the Grad-CAM heatmap** so it actually shows on the 3D viewer
3. **Merge v0.dev's prettier visual design** into the working frontend (located in `frontend-v0/` if present, else just upgrade visual polish)

Do these in order. Don't start the merge until the 3D viewer is working.

## Project Overview

CranioVision is an AI-assisted brain tumor segmentation web app for clinical
use. Pipeline:
- User uploads BraTS MRI folder (4 modalities + optional GT mask)
- Backend runs 3 deep learning models (Attention U-Net, SwinUNETR, nnU-Net)
  + ensemble + atlas registration + anatomical analysis + eloquent cortex risk
- Frontend shows results: 3D viewer with tumor segmentation, anatomical context,
  surgical risk panel, lazy Grad-CAM XAI, downloadable clinical PDF report

User: DP Heshan Ranasinghe, University of Moratuwa.
Project root: `D:\2_ML PROJECTS\30. Brainstorm\CranioVision`

## Repository Structure

```
CranioVision/
├── src/cranovision/         # Python ML pipeline (Phase 1-3, COMPLETE — DO NOT EDIT)
│   ├── pipeline.py          # Orchestrator: run_full_analysis + compute_xai_for_model
│   ├── inference/           # Models, weighted ensemble, Grad-CAM
│   ├── atlas/               # ANTs registration + Harvard-Oxford anatomy + eloquent
│   └── reporting/           # 4-page clinical PDF generator
├── notebooks/               # Jupyter notebooks (DO NOT EDIT)
├── models/                  # Trained .pth checkpoints
├── outputs/                 # Reports, atlas_cache (30 cached registrations)
├── data/                    # BraTS test cases
├── backend/                 # FastAPI server (Phase 4 Week 1, COMPLETE)
│   ├── main.py
│   ├── api/routes.py        # All HTTP endpoints
│   ├── services/
│   │   ├── job_manager.py   # In-memory async job execution
│   │   └── case_parser.py   # BraTS folder parsing
│   └── schemas/api.py       # Pydantic schemas
├── frontend/                # Next.js 14 + TypeScript (Phase 4 Week 2, MVP working)
│   ├── app/page.tsx         # Single-page dashboard
│   ├── components/NiivueViewer.tsx   # BROKEN — replace this
│   ├── lib/api.ts           # Typed API client
│   └── lib/types.ts         # TypeScript types matching backend schemas
├── frontend-v0/             # v0.dev design output (visual reference only)
└── CLAUDE.md                # This file
```

## What Works Right Now (DO NOT BREAK)

**Backend (FastAPI, port 8000) — fully working:**
- `POST /upload` accepts BraTS folder files, returns job_id
- `GET /jobs/{id}/status` snapshot
- `GET /jobs/{id}/progress` Server-Sent Events stream
- `GET /jobs/{id}/result` analysis JSON
- `POST /jobs/{id}/explain` triggers lazy Grad-CAM
- `POST /jobs/{id}/report` generates clinical PDF
- `GET /jobs/{id}/report.pdf` downloads PDF
- `GET /jobs/{id}/t1c.nii.gz` patient T1c
- `GET /jobs/{id}/predictions/{model}.nii.gz` segmentation masks
- `GET /jobs/{id}/xai/{model}/{class}.nii.gz` Grad-CAM heatmaps

**Frontend (Next.js, port 3000) — working but ugly:**
- Drag-and-drop folder upload works
- SSE+polling fallback for progress (this is critical, do not remove)
- All 4 model predictions display
- XAI explain button works
- PDF report generates and downloads
- `lib/api.ts` and `lib/types.ts` are correct, don't change types

**Niivue viewer — half-broken:**
- Loads NIfTI files correctly
- Shows 3-view multi-planar slices in greyscale
- Tumor segmentation overlay works
- **BUG: Grad-CAM heatmap shows almost nothing** — colormap/threshold issue
- **DESIGN PROBLEM: Not rotatable 3D mesh, just flat 2D slices**

## Critical Architectural Decisions Already Made (Do Not Question)

1. **Atlas Option 1 — shared anatomy.** All 4 predictions share one atlas
   analysis (from GT-mask-based registration). Don't try to register each
   prediction separately. The result has `"shared_across_predictions": true`.
2. **XAI shared explainer.** Attention U-Net is the explainer for ALL
   predictions, regardless of which model the user picks. This was validated
   empirically in Phase 3.
3. **In-memory job state.** State is lost on backend restart. Acceptable for
   single-user demo. Don't add Redis or SQLite.
4. **max_workers=1 in job manager.** GPU can only run one job at a time.
   Don't parallelize.
5. **SSE bypass for dev.** Frontend connects EventSource directly to
   `http://localhost:8000` (bypasses Next.js proxy buffering). With polling
   fallback if SSE silent for 8s. This is in `lib/api.ts:subscribeToProgress`.
   DO NOT remove the polling fallback.

## Phase 4 Week 3 — What You're Building

### Task 1 — Replace Niivue with proper 3D mesh viewer

**Why:** User specifically asked for the "rotatable 3D brain like the plotly
mesh3d interactive viewer from Phase 1." Niivue's MPR slice view is correct
for radiologists but not the wow-factor experience the user wants for demos.

**Architecture (recommended):**
- **Backend:** Add endpoint that extracts marching-cubes meshes from NIfTI
  predictions and serves as `.glb` files (small, fast, web-native)
- **Frontend:** Three.js + react-three-fiber renders the meshes, user rotates
  with mouse, zooms with scroll, toggles per-class visibility

**Backend changes:**
1. Create `backend/services/mesh_extractor.py` that uses `scikit-image`'s
   `marching_cubes` to extract surfaces from prediction NIfTI files. Output
   one mesh per tumor class (edema, enhancing, necrotic) plus a brain shell
   from the T1c. Save as `.glb` files (use `trimesh` library for export).
2. Add endpoints to `backend/api/routes.py`:
   - `GET /jobs/{id}/meshes/brain.glb` — semi-transparent brain shell
   - `GET /jobs/{id}/meshes/{model}/{class}.glb` — per-class tumor mesh
   - `GET /jobs/{id}/meshes/{model}/heatmap.glb` — tumor mesh colored by Grad-CAM
3. In `job_manager.py:_save_viewer_artifacts`, call mesh extraction after
   saving NIfTI files. Cache to disk so repeated requests are instant.
4. Mesh extraction is slow (~5-10 sec per mesh) but only runs ONCE per job.
   Trigger it from `POST /jobs/{id}/explain` for the heatmap mesh
   (it needs the heatmap data first).

**Required new dependencies:**
- `pip install scikit-image trimesh pygltflib`
  (scikit-image is probably already installed)

**Frontend changes:**
1. `npm install three @react-three/fiber @react-three/drei`
2. Replace `components/NiivueViewer.tsx` with a new `Brain3DViewer.tsx`
   that uses react-three-fiber. Load the GLB files via
   `useLoader(GLTFLoader, url)`. Render as `<mesh>` components with
   `<OrbitControls>` from drei.
3. The viewer takes the same props as before:
   `jobId, modelName, showSegmentation, showHeatmap, segmentationOpacity, heatmapOpacity`
4. Visual: dark gradient background, semi-transparent brain shell,
   bright tumor classes (yellow=edema, red=enhancing, blue=necrotic),
   smooth orbital rotation, axis indicator in corner.

### Task 2 — Fix Grad-CAM visualization

**Current bug:** Niivue's hot colormap with `cal_min: 0.1` shows almost
nothing. Heatmap NIfTI files load but visual is empty.

**Fix (after Task 1 is done):**
- Drop the volume overlay approach entirely
- Color the tumor mesh vertices by Grad-CAM intensity at that 3D position
- Use a perceptually-uniform colormap (e.g. viridis or magma)
- Show legend: "Grad-CAM intensity" gradient scale

**How:** When generating the heatmap mesh, sample the heatmap NIfTI at each
mesh vertex position. Use trimesh's `vertex_colors` to bake the colormap
into the GLB. Result: when the mesh loads in three.js, it's already colored
correctly — no separate volume overlay needed.

**Pseudo-code for backend:**
```python
import trimesh
from skimage import measure
from matplotlib import cm

def extract_heatmap_mesh(prediction_path, heatmap_path):
    pred = nib.load(prediction_path).get_fdata()
    heatmap = nib.load(heatmap_path).get_fdata()
    # marching cubes on the union of all tumor classes
    verts, faces, _, _ = measure.marching_cubes(pred > 0, level=0.5)
    # sample heatmap at each vertex
    vertex_intensities = sample_at_positions(heatmap, verts)
    # apply colormap
    colors = cm.magma(vertex_intensities / vertex_intensities.max())
    mesh = trimesh.Trimesh(vertices=verts, faces=faces,
                          vertex_colors=(colors * 255).astype(np.uint8))
    mesh.export("heatmap.glb")
```

### Task 3 — Visual polish merge with v0.dev design

After tasks 1+2 work, do the visual merge. The v0.dev design is in
`frontend-v0/` (if present) or referenced in user's earlier messages.

**Critical constraints:**
- Stay on Next 14 / React 18 / Tailwind 3 (don't upgrade — would break Niivue
  port and add risk)
- Don't import shadcn/ui as a dependency (avoid version conflict hell)
- Keep `lib/api.ts`, `lib/types.ts`, and the new `Brain3DViewer.tsx` unchanged
- Just upgrade visual styling: card layouts, typography, color palette,
  shadows, hover states, transitions

**Visual upgrades user wants:**
- "Look much nicer" — production-grade, not Tailwind-default
- Subtle drop shadows, gradient borders, hover lift effects
- Better typography — use Inter or Geist font
- Animated transitions when results load
- Risk badges with proper rounded-pill shape, gradient backgrounds
- The 3D viewer container should feel "premium" — dark background,
  subtle inner glow, proper toolbar styling
- Generous spacing, clean visual hierarchy

**Reference design language:** Linear.app, Vercel dashboard, Stripe — clean
SaaS aesthetic but with clinical/medical seriousness.

### Task 4 — Demo case dropdown (optional polish)

Currently the demo case dropdown shows an info message. Make it actually work:
1. Add backend endpoint `POST /demo/{case_id}` that loads files from
   `data/raw/BraTS2024_small_dataset/{case_id}/` and starts a job (no upload).
2. Frontend dropdown calls this endpoint when user picks a case.

This is optional. Skip if you're running short on time.

## Testing Workflow

Before pushing changes:

1. Backend test: `cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000`
2. Frontend test: `cd frontend && npm run dev`
3. Open http://localhost:3000
4. Drag the folder
   `data\raw\BraTS2024_small_dataset\BraTS-GLI-02143-102` onto the upload zone
5. Verify:
   - Progress bar fills (cached registration → ~2.5 min)
   - Dashboard populates after completion
   - 3D viewer shows rotatable brain (TASK 1)
   - Click each model tab — viewer updates correctly
   - Click "Generate XAI" — heatmap mesh loads, mesh is colored (TASK 2)
   - Click "Generate Clinical Report" — PDF downloads
6. The whole flow should look impressive (TASK 3)

## Known Constraints

- **GPU is GTX 1650 4GB.** SwinUNETR Grad-CAM is forced to CPU due to OOM.
  This is in `MODEL_DEVICE_HINT` in `notebooks/config_overrides.py` (if
  applicable) but you don't need to touch it.
- **Conda env name: `ml_env_fixed`.** Activate before running backend.
- **Cached registrations exist** for 30 test cases under
  `outputs/atlas_cache/`. Uploads of cached cases skip ANTs and run faster.

## Things That Are Done — Do Not Redo

- ANTs registration + Harvard-Oxford parcellation (Phase 3 Week 1)
- Per-model XAI + shared explainer architecture (Phase 3 Week 2)
- Clinical PDF report (Phase 3 Week 3)
- FastAPI backend with all endpoints (Phase 4 Week 1)
- Drag-and-drop frontend with SSE+polling fallback (Phase 4 Week 2)

## Communication Style For Working With Heshan

- He's strong technically. Don't over-explain basics.
- He values verifying before proceeding. Show him outputs and confirm before
  major steps.
- He's been working long hours on this. Be efficient with his time.
- When making big architectural changes, summarize the plan in 5-10 lines
  before writing code.
- Use git for safety: branch off `dev` for big changes, e.g.
  `git checkout -b feat/3d-mesh-viewer` before starting Task 1.

## Order Of Operations

```
1. git checkout -b feat/3d-mesh-viewer
2. Implement Task 1 (mesh extraction backend + Three.js frontend)
3. Test end-to-end with the BraTS-GLI-02143-102 case
4. Implement Task 2 (Grad-CAM colored mesh)
5. Test heatmap visualization
6. git commit, then implement Task 3 (visual polish merge)
7. Final test
8. git merge to dev
```

## Final Notes

- The user is on Windows. Backend command shells use `^` for line continuation.
- The user's preferred working pattern: small steps, verify each before moving on.
- If you're unsure about an architectural choice, ask the user before coding.
- The pdf report and pipeline.py logic are SACRED. Don't touch them.

Welcome to the project. The hard parts are done. Make it look as good as it
deserves to.