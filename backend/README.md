# CranioVision — Backend

FastAPI server that wraps the CranioVision ML pipeline (`src/cranovision/`).

> Full first-time setup — including the checkpoints, BraTS cases and atlas
> templates the server needs but the repository cannot ship — is in the root
> [README](../README.md#running-the-system). This file covers the API itself.

## Setup

The backend imports `src.cranovision.*` from the **project root**, so it shares
the root virtual environment rather than having one of its own:

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt          # Linux/macOS: .venv/bin/pip
.venv/Scripts/pip install -r backend/requirements.txt
```

## Run

```bash
cd backend
uvicorn main:app --reload --reload-dir . --reload-dir ../src --port 8000
```

`--reload-dir ../src` matters: without it uvicorn watches only `backend/`, so
edits to the ML pipeline appear to have no effect until you restart by hand.

Swagger UI: <http://localhost:8000/docs>

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Health check |
| GET | `/cache/status` | Redis cache reachability + how many cases are warm |
| POST | `/cache/flush` | Drop every cached result |
| GET | `/demo-cases` | Cases discovered under `data/raw/BraTS2024_small_dataset/` |
| POST | `/demo-cases/{case_id}/run` | Run a local case without uploading it |
| POST | `/upload` | Upload the four BraTS volumes (+ optional `seg`), get a `job_id` |
| GET | `/jobs/{id}/status` | Snapshot of job state |
| GET | `/jobs/{id}/progress` | SSE stream of pipeline progress |
| GET | `/jobs/{id}/result` | Final analysis dict |
| GET | `/jobs/{id}/t1c.nii.gz` | The preprocessed T1c volume |
| GET | `/jobs/{id}/predictions/{model}.nii.gz` | One model's label map |
| GET | `/jobs/{id}/xai/{model}/{class}.nii.gz` | Grad-CAM heatmap volume |
| GET | `/jobs/{id}/meshes/brain.json` | MNI-space brain shell (Atlas view) |
| GET | `/jobs/{id}/meshes/{model}/manifest.json` | Per-class mesh index, MNI space |
| GET | `/jobs/{id}/meshes/{model}/{class}.json` | One tumour-class mesh, MNI space |
| GET | `/jobs/{id}/meshes/native/brain.json` | Patient-space brain shell (Patient view) |
| GET | `/jobs/{id}/meshes/native/{model}/manifest.json` | Per-class mesh index, patient space |
| GET | `/jobs/{id}/meshes/native/{model}/{class}.json` | One tumour-class mesh, patient space |
| POST | `/jobs/{id}/explain` | Trigger Grad-CAM for one prediction (lazy, ~150 s cold) |
| POST | `/jobs/{id}/report` | Generate the clinical PDF |
| GET | `/jobs/{id}/report.pdf` | Download the generated PDF |

**Route order is load-bearing.** `/meshes/native/...` is registered *before*
`/meshes/{model}/{class}.json`; reversed, the parameterised route swallows
`native` as a model name and every Patient-view request 400s.

## Test from the command line

Replace `<CASE>` with a case ID from `GET /demo-cases`.

```bash
# 1. Start a job from a case already on disk (no upload needed)
curl -X POST http://localhost:8000/demo-cases/<CASE>/run

# ...or upload one explicitly, from inside that case's folder
curl -X POST http://localhost:8000/upload \
  -F "files=@<CASE>-t1n.nii" -F "files=@<CASE>-t1c.nii" \
  -F "files=@<CASE>-t2w.nii" -F "files=@<CASE>-t2f.nii" \
  -F "files=@<CASE>-seg.nii"          # seg is optional; enables Dice vs GT
```

```bash
# 2. Watch progress (Server-Sent Events)
curl -N http://localhost:8000/jobs/<JOB_ID>/progress

# 3. Read the result
curl http://localhost:8000/jobs/<JOB_ID>/result | jq .

# 4. Explain one prediction
curl -X POST http://localhost:8000/jobs/<JOB_ID>/explain \
  -H "Content-Type: application/json" -d '{"model_name": "ensemble"}'

# 5. Generate, then download, the clinical PDF
curl -X POST http://localhost:8000/jobs/<JOB_ID>/report \
  -H "Content-Type: application/json" -d '{"prediction_to_feature": "ensemble"}'
curl http://localhost:8000/jobs/<JOB_ID>/report.pdf -o report.pdf
```

## Architecture notes

- **In-memory job state.** Lost on server restart — adequate for a single-user
  demo, and the reason `--reload` firing mid-job kills that job.
- **Single worker thread** for the pipeline. Parallel jobs would compete for GPU.
- **Sequential model loading.** The three models are loaded, run and freed one
  at a time so the whole pipeline fits in 4 GB of VRAM. Do not "optimise" this
  into holding all three resident.
- **Two caches, different jobs.** ANTs registration is cached per case under
  `outputs/atlas_cache/` and verified by SHA-256 of the source volumes; the
  Redis layer (see [DEMO_CACHE.md](DEMO_CACHE.md)) caches whole analysis
  results, keyed by a hash of the scan bytes.
- **The Redis layer fails open.** If Redis is down the backend logs it and runs
  the full analysis — nothing breaks, it just gets slow.
- **CORS** accepts any `localhost`/`127.0.0.1` port, so the frontend falling
  back to `:3001` when `:3000` is taken does not break the report button.

## Deployment

Not yet containerised. For a hosted deployment you would need a `Dockerfile`
plus persistent storage for `models/` and `outputs/atlas_cache/`.
