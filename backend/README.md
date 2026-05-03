# CranioVision — Backend

FastAPI server that wraps the CranioVision ML pipeline.

## Setup

```bash
# From project root
conda activate ml_env_fixed
pip install -r backend/requirements.txt
```

## Run locally

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Visit http://localhost:8000/docs for the auto-generated Swagger UI.

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Health check |
| POST | `/upload` | Upload BraTS folder files, get job_id |
| GET | `/jobs/{id}/status` | Snapshot of job state |
| GET | `/jobs/{id}/progress` | SSE stream of progress updates |
| GET | `/jobs/{id}/result` | Final analysis dict |
| POST | `/jobs/{id}/explain` | Trigger lazy XAI |
| POST | `/jobs/{id}/report` | Generate clinical PDF |
| GET | `/jobs/{id}/report.pdf` | Download generated PDF |

## Test from the command line

### 1. Upload a BraTS case

```bash
cd "D:\2_ML PROJECTS\30. Brainstorm\CranioVision\data\raw\BraTS2024_small_dataset\BraTS-GLI-02143-102"

curl -X POST http://localhost:8000/upload \
  -F "files=@BraTS-GLI-02143-102-t1n.nii" \
  -F "files=@BraTS-GLI-02143-102-t1c.nii" \
  -F "files=@BraTS-GLI-02143-102-t2w.nii" \
  -F "files=@BraTS-GLI-02143-102-t2f.nii" \
  -F "files=@BraTS-GLI-02143-102-seg.nii"
```

Response:
```json
{
  "job_id": "abc-123-...",
  "case_id": "BraTS-GLI-02143-102",
  "cached_registration": true,
  "status": "queued"
}
```

### 2. Watch progress (SSE)

```bash
curl -N http://localhost:8000/jobs/abc-123-.../progress
```

Output streams as the pipeline runs:
```
data: {"stage": "preprocess", "percent": 5, "message": "Loading...", ...}
data: {"stage": "inference", "percent": 10, "message": "Running Attention U-Net...", ...}
...
data: {"stage": "done", "percent": 100, "message": "Analysis complete", ...}
```

### 3. Get the result

```bash
curl http://localhost:8000/jobs/abc-123-.../result | jq .
```

### 4. Trigger XAI for the chosen prediction

```bash
curl -X POST http://localhost:8000/jobs/abc-123-.../explain \
  -H "Content-Type: application/json" \
  -d '{"model_name": "ensemble"}'
```

### 5. Generate + download the clinical PDF

```bash
# Generate
curl -X POST http://localhost:8000/jobs/abc-123-.../report \
  -H "Content-Type: application/json" \
  -d '{"prediction_to_feature": "ensemble"}'

# Download
curl http://localhost:8000/jobs/abc-123-.../report.pdf -o report.pdf
```

## Architecture notes

- **In-memory job state.** Lost on server restart. Adequate for single-user demo.
- **Single worker thread** for the pipeline. Parallel jobs would compete for GPU.
- **Cached registration check.** If the uploaded case_id matches a folder under `outputs/atlas_cache/`, atlas registration is skipped and we reuse the cached transforms (instant). Otherwise ANTs runs from scratch (~3-5 min).
- **Progress SSE** uses standard EventSource format; any frontend can consume it.

## Deployment

For Hugging Face Spaces deployment, add a `Dockerfile` (TODO Phase 4 Week 4)
and configure persistent storage for `outputs/atlas_cache/` and `models/`.