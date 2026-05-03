# CranioVision — Frontend

Next.js 14 + TypeScript + Tailwind dashboard for CranioVision.

## Setup

```bash
cd frontend
npm install
```

## Run dev server

Make sure the backend is running on port 8000 first:

```bash
# In a separate terminal
cd ../backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Then start the frontend:

```bash
npm run dev
```

Visit http://localhost:3000

## How the proxy works

`next.config.js` rewrites `/api/*` requests to the backend. In dev that's
`localhost:8000`. In production set the `BACKEND_URL` env var.

So `/api/upload` from the browser hits `http://localhost:8000/upload`
on the FastAPI backend. CORS is handled because the browser sees
everything as same-origin.

## Layout

Single page (app/page.tsx) with these components:

- Header with logo + tagline
- UploadCard (drag-and-drop + demo dropdown)
- ProgressCard (progress bar + stage list)
- ModelPicker (4 model tabs)
- ViewerPanel (Niivue 3D viewer)
- MetricsPanel (volumes + Dice scores)
- AnatomyCard (lobe distribution + top regions)
- EloquentCard (6 risk rows + clinical alert)
- AgreementBanner (big stat + download button)

## How the user flow works

1. User drags a folder OR clicks "browse files"
2. Frontend uploads files via POST /api/upload
3. Backend returns job_id, frontend opens an EventSource to /api/jobs/{id}/progress
4. Progress events update the UI in real time
5. When done event arrives, frontend fetches /api/jobs/{id}/result
6. Result populates all the dashboard cards
7. User can switch between 4 model predictions (re-renders viewer)
8. User clicks "+ Grad-CAM" → triggers XAI → heatmap overlay available
9. User clicks "Generate Clinical Report" → PDF generated → "Download PDF" button appears

## Component file map

```
frontend/
├── app/
│   ├── layout.tsx        Root HTML + globals
│   ├── globals.css       Tailwind + custom CSS
│   └── page.tsx          The entire dashboard (single page)
├── components/
│   └── NiivueViewer.tsx  Niivue 3D viewer wrapper
└── lib/
    ├── types.ts          TypeScript types matching backend schemas
    └── api.ts            Typed API client
```

## Production deployment to Vercel

```bash
# From the frontend directory
vercel
```

Set environment variable:
- `BACKEND_URL=https://your-hf-space.hf.space`

Vercel auto-deploys on git push to main if you connect the repo.

## Demo case dropdown

The demo case dropdown currently shows an info message asking the user
to drag the folder. To actually pre-stage demo cases server-side, we'd
add a backend endpoint like POST /demo/{case_id} that loads files from
the dataset path. This is a Phase 4 polish item, not blocking.