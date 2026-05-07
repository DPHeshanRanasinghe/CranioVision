// CranioVision — typed API client.
// All backend interactions go through here so we have one place to swap the
// backend URL or add auth later.
//
// In dev: rewrites in next.config.js proxy /api/* to localhost:8000.
// In prod: set BACKEND_URL env var.

import type {
  CreateJobResponse,
  JobStatus,
  JobProgressEvent,
  AnalysisResult,
  XaiResult,
  ReportInfo,
  ModelName,
} from './types';

// In Next.js, all backend calls go through /api which is proxied.
const API_BASE = '/api';

async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    // Read body once as text — calling .json() then .text() locks the stream.
    const raw = await res.text().catch(() => '');
    let detail = `HTTP ${res.status}`;
    if (raw) {
      try {
        const body = JSON.parse(raw);
        detail = body.detail || JSON.stringify(body);
      } catch {
        detail = `${detail}: ${raw}`;
      }
    }
    throw new Error(detail);
  }
  return res.json();
}

// -----------------------------------------------------------------------------
// UPLOAD
// -----------------------------------------------------------------------------

/**
 * Upload a case folder. Accepts a list of File objects (the contents of a
 * BraTS folder).
 */
export async function uploadCase(files: File[]): Promise<CreateJobResponse> {
  const fd = new FormData();
  for (const f of files) {
    fd.append('files', f);
  }
  const res = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    body: fd,
  });
  return jsonOrThrow<CreateJobResponse>(res);
}

/**
 * Same as `uploadCase` but reports byte-level progress via `onProgress`.
 * Implemented with XHR because `fetch()` does not expose upload progress.
 * Bytes are the network upload bytes, not server-side parsing — but the
 * server-side parse is fast compared to the multi-hundred-MB POST.
 */
export function uploadCaseWithProgress(
  files: File[],
  onProgress?: (loaded: number, total: number) => void,
): Promise<CreateJobResponse> {
  return new Promise((resolve, reject) => {
    const fd = new FormData();
    for (const f of files) fd.append('files', f);
    const xhr = new XMLHttpRequest();
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) onProgress(e.loaded, e.total);
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as CreateJobResponse);
        } catch {
          reject(new Error('Failed to parse upload response'));
        }
      } else {
        let detail = `HTTP ${xhr.status}`;
        try {
          const body = JSON.parse(xhr.responseText);
          detail = body.detail || JSON.stringify(body);
        } catch {
          if (xhr.responseText) detail = `${detail}: ${xhr.responseText}`;
        }
        reject(new Error(detail));
      }
    };
    xhr.onerror = () => reject(new Error('Network error during upload'));
    xhr.onabort = () => reject(new Error('Upload aborted'));
    xhr.open('POST', `${API_BASE}/upload`);
    xhr.send(fd);
  });
}

// -----------------------------------------------------------------------------
// JOB STATUS / PROGRESS / RESULT
// -----------------------------------------------------------------------------

export async function getJobStatus(jobId: string): Promise<JobStatus> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}/status`);
  return jsonOrThrow<JobStatus>(res);
}

export async function getJobResult(jobId: string): Promise<AnalysisResult> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}/result`);
  return jsonOrThrow<AnalysisResult>(res);
}

/**
 * Subscribe to Server-Sent Events for live progress updates.
 *
 * Usage:
 *   const close = subscribeToProgress(jobId, (event) => {
 *     console.log(event.percent, event.stage, event.message);
 *   });
 *   // call close() to disconnect early
 */
export function subscribeToProgress(
  jobId: string,
  onEvent: (event: JobProgressEvent) => void,
  // onError fires for either transport problems (Event) or pipeline failure
  // (string). Pipeline failures must NOT route through onDone, otherwise the
  // result fetch returns 409 and masks the real backend error.
  onError?: (err: Event | string) => void,
  onDone?: () => void,
): () => void {
  // SSE bypass: connect directly to backend, NOT through Next.js proxy.
  // Next.js dev server buffers streaming responses which kills SSE.
  // Backend has CORS configured so direct connection works.
  const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
  const url = `${backendUrl}/jobs/${jobId}/progress`;
  const source = new EventSource(url);

  // Fallback: if SSE produces no events for 8 seconds, fall back to polling.
  let pollTimer: ReturnType<typeof setInterval> | null = null;
  let receivedAnyEvent = false;
  let terminated = false; // guard against double-firing onDone/onError
  const pollFallbackTimeout = setTimeout(() => {
    if (!receivedAnyEvent) {
      console.warn('[progress] SSE silent — falling back to status polling');
      source.close();
      pollTimer = setInterval(async () => {
        if (terminated) return;
        try {
          const res = await fetch(`/api/jobs/${jobId}/status`);
          if (!res.ok) return;
          const status = await res.json();
          if (status.latest_progress) {
            onEvent(status.latest_progress);
          }
          if (status.state === 'failed') {
            terminated = true;
            if (pollTimer) clearInterval(pollTimer);
            if (onError) onError(status.error || 'Pipeline failed');
            return;
          }
          if (status.state === 'done') {
            terminated = true;
            if (pollTimer) clearInterval(pollTimer);
            if (onDone) onDone();
          }
        } catch (e) {
          console.error('[progress] poll error', e);
        }
      }, 2000);
    }
  }, 8000);

  source.onmessage = (msg) => {
    if (terminated) return;
    receivedAnyEvent = true;
    clearTimeout(pollFallbackTimeout);
    try {
      const event = JSON.parse(msg.data) as JobProgressEvent;
      onEvent(event);
      if (event.stage === 'error') {
        terminated = true;
        source.close();
        if (onError) onError(event.message || 'Pipeline failed');
        return;
      }
      if (event.percent >= 100 || event.stage === 'done') {
        terminated = true;
        source.close();
        if (onDone) onDone();
      }
    } catch (e) {
      console.error('Failed to parse SSE event:', e);
    }
  };

  source.onerror = (err) => {
    if (!receivedAnyEvent) {
      // SSE failed before any events — let the poll fallback handle it
      return;
    }
    if (terminated) return;
    if (onError) onError(err);
    source.close();
  };

  return () => {
    terminated = true;
    clearTimeout(pollFallbackTimeout);
    if (pollTimer) clearInterval(pollTimer);
    source.close();
  };
}

// -----------------------------------------------------------------------------
// XAI
// -----------------------------------------------------------------------------

export async function triggerXai(
  jobId: string,
  modelName: ModelName,
): Promise<XaiResult> {
  // BYPASS the Next.js dev proxy for this call — Grad-CAM compute runs ~30s
  // on GPU and the dev rewrite proxy closes the connection mid-stream,
  // surfacing as a misleading HTTP 500. The await on the cancelled
  // request also kills run_xai's cache write + mesh save (the thread-pool
  // task itself completes, but its result is discarded), so the next click
  // re-runs the whole computation. Same pattern SSE uses.
  // Backend has CORS configured so direct POST works.
  const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
  const res = await fetch(`${backendUrl}/jobs/${jobId}/explain`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_name: modelName }),
  });
  return jsonOrThrow<XaiResult>(res);
}

// -----------------------------------------------------------------------------
// REPORT
// -----------------------------------------------------------------------------

export async function generateReport(
  jobId: string,
  predictionToFeature: ModelName,
): Promise<ReportInfo> {
  // Same Next.js dev-proxy timeout problem as /explain — PDF generation
  // (4 pages, several rendered figures) takes long enough to trip the
  // rewrite proxy and produce ECONNRESET in the browser even when the
  // backend completes the work. Bypass the proxy by hitting the backend
  // directly. CORS already allows localhost:3000.
  const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
  const res = await fetch(`${backendUrl}/jobs/${jobId}/report`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prediction_to_feature: predictionToFeature }),
  });
  return jsonOrThrow<ReportInfo>(res);
}

export function getReportDownloadUrl(jobId: string): string {
  // Direct backend URL so the <a download> link doesn't go through the
  // dev proxy either (PDFs are large and proxy buffering is unreliable).
  const backendUrl =
    typeof window !== 'undefined'
      ? (process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000')
      : '';
  return `${backendUrl}/jobs/${jobId}/report.pdf`;
}

// -----------------------------------------------------------------------------
// VIEWER ARTIFACT URLS
// -----------------------------------------------------------------------------

// NIfTI volumes (used by the legacy Niivue viewer + clinical PDF, kept around
// because the backend still exposes them).
export function getT1cUrl(jobId: string): string {
  return `${API_BASE}/jobs/${jobId}/t1c.nii.gz`;
}

export function getPredictionUrl(jobId: string, modelName: ModelName): string {
  return `${API_BASE}/jobs/${jobId}/predictions/${modelName}.nii.gz`;
}

export function getHeatmapUrl(
  jobId: string,
  modelName: ModelName,
  className: 'edema' | 'enhancing' | 'necrotic',
): string {
  return `${API_BASE}/jobs/${jobId}/xai/${modelName}/${className}.nii.gz`;
}

// GLB meshes (Three.js viewer — kept for back-compat / mesh download).
export function getBrainMeshUrl(jobId: string): string {
  return `${API_BASE}/jobs/${jobId}/meshes/brain.glb`;
}

export function getClassMeshUrl(
  jobId: string,
  modelName: ModelName,
  className: string,
): string {
  return `${API_BASE}/jobs/${jobId}/meshes/${modelName}/${className}.glb`;
}

export function getMeshManifestUrl(jobId: string, modelName: ModelName): string {
  return `${API_BASE}/jobs/${jobId}/meshes/${modelName}/manifest.json`;
}

// Plotly Mesh3d JSON sidecars ({x,y,z,i,j,k,intensity?}). Same geometry as
// the GLBs, just in the shape Plotly wants.
export function getBrainMeshJsonUrl(jobId: string): string {
  return `${API_BASE}/jobs/${jobId}/meshes/brain.json`;
}

export function getClassMeshJsonUrl(
  jobId: string,
  modelName: ModelName,
  className: string,
): string {
  return `${API_BASE}/jobs/${jobId}/meshes/${modelName}/${className}.json`;
}

export function getHeatmapMeshJsonUrl(jobId: string, modelName: ModelName): string {
  return `${API_BASE}/jobs/${jobId}/meshes/${modelName}/heatmap.json`;
}