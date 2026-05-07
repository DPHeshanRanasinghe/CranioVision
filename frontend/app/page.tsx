'use client';

import { useEffect, useState, useCallback, useRef } from 'react';
import {
  Brain, Upload, Download, FileText, Loader2, CheckCircle2,
  AlertCircle, Activity, Sparkles, Info, X, ChevronDown, Play,
  Eye,
} from 'lucide-react';
import {
  uploadCaseWithProgress, subscribeToProgress, getJobResult,
  triggerXai, generateReport, getReportDownloadUrl,
  getHeatmapMeshJsonUrl,
} from '@/lib/api';
import type {
  AnalysisResult, JobProgressEvent, ModelName, AtlasResult,
} from '@/lib/types';
import { MODEL_DISPLAY_NAMES } from '@/lib/types';
import dynamic from 'next/dynamic';

// Plotly Mesh3d viewer is client-only (uses window + bundles plotly.js-dist-min).
// Aliased to `Brain3DViewer` so the rest of the page is unchanged.
const Brain3DViewer = dynamic(
  () => import('@/components/PlotlyBrainViewer').then((m) => m.PlotlyBrainViewer),
  { ssr: false },
);

const DEMO_CASES = [
  'BraTS-GLI-02143-102',
  'BraTS-GLI-02196-105',
  'BraTS-GLI-02105-105',
  'BraTS-GLI-02137-104',
];

const STAGE_ORDER = [
  'init', 'preprocess', 'inference', 'ensemble',
  'metrics', 'agreement', 'atlas', 'done',
];

export default function HomePage() {
  // Job state
  const [jobId, setJobId] = useState<string | null>(null);
  const [caseId, setCaseId] = useState<string | null>(null);
  const [progress, setProgress] = useState<JobProgressEvent | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  // Upload-to-progress feedback states. Without this, the BraTS folder
  // POST + backend parse takes ~10s of silent dead-air between drop and the
  // first SSE event. Each state has dedicated visuals in UploadCard /
  // ProgressCard.
  //   idle       — no job, ready for input
  //   reading    — walking webkitGetAsEntry tree (folder drag) or input change
  //   uploading  — POST in flight, bytes streaming
  //   uploaded   — POST done, brief ✓ confirmation (~1s) before starting
  //   starting   — backend accepted, waiting for first progress event
  //   running    — first progress event arrived, determinate bar live
  //   done       — analysis complete
  //   failed     — pipeline or transport error
  type UploadState =
    | 'idle' | 'reading' | 'uploading' | 'uploaded'
    | 'starting' | 'running' | 'done' | 'failed';
  const [uploadState, setUploadState] = useState<UploadState>('idle');
  const [uploadFileCount, setUploadFileCount] = useState<number>(0);
  const [uploadBytes, setUploadBytes] = useState<{ loaded: number; total: number }>({
    loaded: 0,
    total: 0,
  });

  // UI state
  const [selectedModel, setSelectedModel] = useState<ModelName>('ensemble');
  const [showSegmentation, setShowSegmentation] = useState(true);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [segOpacity, setSegOpacity] = useState(0.6);
  const [hmOpacity, setHmOpacity] = useState(0.55);
  // Radiological (default) keeps the clinical convention. Anatomical
  // mirrors the x axis so patient R appears on viewer R — useful for
  // non-clinical demo audiences. Pure camera flip, no data change.
  const [viewMode, setViewMode] = useState<'radiological' | 'anatomical'>('radiological');
  const [xaiLoading, setXaiLoading] = useState(false);
  // XAI is shared: Attention U-Net is the validated explainer for ALL
  // predictions (Phase 3 Week 2 — see CLAUDE.md). Single boolean, not per-model.
  const [xaiReady, setXaiReady] = useState(false);
  const [reportUrl, setReportUrl] = useState<string | null>(null);
  const [reportLoading, setReportLoading] = useState(false);

  // -------------------- upload handlers ----------------------------------

  const handleUpload = useCallback(async (files: File[]) => {
    setError(null);
    setResult(null);
    setProgress(null);
    setReportUrl(null);
    setShowHeatmap(false);
    setXaiReady(false);
    setUploadFileCount(files.length);
    const totalBytes = files.reduce((acc, f) => acc + f.size, 0);
    setUploadBytes({ loaded: 0, total: totalBytes });
    setUploadState('uploading');

    try {
      const created = await uploadCaseWithProgress(files, (loaded, total) => {
        setUploadBytes({ loaded, total });
      });
      setJobId(created.job_id);
      setCaseId(created.case_id);
      // Brief "Uploaded ✓" confirmation before the ProgressCard takes over.
      setUploadState('uploaded');
      setTimeout(() => {
        // Only step to 'starting' if no progress event has bumped us into
        // 'running' already (rare but possible on cached cases).
        setUploadState((s) => (s === 'uploaded' ? 'starting' : s));
      }, 1000);
    } catch (e) {
      setError((e as Error).message);
      setUploadState('failed');
    }
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const files: File[] = [];
      const items = e.dataTransfer.items;

      setError(null);
      setUploadFileCount(0);
      setUploadState('reading');

      // For folder drag: walk webkitGetAsEntry tree
      const promises: Promise<void>[] = [];
      for (let i = 0; i < items.length; i++) {
        const entry = items[i].webkitGetAsEntry?.();
        if (entry) {
          promises.push(walkEntry(entry, files));
        } else {
          const file = items[i].getAsFile();
          if (file) files.push(file);
        }
      }
      Promise.all(promises).then(() => {
        const niiFiles = files.filter(
          (f) => f.name.endsWith('.nii') || f.name.endsWith('.nii.gz'),
        );
        if (niiFiles.length === 0) {
          setError('No .nii/.nii.gz files found in dropped folder');
          setUploadState('failed');
          return;
        }
        setUploadFileCount(niiFiles.length);
        handleUpload(niiFiles);
      });
    },
    [handleUpload],
  );

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setError(null);
      setUploadState('reading');
      const files = Array.from(e.target.files || []).filter(
        (f) => f.name.endsWith('.nii') || f.name.endsWith('.nii.gz'),
      );
      if (files.length === 0) {
        setError('No .nii/.nii.gz files selected');
        setUploadState('failed');
        return;
      }
      setUploadFileCount(files.length);
      handleUpload(files);
    },
    [handleUpload],
  );

  const handleDemoCase = useCallback(
    async (caseName: string) => {
      // For demo case dropdown, we'd ideally pre-stage the files server-side.
      // For now we ask the user to drag/drop instead, or provide a stub.
      setError(
        `To run "${caseName}", drag the corresponding folder from your local ` +
        `BraTS dataset onto the upload zone. (Server-side demo selection ` +
        `coming in a later iteration.)`,
      );
    },
    [],
  );

  // XAI is shared across models — no per-model gating needed. The heatmap
  // mesh is built once for the ensemble prediction and rendered on top of
  // whichever model the user is viewing.

  // -------------------- progress subscription ---------------------------

  useEffect(() => {
    if (!jobId) return;
    const close = subscribeToProgress(
      jobId,
      (event) => {
        setProgress(event);
        // First real progress event (or any subsequent one) means analysis
        // is actively running. Switch out of 'starting' / 'uploaded'.
        setUploadState((s) =>
          s === 'starting' || s === 'uploaded' || s === 'uploading'
            ? 'running'
            : s,
        );
      },
      (err) => {
        // String = pipeline failure (we want to show the real message).
        // Event = transport error (generic message is fine).
        if (typeof err === 'string') {
          setError(`Pipeline failed: ${err}`);
        } else {
          setError('Progress stream error');
        }
        setUploadState('failed');
      },
      () => {
        // On done, fetch the full result
        getJobResult(jobId)
          .then((r) => {
            setResult(r);
            setUploadState('done');
          })
          .catch((e) => {
            setError((e as Error).message);
            setUploadState('failed');
          });
      },
    );
    return () => close();
  }, [jobId]);

  // -------------------- XAI handler -------------------------------------
  // Generates Grad-CAM ONCE for the ensemble prediction. The explainer is
  // always Attention U-Net regardless of the prediction being viewed
  // (validated in Phase 3 Week 2 — see CLAUDE.md). The resulting heatmap
  // applies to whichever model the user clicks on.

  // Refs cancel any in-flight tween if the user toggles modes mid-fade.
  const tweenIdsRef = useRef<{ seg: number | null; hm: number | null }>({
    seg: null, hm: null,
  });

  const tween = useCallback((
    setter: (v: number) => void,
    from: number,
    to: number,
    durationMs: number,
    slot: 'seg' | 'hm',
  ) => {
    if (tweenIdsRef.current[slot] !== null) {
      cancelAnimationFrame(tweenIdsRef.current[slot]!);
    }
    const start = performance.now();
    const step = (now: number) => {
      const t = Math.min(1, (now - start) / durationMs);
      const eased = 1 - Math.pow(1 - t, 3); // ease-out cubic
      setter(from + (to - from) * eased);
      if (t < 1) {
        tweenIdsRef.current[slot] = requestAnimationFrame(step);
      } else {
        tweenIdsRef.current[slot] = null;
      }
    };
    tweenIdsRef.current[slot] = requestAnimationFrame(step);
  }, []);

  const handleGenerateXai = useCallback(async () => {
    if (!jobId) return;
    setXaiLoading(true);
    setError(null);

    let triggerErr: Error | null = null;
    try {
      await triggerXai(jobId, 'ensemble');
    } catch (e) {
      triggerErr = e as Error;
    }

    // Probe with GET — the backend only registers GET on this route
    // (FastAPI does not auto-add HEAD). Body is small JSON.
    let heatmapAvailable = false;
    try {
      const probe = await fetch(getHeatmapMeshJsonUrl(jobId, 'ensemble'));
      heatmapAvailable = probe.ok;
    } catch {
      heatmapAvailable = false;
    }

    if (heatmapAvailable) {
      // Enter XAI mode: heatmap fades in, segmentation dims to 30% backdrop.
      setHmOpacity(0);
      setXaiReady(true);
      setShowHeatmap(true);
      // Defer one frame so showHeatmap takes effect before tweening.
      requestAnimationFrame(() => {
        tween(setSegOpacity, segOpacity, 0.3, 500, 'seg');
        tween(setHmOpacity, 0, 0.7, 500, 'hm');
      });
    } else if (triggerErr) {
      setError(`XAI failed: ${triggerErr.message}`);
    } else {
      setError('XAI ran but no heatmap mesh was produced.');
    }
    setXaiLoading(false);
  }, [jobId, tween, segOpacity]);

  const handleExitXai = useCallback(() => {
    // Reverse the entrance: heatmap fades out, segmentation back to 60%.
    tween(setHmOpacity, hmOpacity, 0, 400, 'hm');
    tween(setSegOpacity, segOpacity, 0.6, 500, 'seg');
    // After the fade-out completes, drop the heatmap trace entirely.
    setTimeout(() => setShowHeatmap(false), 420);
  }, [tween, hmOpacity, segOpacity]);

  // -------------------- Report handler ----------------------------------

  const handleGenerateReport = useCallback(async () => {
    if (!jobId) return;
    setReportLoading(true);
    setReportUrl(null);
    setError(null);
    try {
      await generateReport(jobId, selectedModel);
      setReportUrl(getReportDownloadUrl(jobId));
    } catch (e) {
      setError(`Report failed: ${(e as Error).message}`);
    } finally {
      setReportLoading(false);
    }
  }, [jobId, selectedModel]);

  // -------------------- derived data ------------------------------------

  const currentMetrics = result?.per_model_metrics?.[selectedModel];
  const atlasForModel: AtlasResult | undefined = result?.atlas?.[selectedModel];
  // Block re-upload while anything is in flight. 'idle' / 'failed' / 'done'
  // are the only safe states for a new drop.
  const uploadBusy =
    uploadState === 'reading' ||
    uploadState === 'uploading' ||
    uploadState === 'uploaded' ||
    uploadState === 'starting' ||
    uploadState === 'running';
  const isJobDone = result !== null;

  // ----------------------------------------------------------------------
  // RENDER
  // ----------------------------------------------------------------------

  return (
    <main className="min-h-screen bg-surface-page">
      {/* Header */}
      <header className="border-b border-border-subtle bg-surface-card/80 backdrop-blur-md sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-3.5 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div
              className="w-9 h-9 rounded-lg flex items-center justify-center shadow-sm"
              style={{
                background:
                  'linear-gradient(135deg, #1D4ED8 0%, #003366 100%)',
              }}
            >
              <Brain className="w-5 h-5 text-white" strokeWidth={2.2} />
            </div>
            <div>
              <h1 className="text-lg font-bold text-text-primary tracking-tight leading-tight">
                CranioVision
              </h1>
              <p className="text-xs text-text-tertiary leading-tight">
                AI-assisted brain tumor segmentation &amp; clinical analysis
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <span className="hidden sm:inline-flex items-center gap-1.5 text-[11px] font-medium text-text-tertiary px-2 py-1 rounded-full bg-surface-muted border border-border-subtle">
              <span className="w-1.5 h-1.5 rounded-full bg-risk-minimal animate-pulse" />
              Research use only
            </span>
            <div className="text-xs font-mono text-text-tertiary tabular-nums">v1.0.0</div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-6 space-y-6">

        {/* Top row — upload + progress */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <UploadCard
            onDrop={handleDrop}
            onFileSelect={handleFileSelect}
            onDemoSelect={handleDemoCase}
            disabled={uploadBusy}
            uploadState={uploadState}
            fileCount={uploadFileCount}
            bytes={uploadBytes}
          />
          <ProgressCard
            progress={progress}
            caseId={caseId}
            isDone={isJobDone}
            error={error}
            uploadState={uploadState}
          />
        </div>

        {/* The rest is shown only after analysis is done */}
        {isJobDone && result && (
          <>
            {/* Model picker */}
            <ModelPicker
              models={result.available_models as ModelName[]}
              selected={selectedModel}
              onSelect={setSelectedModel}
              metrics={result.per_model_metrics}
            />

            {/* Viewer + metrics */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <ViewerPanel
                  jobId={jobId!}
                  modelName={selectedModel}
                  showSegmentation={showSegmentation}
                  showHeatmap={showHeatmap}
                  segOpacity={segOpacity}
                  hmOpacity={hmOpacity}
                  setShowSegmentation={setShowSegmentation}
                  setShowHeatmap={setShowHeatmap}
                  setSegOpacity={setSegOpacity}
                  setHmOpacity={setHmOpacity}
                  xaiReady={xaiReady}
                  onExitXai={handleExitXai}
                  viewMode={viewMode}
                  setViewMode={setViewMode}
                />
              </div>
              <MetricsPanel metrics={currentMetrics} />
            </div>

            {/* Anatomy + eloquent */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <AnatomyCard atlas={atlasForModel} modelName={selectedModel} />
              <EloquentCard atlas={atlasForModel} modelName={selectedModel} />
            </div>

            {/* Agreement + actions */}
            <AgreementBanner
              agreement={result.agreement}
              onDownloadReport={handleGenerateReport}
              reportUrl={reportUrl}
              reportLoading={reportLoading}
              onGenerateXai={handleGenerateXai}
              xaiReady={xaiReady}
              xaiLoading={xaiLoading}
            />
          </>
        )}
      </div>

      <footer className="border-t border-border-subtle mt-16 py-8">
        <div className="max-w-7xl mx-auto px-6 flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-text-tertiary">
          <div>
            CranioVision · <span className="font-medium text-text-secondary">Research use only</span> — not for clinical diagnosis
          </div>
          <div className="font-mono">Heshan Ranasinghe · University of Moratuwa · 2026</div>
        </div>
      </footer>
    </main>
  );
}

// ============================================================================
// SUB-COMPONENTS
// ============================================================================

function UploadCard({
  onDrop, onFileSelect, onDemoSelect, disabled,
  uploadState, fileCount, bytes,
}: {
  onDrop: (e: React.DragEvent) => void;
  onFileSelect: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onDemoSelect: (caseName: string) => void;
  disabled: boolean;
  uploadState:
    | 'idle' | 'reading' | 'uploading' | 'uploaded'
    | 'starting' | 'running' | 'done' | 'failed';
  fileCount: number;
  bytes: { loaded: number; total: number };
}) {
  const [isDragging, setIsDragging] = useState(false);
  const [demoChoice, setDemoChoice] = useState<string>('');
  const showOverlay =
    uploadState === 'reading' ||
    uploadState === 'uploading' ||
    uploadState === 'uploaded';
  const totalMB = (bytes.total / (1024 * 1024)).toFixed(1);
  const loadedMB = (bytes.loaded / (1024 * 1024)).toFixed(1);
  const pct = bytes.total > 0
    ? Math.min(100, Math.round((bytes.loaded / bytes.total) * 100))
    : 0;

  return (
    <div className="bg-surface-card rounded-card border border-border-subtle shadow-sm hover:shadow-md transition-shadow p-6">
      <div className="flex items-center gap-2 mb-4">
        <div className="w-8 h-8 rounded-lg bg-brand-50 flex items-center justify-center">
          <Upload className="w-4 h-4 text-brand-600" />
        </div>
        <h2 className="text-base font-semibold text-text-primary tracking-tight">
          Upload MRI Case
        </h2>
      </div>

      <label
        onDragOver={(e) => { e.preventDefault(); if (!disabled) setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => { setIsDragging(false); if (!disabled) onDrop(e); }}
        className={`
          group relative block border-2 border-dashed rounded-card px-6 py-10
          text-center cursor-pointer transition-all duration-200
          ${isDragging
            ? 'border-brand-500 bg-brand-50 scale-[1.01]'
            : 'border-brand-200 bg-surface-card-hover hover:border-brand-500 hover:bg-brand-50'}
          ${disabled ? 'opacity-60 pointer-events-none' : ''}
        `}
      >
        <input
          type="file"
          multiple
          accept=".nii,.gz"
          onChange={onFileSelect}
          className="hidden"
          disabled={disabled}
        />
        <Upload
          className={`w-12 h-12 mx-auto mb-4 text-brand-500 transition-transform duration-200
            ${isDragging ? 'scale-110' : 'group-hover:scale-105'}
          `}
          strokeWidth={1.5}
        />
        <p className="text-sm font-semibold text-text-primary">
          Drop a BraTS case folder here
        </p>
        <p className="text-xs text-text-tertiary mt-1">
          4 modalities + optional GT mask · .nii or .nii.gz
        </p>
        <p className="mt-4 inline-flex items-center gap-1.5 text-xs font-medium text-brand-600 group-hover:text-brand-700">
          or click to browse files
        </p>

        {/* Live overlay — covers the dropzone during reading / uploading
            / uploaded so the user always sees something change after drop.
            Cross-fades via the parent transition class on the label. */}
        {showOverlay && (
          <div className="absolute inset-0 flex items-center justify-center bg-surface-card/95 backdrop-blur-sm rounded-card transition-opacity duration-300">
            <div className="w-full px-6">
              {uploadState === 'reading' && (
                <div className="flex flex-col items-center gap-3">
                  <Loader2 className="w-7 h-7 animate-spin text-brand-600" />
                  <p className="text-sm font-semibold text-text-primary">
                    Reading {fileCount > 0 ? `${fileCount} files…` : 'files…'}
                  </p>
                  <p className="text-xs text-text-tertiary">
                    Scanning the dropped folder
                  </p>
                </div>
              )}
              {uploadState === 'uploading' && (
                <div className="flex flex-col items-center gap-3">
                  <Loader2 className="w-7 h-7 animate-spin text-brand-600" />
                  <p className="text-sm font-semibold text-text-primary">
                    Uploading {fileCount} file{fileCount === 1 ? '' : 's'}…
                  </p>
                  <p className="text-xs font-mono text-text-tertiary tabular-nums">
                    {bytes.total > 0
                      ? `${loadedMB} / ${totalMB} MB · ${pct}%`
                      : 'Sending…'}
                  </p>
                  <div className="w-full max-w-xs h-1.5 bg-surface-muted rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-brand-500 to-brand-700 rounded-full transition-all duration-200 ease-out"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>
              )}
              {uploadState === 'uploaded' && (
                <div className="flex flex-col items-center gap-2">
                  <div className="w-10 h-10 rounded-full bg-emerald-50 flex items-center justify-center">
                    <CheckCircle2 className="w-6 h-6 text-risk-minimal" />
                  </div>
                  <p className="text-sm font-semibold text-text-primary">
                    Uploaded {fileCount} file{fileCount === 1 ? '' : 's'}
                    {bytes.total > 0 ? ` · ${totalMB} MB` : ''}
                  </p>
                  <p className="text-xs text-text-tertiary">
                    Handing off to the analysis pipeline…
                  </p>
                </div>
              )}
            </div>
          </div>
        )}
      </label>

      <div className="mt-5 pt-5 border-t border-border-subtle">
        <p className="text-xs font-medium text-text-tertiary uppercase tracking-wider mb-2">
          Or run a demo case
        </p>
        <div className="flex items-center gap-2">
          <div className="relative flex-1">
            <select
              value={demoChoice}
              onChange={(e) => setDemoChoice(e.target.value)}
              className="appearance-none w-full text-sm bg-surface-card border border-border-default rounded-lg pl-3 pr-9 py-2 text-text-primary
                focus:outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-100 transition-colors"
              disabled={disabled}
            >
              <option value="" disabled>Select a case…</option>
              {DEMO_CASES.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
            <ChevronDown className="absolute right-2.5 top-1/2 -translate-y-1/2 w-4 h-4 text-text-tertiary pointer-events-none" />
          </div>
          <button
            type="button"
            onClick={() => demoChoice && onDemoSelect(demoChoice)}
            disabled={disabled || !demoChoice}
            className="inline-flex items-center gap-1.5 text-sm font-medium px-3 py-2 rounded-lg
              bg-brand-600 text-white hover:bg-brand-700 transition-colors
              disabled:bg-surface-muted disabled:text-text-muted disabled:cursor-not-allowed"
          >
            <Play className="w-3.5 h-3.5" fill="currentColor" />
            Run
          </button>
        </div>
      </div>
    </div>
  );
}

function ProgressCard({
  progress, caseId, isDone, error, uploadState,
}: {
  progress: JobProgressEvent | null;
  caseId: string | null;
  isDone: boolean;
  error: string | null;
  uploadState:
    | 'idle' | 'reading' | 'uploading' | 'uploaded'
    | 'starting' | 'running' | 'done' | 'failed';
}) {
  // 'starting' = backend has the job but no progress event yet. We show
  // an indeterminate shimmer bar so the user keeps seeing motion.
  const showStarting =
    !error && !progress && (uploadState === 'starting' || uploadState === 'uploaded');
  // Idle = no upload in flight, no result yet, no error.
  const showIdle =
    !error && !progress && !showStarting &&
    uploadState !== 'reading' && uploadState !== 'uploading';
  // While the user is still uploading (no job yet), keep the card quiet —
  // the UploadCard owns the visuals during that phase.
  const showUploadingPlaceholder =
    !error && !progress && (uploadState === 'reading' || uploadState === 'uploading');

  return (
    <div className="bg-surface-card rounded-card border border-border-subtle shadow-sm hover:shadow-md transition-shadow p-6">
      <div className="flex items-center gap-2 mb-4">
        <div className="w-8 h-8 rounded-lg bg-brand-50 flex items-center justify-center">
          <Activity className="w-4 h-4 text-brand-600" />
        </div>
        <h2 className="text-base font-semibold text-text-primary tracking-tight">
          Analysis Progress
        </h2>
        {isDone && !error && (
          <span className="ml-auto inline-flex items-center gap-1 text-xs font-medium text-risk-minimal bg-emerald-50 px-2 py-0.5 rounded-full">
            <CheckCircle2 className="w-3 h-3" />
            Complete
          </span>
        )}
      </div>

      {showIdle && (
        <div className="text-center py-10">
          <div className="w-10 h-10 mx-auto rounded-full bg-surface-muted flex items-center justify-center mb-3">
            <Loader2 className="w-4 h-4 text-text-muted" />
          </div>
          <p className="text-sm text-text-tertiary">Awaiting upload</p>
        </div>
      )}

      {showUploadingPlaceholder && (
        <div className="text-center py-10">
          <div className="w-10 h-10 mx-auto rounded-full bg-brand-50 flex items-center justify-center mb-3">
            <Loader2 className="w-4 h-4 text-brand-600 animate-spin" />
          </div>
          <p className="text-sm text-text-tertiary">
            Receiving files…
          </p>
        </div>
      )}

      {showStarting && (
        <div className="text-center py-8">
          <div className="w-10 h-10 mx-auto rounded-full bg-brand-50 flex items-center justify-center mb-3">
            <Loader2 className="w-4 h-4 text-brand-600 animate-spin" />
          </div>
          <p className="text-sm font-semibold text-text-primary">
            Analysis starting…
          </p>
          <p className="text-xs text-text-tertiary mt-1">
            Warming up GPU and loading models
          </p>
          <div className="mt-5 w-full h-2 bg-surface-muted rounded-full overflow-hidden relative">
            {/* Indeterminate shimmer */}
            <div
              className="absolute inset-y-0 w-1/3 bg-gradient-to-r from-brand-500 to-brand-700 rounded-full"
              style={{ animation: 'cv-indeterminate 1.4s ease-in-out infinite' }}
            />
          </div>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3 flex items-start gap-2">
          <AlertCircle className="w-4 h-4 text-risk-high flex-shrink-0 mt-0.5" />
          <div className="text-xs text-red-800 leading-relaxed">{error}</div>
        </div>
      )}

      {progress && !error && (
        <>
          <div className="flex items-center justify-between mb-2">
            <div className="flex flex-col min-w-0">
              {caseId && (
                <span className="text-xs font-medium text-text-tertiary truncate">
                  {caseId}
                </span>
              )}
              <span className="text-sm text-text-primary truncate">
                {progress.message}
              </span>
            </div>
            <span className="text-sm font-mono font-semibold text-brand-700 tabular-nums ml-3">
              {progress.percent}%
            </span>
          </div>
          <div className="w-full h-2 bg-surface-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-brand-500 to-brand-700 rounded-full transition-all duration-300 ease-out"
              style={{ width: `${progress.percent}%` }}
            />
          </div>

          <div className="mt-5 grid grid-cols-2 sm:grid-cols-4 gap-x-3 gap-y-2 text-xs">
            {STAGE_ORDER.slice(0, -1).map((stage) => {
              const reached = STAGE_ORDER.indexOf(progress.stage) >= STAGE_ORDER.indexOf(stage);
              const active = progress.stage === stage;
              return (
                <div key={stage} className="flex items-center gap-1.5">
                  {reached && !active ? (
                    <CheckCircle2 className="w-3.5 h-3.5 text-risk-minimal flex-shrink-0" />
                  ) : active ? (
                    <Loader2 className="w-3.5 h-3.5 animate-spin text-brand-600 flex-shrink-0" />
                  ) : (
                    <div className="w-3.5 h-3.5 border-2 border-border-default rounded-full flex-shrink-0" />
                  )}
                  <span className={`capitalize ${active ? 'text-brand-700 font-medium' : reached ? 'text-text-secondary' : 'text-text-muted'}`}>
                    {stage}
                  </span>
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}

function ModelPicker({
  models, selected, onSelect, metrics,
}: {
  models: ModelName[];
  selected: ModelName;
  onSelect: (m: ModelName) => void;
  metrics: Record<string, any>;
}) {
  // De-duplicate: only append 'ensemble' if not already present.
  const all = models.includes('ensemble' as ModelName)
    ? models
    : [...models, 'ensemble' as ModelName];

  return (
    <div className="bg-surface-card rounded-card border border-border-subtle shadow-sm p-5">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-base font-semibold text-text-primary tracking-tight">
          Choose Prediction
        </h2>
        <span className="text-xs text-text-tertiary">
          Compare 3 models + ensemble
        </span>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {all.map((m) => {
          const data = metrics?.[m];
          const isSelected = selected === m;
          const isEnsemble = m === 'ensemble';
          return (
            <button
              key={m}
              onClick={() => onSelect(m)}
              className={`
                relative p-4 rounded-card border text-left transition-all duration-200
                ${isSelected
                  ? 'border-brand-500 bg-brand-50 shadow-sm ring-2 ring-brand-100'
                  : 'border-border-subtle bg-surface-card hover:border-brand-300 hover:bg-surface-card-hover hover:shadow-sm'}
              `}
            >
              {isEnsemble && (
                <span className="absolute top-2 right-2 text-[10px] font-semibold uppercase tracking-wider text-brand-700 bg-brand-100 px-1.5 py-0.5 rounded">
                  Best
                </span>
              )}
              <div className={`text-sm font-semibold ${isSelected ? 'text-brand-700' : 'text-text-primary'}`}>
                {MODEL_DISPLAY_NAMES[m]}
              </div>
              {data ? (
                <div className="mt-2 space-y-0.5">
                  {data.mean_dice !== undefined && (
                    <div className="flex items-baseline gap-1">
                      <span className="text-[10px] uppercase tracking-wider text-text-muted">Dice</span>
                      <span className="text-xs font-mono font-semibold text-text-primary tabular-nums">
                        {data.mean_dice.toFixed(3)}
                      </span>
                    </div>
                  )}
                  <div className="flex items-baseline gap-1">
                    <span className="text-[10px] uppercase tracking-wider text-text-muted">Vol</span>
                    <span className="text-xs font-mono text-text-secondary tabular-nums">
                      {data.volumes_cm3?.['Total tumor']?.toFixed(0) || 0} cm³
                    </span>
                  </div>
                </div>
              ) : (
                <div className="mt-2 text-xs text-text-muted">No metrics</div>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function ViewerPanel({
  jobId, modelName, showSegmentation, showHeatmap,
  segOpacity, hmOpacity,
  setShowSegmentation, setShowHeatmap, setSegOpacity, setHmOpacity,
  xaiReady, onExitXai,
  viewMode, setViewMode,
}: {
  jobId: string;
  modelName: ModelName;
  showSegmentation: boolean;
  showHeatmap: boolean;
  segOpacity: number;
  hmOpacity: number;
  setShowSegmentation: (v: boolean) => void;
  setShowHeatmap: (v: boolean) => void;
  setSegOpacity: (v: number) => void;
  setHmOpacity: (v: number) => void;
  xaiReady: boolean;
  onExitXai: () => void;
  viewMode: 'radiological' | 'anatomical';
  setViewMode: (m: 'radiological' | 'anatomical') => void;
}) {
  // Mode is derived from state: XAI mode only when XAI is ready AND the
  // heatmap toggle is on. Switching showHeatmap inside XAI mode flips the
  // toggle's checkbox but the segmentation/heatmap traces stay independent.
  const inXaiMode = xaiReady && showHeatmap;

  return (
    <div className="bg-surface-card rounded-card border border-border-subtle shadow-sm overflow-hidden">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-5 py-3 border-b border-border-subtle bg-surface-card">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-brand-50 flex items-center justify-center">
            <Brain className="w-4 h-4 text-brand-600" />
          </div>
          <h2 className="text-base font-semibold text-text-primary tracking-tight">
            3D Anatomical Viewer
          </h2>
          {inXaiMode && (
            <span className="inline-flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wider px-2 py-1 rounded-full text-white bg-gradient-to-r from-purple-500 to-purple-700 shadow-sm">
              <Sparkles className="w-3 h-3" />
              XAI Mode
            </span>
          )}
        </div>

        <div className="flex gap-1.5 items-center">
          <button
            type="button"
            onClick={() =>
              setViewMode(viewMode === 'radiological' ? 'anatomical' : 'radiological')
            }
            title="Radiological convention is standard in clinical practice. Anatomical view is intuitive for non-clinical users."
            className="inline-flex items-center gap-1.5 text-xs font-medium px-2.5 py-1.5 rounded-lg cursor-pointer transition-colors bg-surface-card text-text-secondary border border-border-subtle hover:border-border-default"
          >
            <Eye className="w-3.5 h-3.5" />
            <span className="capitalize">{viewMode}</span>
          </button>

          <label
            className={`
              inline-flex items-center gap-1.5 text-xs font-medium px-2.5 py-1.5 rounded-lg cursor-pointer transition-colors
              ${showSegmentation
                ? 'bg-brand-50 text-brand-700 border border-brand-200'
                : 'bg-surface-card text-text-tertiary border border-border-subtle hover:border-border-default'}
            `}
          >
            <input
              type="checkbox"
              checked={showSegmentation}
              onChange={(e) => setShowSegmentation(e.target.checked)}
              className="accent-brand-600"
            />
            <span>Segmentation{inXaiMode ? ' (backdrop)' : ''}</span>
          </label>

          {xaiReady ? (
            <label
              className={`
                inline-flex items-center gap-1.5 text-xs font-medium px-2.5 py-1.5 rounded-lg cursor-pointer transition-colors
                ${showHeatmap
                  ? 'bg-purple-50 text-purple-700 border border-purple-200'
                  : 'bg-surface-card text-text-tertiary border border-border-subtle hover:border-border-default'}
              `}
            >
              <input
                type="checkbox"
                checked={showHeatmap}
                onChange={(e) => setShowHeatmap(e.target.checked)}
                className="accent-purple-600"
              />
              Grad-CAM
            </label>
          ) : (
            <span
              className="inline-flex items-center gap-1.5 text-xs font-medium px-2.5 py-1.5 rounded-lg text-text-muted bg-surface-muted border border-border-subtle cursor-not-allowed"
              title="Click 'Generate XAI Explanation' below to enable"
            >
              <input type="checkbox" disabled className="accent-text-muted" />
              Grad-CAM
            </span>
          )}
        </div>
      </div>

      {/* Info banner — only in XAI mode */}
      {inXaiMode && (
        <div className="px-5 py-2.5 bg-purple-50 border-b border-purple-100 flex items-start gap-2">
          <Info className="w-3.5 h-3.5 text-purple-600 flex-shrink-0 mt-0.5" />
          <span className="text-xs text-purple-900 leading-relaxed">
            <strong className="font-semibold">Grad-CAM heatmap</strong> (Attention U-Net explainer, applied to all
            predictions). Switch model tabs to compare each prediction's
            tumor surface against the same attention map.
          </span>
        </div>
      )}

      {/* 3D canvas — premium dark stage with subtle inner glow */}
      <div
        className="h-[500px] relative"
        style={{
          background:
            'radial-gradient(ellipse at center, #1a2332 0%, #0a1018 100%)',
        }}
      >
        <div className="absolute inset-0">
          <Brain3DViewer
            jobId={jobId}
            modelName={modelName}
            showSegmentation={showSegmentation}
            showHeatmap={showHeatmap}
            segmentationOpacity={segOpacity}
            heatmapOpacity={hmOpacity}
            xaiAvailable={xaiReady}
            viewMode={viewMode}
          />
        </div>
      </div>

      {/* Educational orientation caption — persistent, mode-aware, with
          color-matched R / L pills so the user can connect the in-canvas
          labels (red R, blue L) to the convention being used. */}
      <div className="px-5 py-2.5 border-b border-border-subtle bg-surface-card-hover flex items-center justify-between gap-3 flex-wrap">
        <div className="flex items-center gap-2 text-xs text-text-secondary">
          <span
            className="inline-flex items-center justify-center w-5 h-5 rounded-md font-bold text-white text-[11px]"
            style={{ backgroundColor: '#EF4444' }}
            aria-label="R"
          >
            R
          </span>
          <span
            className="inline-flex items-center justify-center w-5 h-5 rounded-md font-bold text-white text-[11px]"
            style={{ backgroundColor: '#3B82F6' }}
            aria-label="L"
          >
            L
          </span>
          {viewMode === 'radiological' ? (
            <span>
              <strong className="font-semibold text-text-primary">Radiological view</strong>
              {' '}— patient&apos;s R hemisphere appears on viewer&apos;s L side.
            </span>
          ) : (
            <span>
              <strong className="font-semibold text-text-primary">Anatomical view</strong>
              {' '}— patient&apos;s R hemisphere appears on viewer&apos;s R side.
            </span>
          )}
        </div>
        <span className="text-[11px] text-text-tertiary italic">
          Toggle convention in the toolbar above
        </span>
      </div>

      {/* Footer controls */}
      <div className="flex items-center justify-between gap-4 px-5 py-3 border-t border-border-subtle bg-surface-card-hover">
        <div className="flex items-center gap-5">
          <SliderControl
            label="Seg opacity"
            value={segOpacity}
            onChange={setSegOpacity}
            disabled={!showSegmentation}
          />
          <SliderControl
            label="Heatmap"
            value={hmOpacity}
            onChange={setHmOpacity}
            disabled={!inXaiMode}
            accentClass="accent-purple-600"
          />
        </div>

        {inXaiMode && (
          <button
            onClick={onExitXai}
            className="inline-flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-lg border border-border-default text-text-secondary bg-surface-card hover:bg-surface-muted hover:border-border-strong transition-colors"
          >
            <X className="w-3.5 h-3.5" />
            Exit XAI Mode
          </button>
        )}
      </div>
    </div>
  );
}

function SliderControl({
  label, value, onChange, disabled, accentClass = 'accent-brand-600',
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  disabled: boolean;
  accentClass?: string;
}) {
  return (
    <div className={`flex items-center gap-2 text-xs ${disabled ? 'opacity-50' : ''}`}>
      <span className="font-medium text-text-secondary">{label}</span>
      <input
        type="range"
        min={0}
        max={1}
        step={0.05}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        disabled={disabled}
        className={`w-24 ${accentClass}`}
      />
      <span className="font-mono text-text-tertiary tabular-nums w-8 text-right">
        {(value * 100).toFixed(0)}%
      </span>
    </div>
  );
}

function MetricsPanel({ metrics }: { metrics?: any }) {
  if (!metrics) return null;
  const v = metrics.volumes_cm3 || {};
  const dice = metrics.brats_regions || {};

  const volumeRows = [
    { label: 'Total tumor', value: v['Total tumor'], highlight: true, dotClass: 'bg-text-primary' },
    { label: 'Edema',           value: v.Edema,             dotClass: 'bg-tumor-edema' },
    { label: 'Enhancing tumor', value: v['Enhancing tumor'], dotClass: 'bg-tumor-enhancing' },
    { label: 'Necrotic core',   value: v['Necrotic core'],   dotClass: 'bg-tumor-necrotic' },
  ];

  return (
    <div className="bg-surface-card rounded-card border border-border-subtle shadow-sm p-5 space-y-5">
      <div>
        <h3 className="text-sm font-semibold text-text-primary tracking-tight mb-3">
          Volume Breakdown
        </h3>
        <dl className="space-y-2 text-sm">
          {volumeRows.map((r) => (
            <MetricRow
              key={r.label}
              label={r.label}
              value={`${r.value?.toFixed(2) || 0} cm³`}
              highlight={r.highlight}
              dotClass={r.dotClass}
            />
          ))}
        </dl>
      </div>

      {metrics.mean_dice !== undefined && (
        <div className="border-t border-border-subtle pt-4">
          <h3 className="text-sm font-semibold text-text-primary tracking-tight mb-3">
            Performance
          </h3>
          <dl className="space-y-2 text-sm">
            <MetricRow label="Mean Dice"   value={metrics.mean_dice.toFixed(4)} highlight />
            <MetricRow label="Whole tumor" value={dice.WT?.toFixed(4) || '–'} />
            <MetricRow label="Tumor core"  value={dice.TC?.toFixed(4) || '–'} />
            <MetricRow label="Enhancing"   value={dice.ET?.toFixed(4) || '–'} />
          </dl>
        </div>
      )}
    </div>
  );
}

function MetricRow({
  label, value, highlight, dotClass,
}: {
  label: string;
  value: any;
  highlight?: boolean;
  dotClass?: string;
}) {
  return (
    <div className="flex items-center justify-between gap-3">
      <dt className="flex items-center gap-2 text-text-secondary">
        {dotClass && <span className={`w-2 h-2 rounded-full ${dotClass}`} />}
        <span>{label}</span>
      </dt>
      <dd className={`font-mono tabular-nums ${highlight ? 'font-semibold text-text-primary' : 'text-text-primary'}`}>
        {value}
      </dd>
    </div>
  );
}

function AnatomyCard({
  atlas,
  modelName,
}: {
  atlas?: AtlasResult;
  modelName: ModelName;
}) {
  const modelLabel = MODEL_DISPLAY_NAMES[modelName];
  if (!atlas?.anatomy) {
    return (
      <div className="bg-surface-card rounded-card border border-border-subtle shadow-sm p-5">
        <h2 className="text-base font-semibold text-text-primary tracking-tight mb-2">
          Anatomical Context
        </h2>
        <p className="text-sm text-text-tertiary">No anatomical data available.</p>
      </div>
    );
  }
  const a = atlas.anatomy;
  const lobes = Object.entries(a.lobes).sort((x, y) => y[1].voxels - x[1].voxels);
  const regions = a.regions_involved.slice(0, 5);
  const lateralColor =
    a.lateralization === 'left'
      ? 'bg-blue-50 text-blue-700 border-blue-200'
      : a.lateralization === 'right'
      ? 'bg-amber-50 text-amber-700 border-amber-200'
      : 'bg-surface-muted text-text-secondary border-border-subtle';

  return (
    <div className="bg-surface-card rounded-card border border-border-subtle shadow-sm p-5">
      <h2 className="text-base font-semibold text-text-primary tracking-tight mb-1">
        Anatomical Context
      </h2>
      <p className="text-xs text-text-tertiary mb-5">
        Tumor location from Harvard-Oxford atlas, computed from{' '}
        <span className="font-medium text-text-secondary">{modelLabel}</span>'s prediction
      </p>

      {/* Primary location — hero block */}
      <div className="mb-5 p-3 rounded-lg bg-brand-50 border border-brand-100">
        <div className="text-[10px] font-semibold uppercase tracking-wider text-brand-700 mb-1">
          Primary Location
        </div>
        <div className="flex items-center justify-between gap-2 flex-wrap">
          <div className="text-sm font-semibold text-text-primary">
            {a.primary_region}
            <span className="ml-2 font-mono text-xs font-normal text-text-secondary tabular-nums">
              {a.primary_pct.toFixed(1)}%
            </span>
          </div>
          <span className={`inline-flex items-center text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full border ${lateralColor}`}>
            {a.lateralization} hemisphere
          </span>
        </div>
      </div>

      {/* Lobe distribution */}
      <div className="mb-5">
        <h4 className="text-[10px] font-semibold uppercase tracking-wider text-text-tertiary mb-2.5">
          Lobe Distribution
        </h4>
        <div className="space-y-2">
          {lobes.map(([name, info]) => (
            <div key={name} className="flex items-center gap-2.5 text-sm">
              <div className="w-28 text-text-secondary truncate">{name}</div>
              <div className="flex-1 h-2 bg-surface-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-brand-500 to-brand-700 rounded-full transition-all"
                  style={{ width: `${info.pct_of_tumor}%` }}
                />
              </div>
              <div className="w-12 text-xs text-right font-mono tabular-nums text-text-primary">
                {info.pct_of_tumor.toFixed(1)}%
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Top regions */}
      <div>
        <h4 className="text-[10px] font-semibold uppercase tracking-wider text-text-tertiary mb-2">
          Top Regions Involved
        </h4>
        <div className="rounded-lg border border-border-subtle overflow-hidden">
          <table className="w-full text-xs">
            <tbody>
              {regions.map(([region, _voxels, pct], i) => (
                <tr
                  key={region as string}
                  className={i > 0 ? 'border-t border-border-subtle' : ''}
                >
                  <td className="py-2 px-3 text-text-secondary">{region as string}</td>
                  <td className="py-2 px-3 text-right font-mono tabular-nums text-text-primary">
                    {(pct as number).toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function EloquentCard({
  atlas,
  modelName,
}: {
  atlas?: AtlasResult;
  modelName: ModelName;
}) {
  const modelLabel = MODEL_DISPLAY_NAMES[modelName];
  if (!atlas?.eloquent) {
    return (
      <div className="bg-surface-card rounded-card border border-border-subtle shadow-sm p-5">
        <h2 className="text-base font-semibold text-text-primary tracking-tight mb-2">
          Eloquent Cortex Risk
        </h2>
        <p className="text-sm text-text-tertiary">No eloquent data available.</p>
      </div>
    );
  }

  const eloquent = atlas.eloquent;
  const sorted = Object.entries(eloquent).sort(
    (a, b) => (a[1].distance_mm ?? 999) - (b[1].distance_mm ?? 999),
  );
  const highRisk = sorted.filter(([, info]) => info.risk_level === 'high');

  // Pill-shaped badges with gradient backgrounds.
  const riskBadge: Record<string, string> = {
    high:     'bg-gradient-to-r from-red-500 to-red-700 text-white shadow-sm',
    moderate: 'bg-gradient-to-r from-amber-400 to-amber-600 text-white shadow-sm',
    low:      'bg-gradient-to-r from-yellow-300 to-yellow-500 text-yellow-900',
    minimal:  'bg-gradient-to-r from-emerald-400 to-emerald-600 text-white shadow-sm',
  };

  return (
    <div className="bg-surface-card rounded-card border border-border-subtle shadow-sm p-5">
      <h2 className="text-base font-semibold text-text-primary tracking-tight mb-1">
        Eloquent Cortex Risk
      </h2>
      <p className="text-xs text-text-tertiary mb-5">
        Surgical risk based on proximity to functional regions, computed from{' '}
        <span className="font-medium text-text-secondary">{modelLabel}</span>'s prediction
      </p>

      <div className="rounded-lg border border-border-subtle overflow-hidden">
        {sorted.map(([name, info], i) => (
          <div
            key={name}
            className={`flex items-center justify-between gap-3 text-sm px-3 py-2.5 ${
              i > 0 ? 'border-t border-border-subtle' : ''
            } ${info.involved ? 'bg-red-50/50' : 'bg-surface-card'}`}
          >
            <div className="flex-1 text-text-primary truncate">{name}</div>
            <div className="text-xs font-mono text-text-secondary tabular-nums w-32 text-right">
              {info.involved
                ? <span className="text-risk-high font-semibold whitespace-nowrap">Edge in/at region</span>
                : info.distance_mm !== null
                ? `${info.distance_mm.toFixed(1)} mm`
                : 'n/a'}
            </div>
            <span
              className={`text-[10px] font-bold px-2.5 py-1 rounded-full uppercase tracking-wider ${
                riskBadge[info.risk_level] || 'bg-surface-muted text-text-secondary'
              }`}
            >
              {info.risk_level}
            </span>
          </div>
        ))}
      </div>

      {highRisk.length > 0 && (
        <div className="mt-4 bg-red-50 border border-red-200 rounded-lg p-3 flex items-start gap-2">
          <Activity className="w-4 h-4 text-risk-high flex-shrink-0 mt-0.5" />
          <div className="text-xs text-red-900 leading-relaxed">
            <div className="font-semibold mb-0.5">Clinical note</div>
            Atlas-based proximity analysis suggests {highRisk.length} eloquent
            region{highRisk.length === 1 ? '' : 's'} at or near the tumor edge.
            Pre-operative fMRI and DTI tractography are strongly recommended to
            verify these atlas-based findings.
          </div>
        </div>
      )}
    </div>
  );
}

function AgreementBanner({
  agreement, onDownloadReport, reportUrl, reportLoading,
  onGenerateXai, xaiReady, xaiLoading,
}: {
  agreement: {
    unanimous_fraction: number;
    tumor_region_agreement: number;
    n_models_compared: number;
  };
  onDownloadReport: () => void;
  reportUrl: string | null;
  reportLoading: boolean;
  onGenerateXai: () => void;
  xaiReady: boolean;
  xaiLoading: boolean;
}) {
  const wholePct = (agreement.unanimous_fraction * 100).toFixed(2);
  const tumorPct = (agreement.tumor_region_agreement * 100).toFixed(1);
  return (
    <div
      className="relative rounded-card p-7 overflow-hidden shadow-lg"
      style={{
        background:
          'linear-gradient(135deg, #003366 0%, #1E3A8A 60%, #1D4ED8 100%)',
      }}
    >
      {/* Subtle decorative glow */}
      <div
        className="absolute -top-24 -right-24 w-64 h-64 rounded-full opacity-20"
        style={{ background: 'radial-gradient(circle, #60A5FA 0%, transparent 70%)' }}
      />

      <div className="relative grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
        {/* Two stacked agreement metrics — whole-volume vs tumor-region */}
        <div className="text-center md:text-left space-y-3">
          <div>
            <div className="text-4xl font-bold tabular-nums text-white tracking-tight leading-none">
              {wholePct}<span className="text-2xl text-white/70">%</span>
            </div>
            <div className="text-[10px] font-semibold text-white/70 mt-1 uppercase tracking-wider">
              Whole-Volume Agreement
            </div>
          </div>
          <div className="pt-3 border-t border-white/15">
            <div className="text-4xl font-bold tabular-nums text-white tracking-tight leading-none">
              {tumorPct}<span className="text-2xl text-white/70">%</span>
            </div>
            <div className="text-[10px] font-semibold text-white/70 mt-1 uppercase tracking-wider">
              Tumor-Region Agreement
            </div>
          </div>
        </div>

        <div className="text-sm text-white/90 leading-relaxed">
          All {agreement.n_models_compared} architectures agreed on{' '}
          <span className="font-semibold text-white">{wholePct}%</span> of all
          voxels — but most are background. Inside the tumor envelope they
          agreed on <span className="font-semibold text-white">{tumorPct}%</span>{' '}
          of voxels; disagreement concentrates at tumor boundaries, so
          radiologist review at edge regions is recommended.
          <div className="mt-3 flex items-start gap-1.5 text-xs text-white/70">
            <Info className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
            <span>
              Grad-CAM uses Attention U-Net as the shared explainer
              (validated empirically).
            </span>
          </div>
        </div>

        <div className="flex flex-col gap-2.5 md:items-end">
          {xaiReady ? (
            <div className="inline-flex items-center gap-2 text-sm text-white/90 px-3 py-2 rounded-lg bg-white/10 border border-white/20 backdrop-blur-sm">
              <CheckCircle2 className="w-4 h-4 text-emerald-300" />
              XAI ready — toggle in viewer
            </div>
          ) : (
            <button
              onClick={onGenerateXai}
              disabled={xaiLoading}
              className="inline-flex items-center gap-2 bg-white/10 hover:bg-white/20 border border-white/30 text-white px-4 py-2.5 rounded-lg font-medium text-sm transition-all disabled:opacity-50 disabled:cursor-not-allowed backdrop-blur-sm shadow-sm"
            >
              {xaiLoading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Sparkles className="w-4 h-4" />
              )}
              {xaiLoading ? 'Generating XAI…' : 'Generate XAI Explanation'}
            </button>
          )}

          {reportUrl ? (
            <a
              href={reportUrl}
              download
              className="inline-flex items-center gap-2 bg-white text-brand-navy hover:bg-brand-50 px-4 py-2.5 rounded-lg font-semibold text-sm transition-colors shadow-md hover:shadow-lg"
            >
              <Download className="w-4 h-4" />
              Download PDF Report
            </a>
          ) : (
            <button
              onClick={onDownloadReport}
              disabled={reportLoading}
              className="inline-flex items-center gap-2 bg-white text-brand-navy hover:bg-brand-50 px-4 py-2.5 rounded-lg font-semibold text-sm transition-colors shadow-md hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {reportLoading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <FileText className="w-4 h-4" />
              )}
              {reportLoading ? 'Generating…' : 'Generate Analysis Report'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

// -------------------- helpers ------------------------------------------

async function walkEntry(entry: any, files: File[]): Promise<void> {
  return new Promise((resolve) => {
    if (entry.isFile) {
      entry.file((file: File) => { files.push(file); resolve(); });
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      reader.readEntries((entries: any[]) => {
        Promise.all(entries.map((e) => walkEntry(e, files))).then(() => resolve());
      });
    } else {
      resolve();
    }
  });
}