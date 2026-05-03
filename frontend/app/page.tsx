'use client';

import { useEffect, useState, useCallback } from 'react';
import {
  Brain, Upload, Download, FileText, Loader2, CheckCircle2,
  AlertCircle, Activity,
} from 'lucide-react';
import {
  uploadCase, subscribeToProgress, getJobResult,
  triggerXai, generateReport, getReportDownloadUrl,
} from '@/lib/api';
import type {
  AnalysisResult, JobProgressEvent, ModelName, AtlasResult,
} from '@/lib/types';
import { MODEL_DISPLAY_NAMES } from '@/lib/types';
import { NiivueViewer } from '@/components/NiivueViewer';

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

  // UI state
  const [selectedModel, setSelectedModel] = useState<ModelName>('ensemble');
  const [showSegmentation, setShowSegmentation] = useState(true);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [segOpacity, setSegOpacity] = useState(0.6);
  const [hmOpacity, setHmOpacity] = useState(0.55);
  const [xaiLoading, setXaiLoading] = useState(false);
  const [xaiAvailableForModel, setXaiAvailableForModel] = useState<Set<string>>(new Set());
  const [reportUrl, setReportUrl] = useState<string | null>(null);
  const [reportLoading, setReportLoading] = useState(false);

  // -------------------- upload handlers ----------------------------------

  const handleUpload = useCallback(async (files: File[]) => {
    setError(null);
    setResult(null);
    setProgress(null);
    setReportUrl(null);
    setShowHeatmap(false);
    setXaiAvailableForModel(new Set());

    try {
      const created = await uploadCase(files);
      setJobId(created.job_id);
      setCaseId(created.case_id);
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const files: File[] = [];
      const items = e.dataTransfer.items;

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
          return;
        }
        handleUpload(niiFiles);
      });
    },
    [handleUpload],
  );

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(e.target.files || []).filter(
        (f) => f.name.endsWith('.nii') || f.name.endsWith('.nii.gz'),
      );
      if (files.length > 0) handleUpload(files);
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

  // -------------------- progress subscription ---------------------------

  useEffect(() => {
    if (!jobId) return;
    const close = subscribeToProgress(
      jobId,
      (event) => setProgress(event),
      (err) => setError('Progress stream error'),
      () => {
        // On done, fetch the full result
        getJobResult(jobId)
          .then((r) => setResult(r))
          .catch((e) => setError((e as Error).message));
      },
    );
    return () => close();
  }, [jobId]);

  // -------------------- XAI handler -------------------------------------

  const handleGenerateXai = useCallback(async () => {
    if (!jobId || !selectedModel) return;
    setXaiLoading(true);
    try {
      await triggerXai(jobId, selectedModel);
      setXaiAvailableForModel((prev) => new Set([...prev, selectedModel]));
      setShowHeatmap(true);
    } catch (e) {
      setError(`XAI failed: ${(e as Error).message}`);
    } finally {
      setXaiLoading(false);
    }
  }, [jobId, selectedModel]);

  // -------------------- Report handler ----------------------------------

  const handleGenerateReport = useCallback(async () => {
    if (!jobId) return;
    setReportLoading(true);
    setReportUrl(null);
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
  const isJobRunning = progress && progress.percent < 100 && progress.stage !== 'error';
  const isJobDone = result !== null;

  // ----------------------------------------------------------------------
  // RENDER
  // ----------------------------------------------------------------------

  return (
    <main className="min-h-screen bg-white">
      {/* Header */}
      <header className="border-b border-gray-200 bg-white sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Brain className="w-7 h-7 text-clinical-navy" />
            <div>
              <h1 className="text-xl font-bold text-clinical-navy tracking-tight">
                CranioVision
              </h1>
              <p className="text-xs text-gray-500">
                AI-assisted brain tumor segmentation and clinical analysis
              </p>
            </div>
          </div>
          <div className="text-xs text-gray-500">v1.0.0</div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-6 space-y-6">

        {/* Top row — upload + progress */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <UploadCard
            onDrop={handleDrop}
            onFileSelect={handleFileSelect}
            onDemoSelect={handleDemoCase}
            disabled={isJobRunning}
          />
          <ProgressCard
            progress={progress}
            caseId={caseId}
            isDone={isJobDone}
            error={error}
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
                  xaiAvailable={xaiAvailableForModel.has(selectedModel)}
                  xaiLoading={xaiLoading}
                  onGenerateXai={handleGenerateXai}
                />
              </div>
              <MetricsPanel metrics={currentMetrics} />
            </div>

            {/* Anatomy + eloquent */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <AnatomyCard atlas={atlasForModel} />
              <EloquentCard atlas={atlasForModel} />
            </div>

            {/* Agreement + actions */}
            <AgreementBanner
              agreement={result.agreement}
              onDownloadReport={handleGenerateReport}
              reportUrl={reportUrl}
              reportLoading={reportLoading}
            />
          </>
        )}
      </div>

      <footer className="border-t border-gray-200 mt-12 py-6 text-center text-xs text-gray-500">
        CranioVision · Research use only — not for clinical diagnosis ·
        University of Moratuwa · 2026
      </footer>
    </main>
  );
}

// ============================================================================
// SUB-COMPONENTS
// ============================================================================

function UploadCard({
  onDrop, onFileSelect, onDemoSelect, disabled,
}: {
  onDrop: (e: React.DragEvent) => void;
  onFileSelect: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onDemoSelect: (caseName: string) => void;
  disabled: boolean;
}) {
  const [isDragging, setIsDragging] = useState(false);
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      <h2 className="text-base font-semibold text-clinical-navy mb-3">
        Upload MRI Case
      </h2>
      <div
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => { setIsDragging(false); onDrop(e); }}
        className={`
          border-2 border-dashed rounded-md p-8 text-center cursor-pointer
          transition-colors
          ${isDragging ? 'border-clinical-accent bg-blue-50' : 'border-gray-300 bg-gray-50'}
          ${disabled ? 'opacity-50 pointer-events-none' : 'hover:border-clinical-accent'}
        `}
      >
        <Upload className="w-8 h-8 mx-auto text-gray-400 mb-3" />
        <p className="text-sm font-medium text-gray-700">
          Drop a BraTS case folder here
        </p>
        <p className="text-xs text-gray-500 mt-1 mb-3">
          (4 modalities + optional GT mask, .nii or .nii.gz)
        </p>
        <label className="inline-block">
          <span className="text-xs text-clinical-accent underline cursor-pointer">
            or click to browse files
          </span>
          <input
            type="file"
            multiple
            accept=".nii,.gz"
            onChange={onFileSelect}
            className="hidden"
          />
        </label>
      </div>

      <div className="mt-4 flex items-center gap-2">
        <span className="text-xs text-gray-500">Or demo case:</span>
        <select
          onChange={(e) => onDemoSelect(e.target.value)}
          defaultValue=""
          className="flex-1 text-xs border border-gray-300 rounded px-2 py-1.5"
          disabled={disabled}
        >
          <option value="" disabled>Select a case...</option>
          {DEMO_CASES.map((c) => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
      </div>
    </div>
  );
}

function ProgressCard({
  progress, caseId, isDone, error,
}: {
  progress: JobProgressEvent | null;
  caseId: string | null;
  isDone: boolean;
  error: string | null;
}) {
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      <h2 className="text-base font-semibold text-clinical-navy mb-3">
        Analysis Progress
      </h2>

      {!progress && !error && (
        <div className="text-center text-gray-400 text-sm py-8">
          Awaiting upload
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 rounded p-3 flex items-start gap-2">
          <AlertCircle className="w-4 h-4 text-red-600 flex-shrink-0 mt-0.5" />
          <div className="text-xs text-red-800">{error}</div>
        </div>
      )}

      {progress && !error && (
        <>
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-700">
              {caseId} · {progress.message}
            </span>
            <span className="text-sm font-mono text-clinical-navy tabular-nums">
              {progress.percent}%
            </span>
          </div>
          <div className="w-full h-2 bg-gray-100 rounded overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-clinical-accent to-clinical-navy transition-all"
              style={{ width: `${progress.percent}%` }}
            />
          </div>

          <div className="mt-4 grid grid-cols-4 gap-2 text-xs text-gray-500">
            {STAGE_ORDER.slice(0, -1).map((stage) => {
              const reached = STAGE_ORDER.indexOf(progress.stage) >= STAGE_ORDER.indexOf(stage);
              return (
                <div key={stage} className="flex items-center gap-1">
                  {reached ? (
                    <CheckCircle2 className="w-3 h-3 text-green-600" />
                  ) : progress.stage === stage ? (
                    <Loader2 className="w-3 h-3 animate-spin text-clinical-accent" />
                  ) : (
                    <div className="w-3 h-3 border border-gray-300 rounded-full" />
                  )}
                  <span className={reached ? 'text-gray-700' : ''}>{stage}</span>
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
  const all = [...models, 'ensemble' as ModelName];
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <h2 className="text-base font-semibold text-clinical-navy mb-3">
        Choose Prediction
      </h2>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        {all.map((m) => {
          const data = metrics?.[m];
          const isSelected = selected === m;
          return (
            <button
              key={m}
              onClick={() => onSelect(m)}
              className={`
                p-3 rounded-md border text-left transition
                ${isSelected
                  ? 'border-clinical-accent bg-blue-50'
                  : 'border-gray-200 bg-white hover:border-gray-300'}
              `}
            >
              <div className="text-sm font-medium text-clinical-navy">
                {MODEL_DISPLAY_NAMES[m]}
              </div>
              {data && (
                <div className="text-xs text-gray-500 mt-1 tabular-nums">
                  {data.mean_dice && `Dice ${data.mean_dice.toFixed(3)} · `}
                  {data.volumes_cm3?.['Total tumor']?.toFixed(0) || 0} cm³
                </div>
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
  xaiAvailable, xaiLoading, onGenerateXai,
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
  xaiAvailable: boolean;
  xaiLoading: boolean;
  onGenerateXai: () => void;
}) {
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-base font-semibold text-clinical-navy">
          3D Anatomical Viewer
        </h2>
        <div className="flex gap-2 items-center">
          <label className="flex items-center gap-1 text-xs">
            <input
              type="checkbox"
              checked={showSegmentation}
              onChange={(e) => setShowSegmentation(e.target.checked)}
            />
            Segmentation
          </label>

          {xaiAvailable ? (
            <label className="flex items-center gap-1 text-xs">
              <input
                type="checkbox"
                checked={showHeatmap}
                onChange={(e) => setShowHeatmap(e.target.checked)}
              />
              Grad-CAM
            </label>
          ) : (
            <button
              onClick={onGenerateXai}
              disabled={xaiLoading}
              className="text-xs px-3 py-1 rounded bg-clinical-navy text-white hover:bg-clinical-accent disabled:opacity-50"
            >
              {xaiLoading ? 'Generating…' : '+ Grad-CAM'}
            </button>
          )}
        </div>
      </div>

      <div className="h-[500px] mb-3">
        <NiivueViewer
          jobId={jobId}
          modelName={modelName}
          showSegmentation={showSegmentation}
          showHeatmap={showHeatmap}
          segmentationOpacity={segOpacity}
          heatmapOpacity={hmOpacity}
        />
      </div>

      <div className="grid grid-cols-2 gap-3 text-xs text-gray-600">
        <div>
          Seg opacity:{' '}
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={segOpacity}
            onChange={(e) => setSegOpacity(Number(e.target.value))}
            className="ml-1 w-32 align-middle"
          />
        </div>
        <div>
          Heatmap opacity:{' '}
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={hmOpacity}
            onChange={(e) => setHmOpacity(Number(e.target.value))}
            disabled={!showHeatmap}
            className="ml-1 w-32 align-middle"
          />
        </div>
      </div>
    </div>
  );
}

function MetricsPanel({ metrics }: { metrics?: any }) {
  if (!metrics) return null;
  const v = metrics.volumes_cm3 || {};
  const dice = metrics.brats_regions || {};
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4 space-y-4">
      <div>
        <h3 className="text-sm font-semibold text-clinical-navy mb-2">
          Volume Breakdown
        </h3>
        <dl className="space-y-1.5 text-sm">
          <Row label="Total tumor" value={`${v['Total tumor']?.toFixed(2) || 0} cm³`} highlight />
          <Row label="Edema" value={`${v.Edema?.toFixed(2) || 0} cm³`} />
          <Row label="Enhancing tumor" value={`${v['Enhancing tumor']?.toFixed(2) || 0} cm³`} />
          <Row label="Necrotic core" value={`${v['Necrotic core']?.toFixed(2) || 0} cm³`} />
        </dl>
      </div>

      {metrics.mean_dice !== undefined && (
        <div className="border-t border-gray-100 pt-3">
          <h3 className="text-sm font-semibold text-clinical-navy mb-2">
            Performance
          </h3>
          <dl className="space-y-1.5 text-sm">
            <Row label="Mean Dice" value={metrics.mean_dice.toFixed(4)} highlight />
            <Row label="Whole tumor" value={dice.WT?.toFixed(4) || '–'} />
            <Row label="Tumor core" value={dice.TC?.toFixed(4) || '–'} />
            <Row label="Enhancing" value={dice.ET?.toFixed(4) || '–'} />
          </dl>
        </div>
      )}
    </div>
  );
}

function Row({ label, value, highlight }: { label: string; value: any; highlight?: boolean }) {
  return (
    <div className="flex justify-between">
      <dt className="text-gray-500">{label}</dt>
      <dd className={`tabular-nums ${highlight ? 'font-semibold text-clinical-navy' : 'text-gray-900'}`}>
        {value}
      </dd>
    </div>
  );
}

function AnatomyCard({ atlas }: { atlas?: AtlasResult }) {
  if (!atlas?.anatomy) {
    return (
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <h2 className="text-base font-semibold text-clinical-navy mb-2">
          Anatomical Context
        </h2>
        <p className="text-sm text-gray-500">No anatomical data available.</p>
      </div>
    );
  }
  const a = atlas.anatomy;
  const lobes = Object.entries(a.lobes).sort((x, y) => y[1].voxels - x[1].voxels);
  const regions = a.regions_involved.slice(0, 5);

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <h2 className="text-base font-semibold text-clinical-navy mb-1">
        Anatomical Context
      </h2>
      <p className="text-xs text-gray-500 mb-4">
        Tumor location based on Harvard-Oxford atlas
      </p>

      <div className="mb-4">
        <div className="text-xs text-gray-500 mb-1">Primary location</div>
        <div className="text-base font-semibold text-clinical-navy">
          {a.primary_region} ({a.primary_pct.toFixed(1)}%) · {a.lateralization} hemisphere
        </div>
      </div>

      <div className="space-y-2 mb-4">
        <h4 className="text-xs font-medium text-gray-500 uppercase">Lobe distribution</h4>
        {lobes.map(([name, info]) => (
          <div key={name} className="flex items-center gap-2 text-sm">
            <div className="w-32 text-gray-700">{name}</div>
            <div className="flex-1 h-2 bg-gray-100 rounded overflow-hidden">
              <div
                className="h-full bg-clinical-accent"
                style={{ width: `${info.pct_of_tumor}%` }}
              />
            </div>
            <div className="w-12 text-xs text-right tabular-nums text-gray-600">
              {info.pct_of_tumor.toFixed(1)}%
            </div>
          </div>
        ))}
      </div>

      <div>
        <h4 className="text-xs font-medium text-gray-500 uppercase mb-2">Top regions involved</h4>
        <table className="w-full text-xs">
          <tbody>
            {regions.map(([region, _voxels, pct]) => (
              <tr key={region as string} className="border-t border-gray-100">
                <td className="py-1.5 text-gray-700">{region as string}</td>
                <td className="py-1.5 text-right tabular-nums text-gray-900">
                  {(pct as number).toFixed(1)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function EloquentCard({ atlas }: { atlas?: AtlasResult }) {
  if (!atlas?.eloquent) {
    return (
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <h2 className="text-base font-semibold text-clinical-navy mb-2">
          Eloquent Cortex Risk
        </h2>
        <p className="text-sm text-gray-500">No eloquent data available.</p>
      </div>
    );
  }

  const eloquent = atlas.eloquent;
  const sorted = Object.entries(eloquent).sort(
    (a, b) => (a[1].distance_mm ?? 999) - (b[1].distance_mm ?? 999),
  );
  const highRisk = sorted.filter(([, info]) => info.risk_level === 'high');

  const riskColor: Record<string, string> = {
    high: 'bg-risk-high text-white',
    moderate: 'bg-risk-moderate text-white',
    low: 'bg-risk-low text-gray-900',
    minimal: 'bg-risk-minimal text-white',
  };

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <h2 className="text-base font-semibold text-clinical-navy mb-1">
        Eloquent Cortex Risk
      </h2>
      <p className="text-xs text-gray-500 mb-4">
        Surgical risk based on proximity to functional regions
      </p>

      <div className="space-y-1.5">
        {sorted.map(([name, info]) => (
          <div
            key={name}
            className="flex items-center justify-between gap-2 text-sm py-1.5 border-b border-gray-100 last:border-b-0"
          >
            <div className="flex-1 text-gray-800">{name}</div>
            <div className="text-xs text-gray-600 tabular-nums w-20 text-right">
              {info.involved
                ? 'INVOLVED'
                : info.distance_mm !== null
                ? `${info.distance_mm.toFixed(1)} mm`
                : 'n/a'}
            </div>
            <span
              className={`text-[10px] font-semibold px-2 py-0.5 rounded uppercase ${
                riskColor[info.risk_level] || 'bg-gray-200'
              }`}
            >
              {info.risk_level}
            </span>
          </div>
        ))}
      </div>

      {highRisk.length > 0 && (
        <div className="mt-4 bg-red-50 border border-red-200 rounded p-3 text-xs text-red-900">
          <div className="font-semibold mb-1 flex items-center gap-1">
            <Activity className="w-3 h-3" /> Clinical recommendation
          </div>
          Pre-operative functional MRI and awake-craniotomy planning
          recommended due to involvement of {highRisk.length} eloquent
          region{highRisk.length === 1 ? '' : 's'}.
        </div>
      )}
    </div>
  );
}

function AgreementBanner({
  agreement, onDownloadReport, reportUrl, reportLoading,
}: {
  agreement: { unanimous_fraction: number; n_models_compared: number };
  onDownloadReport: () => void;
  reportUrl: string | null;
  reportLoading: boolean;
}) {
  const pct = (agreement.unanimous_fraction * 100).toFixed(2);
  return (
    <div className="bg-clinical-navy text-white rounded-lg p-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
        <div className="text-center md:text-left">
          <div className="text-5xl font-bold tabular-nums">{pct}%</div>
          <div className="text-sm opacity-90 mt-1">Multi-model unanimous</div>
        </div>
        <div className="text-sm opacity-90 col-span-2 md:col-span-1">
          All {agreement.n_models_compared} architectures agreed on {pct}% of voxels.
          Disagreement concentrates at tumor boundaries — radiologist review
          recommended at edge regions.
        </div>
        <div className="flex flex-col gap-2 md:items-end">
          {reportUrl ? (
            <a
              href={reportUrl}
              download
              className="inline-flex items-center gap-2 bg-white text-clinical-navy px-4 py-2 rounded font-medium text-sm hover:bg-gray-100"
            >
              <Download className="w-4 h-4" />
              Download PDF
            </a>
          ) : (
            <button
              onClick={onDownloadReport}
              disabled={reportLoading}
              className="inline-flex items-center gap-2 bg-white text-clinical-navy px-4 py-2 rounded font-medium text-sm hover:bg-gray-100 disabled:opacity-50"
            >
              <FileText className="w-4 h-4" />
              {reportLoading ? 'Generating…' : 'Generate Clinical Report'}
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