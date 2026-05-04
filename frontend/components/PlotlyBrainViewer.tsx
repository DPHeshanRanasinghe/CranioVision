'use client';

// Plotly-Mesh3d-backed 3D viewer.
// Replaces the Three.js Brain3DViewer because the user explicitly wanted the
// "rotatable plotly mesh3d viewer from Phase 1" (see CLAUDE.md TL;DR).
//
// Each backend mesh ships as a JSON sidecar in Plotly-Mesh3d shape:
//   { x[], y[], z[], i[], j[], k[], intensity?[] }
// We hand those arrays directly to a `mesh3d` trace.
//
// SSR: this file is mounted via next/dynamic({ ssr: false }) from page.tsx, so
// we can hit `window` and bundle plotly.js-dist-min without breaking the build.

import { useEffect, useMemo, useRef, useState } from 'react';
import dynamic from 'next/dynamic';
import type { ModelName } from '@/lib/types';
import {
  getBrainMeshJsonUrl,
  getClassMeshJsonUrl,
  getMeshManifestUrl,
  getHeatmapMeshJsonUrl,
} from '@/lib/api';

// react-plotly.js needs plotly.js-dist-min injected as `Plotly` (the basic
// distro lacks Mesh3d). We require the whole bundle once so every Plot uses it.
const Plot = dynamic(
  async () => {
    const factory = (await import('react-plotly.js/factory')).default;
    const plotly = await import('plotly.js-dist-min');
    return factory(plotly as any);
  },
  { ssr: false },
) as any;

// ---------------------------------------------------------------------------
// PROPS — kept identical to the Three.js viewer so swapping is a 1-line edit
// ---------------------------------------------------------------------------

interface ViewerProps {
  jobId: string;
  modelName: ModelName;
  showSegmentation: boolean;
  showHeatmap: boolean;
  segmentationOpacity: number;
  heatmapOpacity: number;
  // Heatmap mesh is only fetched once XAI has been generated for the job.
  // XAI is shared (Attention U-Net explains all predictions), so this is a
  // single boolean rather than a per-model flag.
  xaiAvailable?: boolean;
}

// Class colours match the Three.js viewer + the legend in the dashboard.
const CLASS_COLORS: Record<string, string> = {
  edema:     '#FFD23F', // warm yellow
  enhancing: '#FF4848', // saturated red
  necrotic:  '#3F8FFF', // cool blue
};

const BRAIN_SHELL_COLOR   = '#E5E7EB';
const BRAIN_SHELL_OPACITY = 0.10;
const SCENE_BG            = '#1f2937';

interface MeshManifest {
  model: string;
  available_classes: string[];
  brain_bounds: number[][] | null;
}

interface MeshJson {
  x: number[];
  y: number[];
  z: number[];
  i: number[];
  j: number[];
  k: number[];
  intensity?: number[];
}

// ---------------------------------------------------------------------------
// FETCH HELPERS — silent on 404 so the viewer can render partial scenes
// ---------------------------------------------------------------------------

async function fetchMesh(url: string): Promise<MeshJson | null> {
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    return (await res.json()) as MeshJson;
  } catch {
    return null;
  }
}

async function fetchManifest(url: string): Promise<MeshManifest | null> {
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    return (await res.json()) as MeshManifest;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// MAIN VIEWER
// ---------------------------------------------------------------------------

export function PlotlyBrainViewer({
  jobId,
  modelName,
  showSegmentation,
  showHeatmap,
  segmentationOpacity,
  heatmapOpacity,
  xaiAvailable = false,
}: ViewerProps) {
  const [manifest, setManifest] = useState<MeshManifest | null>(null);
  const [brainMesh, setBrainMesh] = useState<MeshJson | null>(null);
  const [classMeshes, setClassMeshes] = useState<Record<string, MeshJson>>({});
  const [heatmapMesh, setHeatmapMesh] = useState<MeshJson | null>(null);
  const [error, setError] = useState<string | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);

  // Brain shell is shared across models — fetch once per job.
  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    fetchMesh(getBrainMeshJsonUrl(jobId)).then((m) => {
      if (!cancelled) setBrainMesh(m);
    });
    return () => {
      cancelled = true;
    };
  }, [jobId]);

  // Manifest + per-class meshes refetch whenever the model tab changes.
  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    setManifest(null);
    setClassMeshes({});
    setError(null);

    (async () => {
      const mf = await fetchManifest(getMeshManifestUrl(jobId, modelName));
      if (cancelled) return;
      if (!mf) {
        setError('Mesh manifest unavailable — backend may still be extracting.');
        return;
      }
      setManifest(mf);

      const entries = await Promise.all(
        mf.available_classes.map(async (cls) => {
          const m = await fetchMesh(getClassMeshJsonUrl(jobId, modelName, cls));
          return [cls, m] as const;
        }),
      );
      if (cancelled) return;
      const populated: Record<string, MeshJson> = {};
      for (const [cls, m] of entries) {
        if (m) populated[cls] = m;
      }
      setClassMeshes(populated);
    })();

    return () => {
      cancelled = true;
    };
  }, [jobId, modelName]);

  // Heatmap fetched once per job: XAI uses Attention U-Net as the shared
  // explainer for all predictions (validated in Phase 3 Week 2 — see
  // CLAUDE.md), so the heatmap mesh is always built against the ensemble
  // and overlaid regardless of which model the user is viewing.
  useEffect(() => {
    if (!jobId || !xaiAvailable) {
      setHeatmapMesh(null);
      return;
    }
    let cancelled = false;
    fetchMesh(getHeatmapMeshJsonUrl(jobId, 'ensemble')).then((m) => {
      if (!cancelled) setHeatmapMesh(m);
    });
    return () => {
      cancelled = true;
    };
  }, [jobId, xaiAvailable]);

  // ------------------------------------------------------------------------
  // PLOTLY TRACE ASSEMBLY
  // ------------------------------------------------------------------------

  const traces = useMemo(() => {
    const out: any[] = [];

    if (brainMesh) {
      out.push({
        type: 'mesh3d',
        x: brainMesh.x, y: brainMesh.y, z: brainMesh.z,
        i: brainMesh.i, j: brainMesh.j, k: brainMesh.k,
        color: BRAIN_SHELL_COLOR,
        opacity: BRAIN_SHELL_OPACITY,
        flatshading: false,
        lighting: { ambient: 0.55, diffuse: 0.6, specular: 0.05, roughness: 0.9 },
        lightposition: { x: 200, y: 250, z: 300 },
        hoverinfo: 'skip',
        showscale: false,
        name: 'Brain',
      });
    }

    // Segmentation and heatmap are independent layers. In XAI mode the
    // segmentation acts as a dimmed backdrop (driven by segmentationOpacity
    // tweened to 0.3 by the parent), and the magma heatmap rides on top.
    if (showSegmentation) {
      for (const cls of Object.keys(classMeshes)) {
        const m = classMeshes[cls];
        out.push({
          type: 'mesh3d',
          x: m.x, y: m.y, z: m.z,
          i: m.i, j: m.j, k: m.k,
          color: CLASS_COLORS[cls] ?? '#cccccc',
          opacity: segmentationOpacity,
          flatshading: false,
          lighting: { ambient: 0.7, diffuse: 0.6, specular: 0.15, roughness: 0.5 },
          lightposition: { x: 200, y: 250, z: 300 },
          showscale: false,
          name: cls,
        });
      }
    }

    const heatmapVisible = showHeatmap && xaiAvailable && heatmapMesh;
    if (heatmapVisible) {
      out.push({
        type: 'mesh3d',
        x: heatmapMesh.x, y: heatmapMesh.y, z: heatmapMesh.z,
        i: heatmapMesh.i, j: heatmapMesh.j, k: heatmapMesh.k,
        intensity: heatmapMesh.intensity ?? undefined,
        intensitymode: 'vertex',
        colorscale: 'Magma',
        cmin: 0,
        cmax: 1,
        opacity: heatmapOpacity,
        flatshading: false,
        lighting: { ambient: 0.7, diffuse: 0.55, specular: 0.1 },
        showscale: true,
        colorbar: {
          title: { text: 'Grad-CAM', font: { color: '#e5e7eb', size: 11 } },
          tickfont: { color: '#e5e7eb', size: 10 },
          thickness: 12,
          len: 0.5,
          x: 1.02,
          bgcolor: 'rgba(0,0,0,0)',
          outlinewidth: 0,
        },
        name: 'Grad-CAM',
      });
    }

    return out;
  }, [
    brainMesh, classMeshes, heatmapMesh,
    showSegmentation, showHeatmap, xaiAvailable,
    segmentationOpacity, heatmapOpacity,
  ]);

  // Camera target = brain centre (from manifest bounds), so the user opens
  // onto a centred orbit instead of a default Plotly view.
  const layout = useMemo(() => {
    const bounds = manifest?.brain_bounds;
    let centre: [number, number, number] = [0, 0, 0];
    if (bounds && bounds.length === 2) {
      const [mn, mx] = bounds;
      centre = [(mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2];
    }
    return {
      autosize: true,
      paper_bgcolor: SCENE_BG,
      plot_bgcolor: SCENE_BG,
      margin: { l: 0, r: 0, t: 0, b: 0 },
      showlegend: false,
      scene: {
        bgcolor: SCENE_BG,
        aspectmode: 'data',
        xaxis: { visible: false, showbackground: false },
        yaxis: { visible: false, showbackground: false },
        zaxis: { visible: false, showbackground: false },
        camera: {
          eye: { x: 1.6, y: 1.4, z: 1.2 },
          center: { x: 0, y: 0, z: 0 },
          up: { x: 0, y: 0, z: 1 },
        },
        // No native "look at point" — we centre the data via aspectmode='data'
        // and rely on the bounds-derived centre via annotations only when
        // useful. The eye vector above gives a clean 3/4 orbit.
      },
    };
  }, [manifest]);

  const config = useMemo(
    () => ({
      displaylogo: false,
      responsive: true,
      modeBarButtonsToRemove: ['toImage', 'sendDataToCloud'],
    }),
    [],
  );

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full bg-slate-800 rounded-lg overflow-hidden"
    >
      <Plot
        data={traces}
        layout={layout}
        config={config}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler
      />

      {/* Class legend — top-left */}
      <div className="absolute top-3 left-3 bg-slate-900/55 backdrop-blur-sm border border-white/10 rounded-md px-3 py-2 text-[11px] text-slate-100 space-y-1 pointer-events-none">
        <div className="font-semibold text-white">Legend</div>
        {manifest?.available_classes.map((cls) => (
          <div key={cls} className="flex items-center gap-2">
            <span
              className="inline-block w-3 h-3 rounded-sm"
              style={{ backgroundColor: CLASS_COLORS[cls] ?? '#cccccc' }}
            />
            <span className="capitalize">{cls}</span>
          </div>
        ))}
        {!manifest && !error && (
          <div className="text-slate-300">Loading meshes…</div>
        )}
        {error && (
          <div className="text-amber-300 max-w-xs">{error}</div>
        )}
      </div>

      <div className="absolute bottom-3 left-3 text-[10px] text-slate-300/80 pointer-events-none">
        Drag to rotate · Scroll to zoom · Right-click to pan
      </div>
    </div>
  );
}

export default PlotlyBrainViewer;
