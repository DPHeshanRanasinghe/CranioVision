'use client';

// Niivue 3D viewer wrapper.
// Loads T1c base + segmentation overlay + (optional) Grad-CAM heatmap.
// Keeps the Niivue instance in a ref so we can update layers when the user
// changes models or toggles the heatmap.

import { useEffect, useRef, useState } from 'react';
import { Niivue } from '@niivue/niivue';
import type { ModelName } from '@/lib/types';
import {
  getT1cUrl,
  getPredictionUrl,
  getHeatmapUrl,
} from '@/lib/api';

interface ViewerProps {
  jobId: string;
  modelName: ModelName;
  showSegmentation: boolean;
  showHeatmap: boolean;
  segmentationOpacity: number;     // 0..1
  heatmapOpacity: number;          // 0..1
}

const TUMOR_COLORMAP = {
  // Custom colormap: BG, Edema, Enhancing, Necrotic
  R: [0, 255, 235, 51],
  G: [0, 215, 51, 115],
  B: [0, 0, 51, 217],
  A: [0, 255, 255, 255],
  I: [0, 1, 2, 3],
};

export function NiivueViewer({
  jobId,
  modelName,
  showSegmentation,
  showHeatmap,
  segmentationOpacity,
  heatmapOpacity,
}: ViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const niivueRef = useRef<Niivue | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Initialize Niivue once
  useEffect(() => {
    if (!canvasRef.current) return;

    const nv = new Niivue({
      backColor: [0.06, 0.06, 0.06, 1],   // near-black for medical look
      crosshairColor: [1, 1, 1, 1],
      show3Dcrosshair: false,
      isOrientCube: true,
    });
    nv.attachToCanvas(canvasRef.current);
    nv.setSliceType(nv.sliceTypeMultiplanar);

    // Register custom tumor colormap
    try {
      nv.addColormap('cranio_tumor', TUMOR_COLORMAP);
    } catch (e) {
      // Colormap might already be registered on hot reload — ignore
    }

    niivueRef.current = nv;
  }, []);

  // Load volumes when jobId or modelName changes
  useEffect(() => {
    const nv = niivueRef.current;
    if (!nv || !jobId) return;

    setLoading(true);
    setError(null);

    const volumes: any[] = [
      {
        url: getT1cUrl(jobId),
        colormap: 'gray',
        opacity: 1.0,
      },
    ];

    if (showSegmentation) {
      volumes.push({
        url: getPredictionUrl(jobId, modelName),
        colormap: 'cranio_tumor',
        opacity: segmentationOpacity,
      });
    }

    if (showHeatmap) {
      // Just load the enhancing-tumor heatmap by default
      // (could be extended to layer all 3)
      volumes.push({
        url: getHeatmapUrl(jobId, modelName, 'enhancing'),
        colormap: 'hot',
        opacity: heatmapOpacity,
        cal_min: 0.1,
        cal_max: 1.0,
      });
    }

    nv.loadVolumes(volumes)
      .then(() => setLoading(false))
      .catch((err: Error) => {
        setError(`Failed to load volumes: ${err.message}`);
        setLoading(false);
      });
  }, [jobId, modelName, showSegmentation, showHeatmap]);

  // Update opacities live without reloading
  useEffect(() => {
    const nv = niivueRef.current;
    if (!nv || !nv.volumes || nv.volumes.length === 0) return;

    // Volume[0] is T1c, volume[1] is segmentation if present, etc.
    if (showSegmentation && nv.volumes.length > 1) {
      nv.setOpacity(1, segmentationOpacity);
    }
    if (showHeatmap && nv.volumes.length > 2) {
      nv.setOpacity(2, heatmapOpacity);
    }
    nv.updateGLVolume();
  }, [segmentationOpacity, heatmapOpacity, showSegmentation, showHeatmap]);

  return (
    <div className="relative w-full h-full bg-gray-900 rounded-lg overflow-hidden">
      <canvas ref={canvasRef} className="w-full h-full" />

      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-60">
          <div className="text-white text-sm flex items-center gap-2">
            <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
            Loading volumes...
          </div>
        </div>
      )}

      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-80 px-6">
          <div className="text-red-300 text-sm text-center">
            <div className="font-semibold mb-1">Viewer error</div>
            <div className="text-xs opacity-80">{error}</div>
          </div>
        </div>
      )}
    </div>
  );
}