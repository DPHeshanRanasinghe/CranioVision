// TypeScript types matching the FastAPI backend Pydantic schemas.
// These should stay in sync with backend/schemas/api.py.

export interface CreateJobResponse {
  job_id: string;
  case_id: string;
  cached_registration: boolean;
  status: string;
}

export interface JobProgressEvent {
  stage: string;
  percent: number;
  message: string;
  timestamp: number;
}

export interface JobStatus {
  job_id: string;
  case_id: string;
  state: 'queued' | 'running' | 'done' | 'failed';
  latest_progress?: JobProgressEvent | null;
  error?: string | null;
  elapsed_seconds?: number | null;
}

export interface VolumeBreakdown {
  Edema: number;
  'Enhancing tumor': number;
  'Necrotic core': number;
  'Total tumor': number;
}

export interface DicePerClass {
  Edema?: number;
  'Enhancing tumor'?: number;
  'Necrotic core'?: number;
}

export interface BratsRegions {
  WT?: number;
  TC?: number;
  ET?: number;
}

export interface ModelMetrics {
  volumes_cm3: VolumeBreakdown;
  mean_dice?: number;
  dice_per_class?: DicePerClass;
  brats_regions?: BratsRegions;
}

export interface EloquentRegionInfo {
  distance_mm: number | null;
  risk_level: 'high' | 'moderate' | 'low' | 'minimal' | 'n/a' | 'unknown';
  involved: boolean;
  function: string;
  deficit_if_damaged: string;
}

export interface AnatomyInfo {
  primary_region: string;
  primary_pct: number;
  lateralization: string;
  left_hemisphere_pct: number;
  right_hemisphere_pct: number;
  total_voxels: number;
  total_volume_cm3: number;
  lobes: Record<string, { voxels: number; pct_of_tumor: number }>;
  regions_involved: Array<[string, number, number]>;
}

export interface AtlasResult {
  anatomy?: AnatomyInfo;
  eloquent?: Record<string, EloquentRegionInfo>;
  registration_source?: string;
  shared_across_predictions?: boolean;
  error?: string;
}

export interface AgreementInfo {
  unanimous_fraction: number;
  n_models_compared: number;
}

export interface AnalysisResult {
  case_id: string;
  available_models: string[];
  weights_used: Record<string, number>;
  agreement: AgreementInfo;
  elapsed_seconds: number;
  per_model_metrics: Record<string, ModelMetrics>;
  atlas: Record<string, AtlasResult>;
  prediction_names: string[];
}

export interface XaiResult {
  prediction_being_explained: string;
  explainer_model: string;
  target_layer: string;
  ready: boolean;
  heatmap_urls: {
    edema: string;
    enhancing: string;
    necrotic: string;
  };
}

export interface ReportInfo {
  ready: boolean;
  filename: string;
  size_kb: number;
  download_url: string;
}

// Convenient type for the four model prediction names
export type ModelName = 'attention_unet' | 'swin_unetr' | 'nnunet' | 'ensemble';

export const MODEL_DISPLAY_NAMES: Record<ModelName, string> = {
  attention_unet: 'Attention U-Net',
  swin_unetr: 'SwinUNETR',
  nnunet: 'nnU-Net',
  ensemble: 'Ensemble',
};