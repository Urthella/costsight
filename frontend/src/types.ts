// Mirrors the JSON returned by FastAPI's GET /api/snapshot (build_snapshot).
export type Severity = "LOW" | "MEDIUM" | "HIGH";

export interface Meta {
  scenario: string;
  n_days: number;
  seed: number;
  dataset_days: number;
  n_services: number;
  services: string[];
  total_spend: number;
}

export interface Kpis {
  total_spend: number;
  n_services: number;
  flags_per_detector: Record<string, number>;
  total_flags: number;
  consensus_alerts: number;
  best_detector: string | null;
  savable_usd: number;
  leak_ratio: number;
  total_leak_usd: number;
}

export interface DailyPoint {
  date: string;
  cost: number;
}
export interface SeriesPoint {
  date: string;
  service: string;
  cost: number;
}
export interface Detection {
  date: string;
  service: string;
  cost: number;
  score: number;
  is_anomaly: boolean;
}
export interface Alert {
  date: string;
  service: string;
  cost: number;
  score: number;
  severity_score: number;
  severity: Severity;
  detector: string;
}
export interface ComparisonRow {
  detector: string;
  anomaly_type: string;
  precision: number;
  recall: number;
  f1: number;
  tp: number;
  fp: number;
  fn: number;
}
export interface Carbon {
  kg_co2: number;
  km_driven_equiv: number;
  tree_years_equiv: number;
  cost_usd: number;
  by_service: Record<string, unknown>[];
  by_region: Record<string, unknown>[];
}
export interface Recommendation {
  category: string;
  service: string;
  region: string;
  impact_usd_per_month: number;
  confidence: string;
  action: string;
  rationale: string;
}
export interface Tagging {
  debt_usd: number;
  coverage: Record<string, unknown>[];
  worst_services: Record<string, unknown>[];
  policy_yaml: string;
}

export interface Snapshot {
  meta: Meta;
  kpis: Kpis;
  detectors: string[];
  daily: DailyPoint[];
  series: SeriesPoint[];
  detections: Record<string, Detection[]>;
  alerts: Alert[];
  attribution: Record<string, unknown>[];
  comparison: ComparisonRow[];
  carbon: Carbon;
  recommendations: Recommendation[];
  tagging: Tagging;
  drift: Record<string, unknown>[];
  incidents: Record<string, unknown>[];
  forecast: Record<string, unknown>[];
  projected_monthly: Record<string, unknown>[];
  ground_truth: Record<string, unknown>[];
  playbooks: Record<string, unknown>;
}

export interface Scenario {
  key: string;
  description: string;
}

export interface DashboardParams {
  scenario: string;
  nDays: number;
  seed: number;
}
