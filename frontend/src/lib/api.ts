import type { Scenario, Snapshot, DashboardParams } from "../types";

// Relative in dev (Vite proxies /api -> :8000). Set VITE_API_URL for prod.
const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? "";

async function getJSON<T>(url: string): Promise<T> {
  const res = await fetch(`${BASE}${url}`);
  if (!res.ok) throw new Error(`${url} → ${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

export function getSnapshot(p: DashboardParams): Promise<Snapshot> {
  const q = new URLSearchParams({
    scenario: p.scenario,
    n_days: String(p.nDays),
    seed: String(p.seed),
  });
  return getJSON<Snapshot>(`/api/snapshot?${q.toString()}`);
}

export async function getScenarios(): Promise<Scenario[]> {
  const data = await getJSON<{ scenarios: Scenario[] }>("/api/scenarios");
  return data.scenarios;
}

export function getPerf(
  p: DashboardParams,
): Promise<{ perf: Record<string, unknown>[] }> {
  const q = new URLSearchParams({
    scenario: p.scenario,
    n_days: String(p.nDays),
    seed: String(p.seed),
  });
  return getJSON(`/api/perf?${q.toString()}`);
}

export interface ExplainBody {
  service: string;
  date: string;
  severity: string;
  cost: number;
  flagged_by: string;
  top_dimension: string;
  top_value: string;
}

export async function postExplain(body: ExplainBody): Promise<string> {
  const res = await fetch(`${BASE}/api/explain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`explain → ${res.status}`);
  const data = (await res.json()) as { explanation: string };
  return data.explanation;
}
