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
