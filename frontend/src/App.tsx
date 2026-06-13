import { Routes, Route } from "react-router-dom";
import type { ComponentType } from "react";
import { Sidebar } from "./components/Sidebar";
import { KpiStrip } from "./components/KpiStrip";
import { useSnapshot } from "./hooks/useSnapshot";
import { ALL_ITEMS } from "./nav";
import Summary from "./views/Summary";
import CostTrend from "./views/CostTrend";
import Placeholder from "./views/Placeholder";

// Views migrated so far; the rest fall back to a Placeholder.
const VIEWS: Record<string, ComponentType> = {
  summary: Summary,
  trend: CostTrend,
};

function Content() {
  const { data, isLoading, isError, error } = useSnapshot();
  return (
    <main className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-7xl p-6">
        <h1 className="text-2xl font-bold tracking-tight">
          Automated Cloud Cost Anomaly Detector
        </h1>
        <p className="text-sm text-muted-foreground">
          Project 13 · Cloud Computing · Spring 2025–2026
        </p>

        {isLoading && (
          <div className="mt-8 animate-pulse text-muted-foreground">
            Loading snapshot…
          </div>
        )}
        {isError && (
          <div className="mt-8 rounded-lg border border-high/30 bg-high/10 p-4 text-sm text-high">
            Couldn't reach the API ({String(error)}). Start it with{" "}
            <code>uvicorn cloud_anomaly.api:app --port 8000</code>.
          </div>
        )}

        {data && (
          <>
            <div className="mt-4">
              <KpiStrip kpis={data.kpis} />
            </div>
            <div className="mt-6">
              <Routes>
                {ALL_ITEMS.map((it) => {
                  const Comp = VIEWS[it.key];
                  return (
                    <Route
                      key={it.key}
                      path={it.path}
                      element={Comp ? <Comp /> : <Placeholder name={it.label} />}
                    />
                  );
                })}
              </Routes>
            </div>
          </>
        )}
      </div>
    </main>
  );
}

export default function App() {
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <Content />
    </div>
  );
}
