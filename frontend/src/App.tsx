import { Routes, Route } from "react-router-dom";
import type { ComponentType } from "react";
import { Sidebar } from "./components/Sidebar";
import { KpiStrip } from "./components/KpiStrip";
import { useSnapshot } from "./hooks/useSnapshot";
import { ALL_ITEMS } from "./nav";
import Summary from "./views/Summary";
import CostTrend from "./views/CostTrend";
import Calendar from "./views/Calendar";
import AlertLog from "./views/AlertLog";
import RootCause from "./views/RootCause";
import DetectorComparison from "./views/DetectorComparison";
import Incidents from "./views/Incidents";
import Drift from "./views/Drift";
import Forecast from "./views/Forecast";
import Budget from "./views/Budget";
import Recommendations from "./views/Recommendations";
import Playbook from "./views/Playbook";
import Carbon from "./views/Carbon";
import Tagging from "./views/Tagging";
import AIExplain from "./views/AIExplain";
import Perf from "./views/Perf";
import Lab from "./views/Lab";
import Replay from "./views/Replay";
import RawData from "./views/RawData";
import Placeholder from "./views/Placeholder";

const VIEWS: Record<string, ComponentType> = {
  summary: Summary,
  trend: CostTrend,
  calendar: Calendar,
  alerts: AlertLog,
  rootcause: RootCause,
  comparison: DetectorComparison,
  incidents: Incidents,
  drift: Drift,
  forecast: Forecast,
  budget: Budget,
  reco: Recommendations,
  playbook: Playbook,
  carbon: Carbon,
  tagging: Tagging,
  ai: AIExplain,
  perf: Perf,
  lab: Lab,
  replay: Replay,
  raw: RawData,
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
