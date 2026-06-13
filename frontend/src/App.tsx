import { lazy, Suspense, useEffect, type ComponentType } from "react";
import { Routes, Route, useLocation } from "react-router-dom";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { HelpCircle } from "lucide-react";
import { Sidebar } from "./components/Sidebar";
import { KpiStrip } from "./components/KpiStrip";
import { KpiSkeleton, ViewSkeleton } from "./components/ui";
import { useSnapshot } from "./hooks/useSnapshot";
import { startTour, maybeAutoTour } from "./lib/tour";
import { ALL_ITEMS } from "./nav";
import Placeholder from "./views/Placeholder";

// Code-split every view so the initial bundle stays small and heavy deps
// (Plotly, three.js for the 3D explorer) load only when their view is opened.
const VIEWS: Record<string, ComponentType> = {
  summary: lazy(() => import("./views/Summary")),
  trend: lazy(() => import("./views/CostTrend")),
  threed: lazy(() => import("./views/ThreeD")),
  calendar: lazy(() => import("./views/Calendar")),
  alerts: lazy(() => import("./views/AlertLog")),
  rootcause: lazy(() => import("./views/RootCause")),
  comparison: lazy(() => import("./views/DetectorComparison")),
  incidents: lazy(() => import("./views/Incidents")),
  drift: lazy(() => import("./views/Drift")),
  forecast: lazy(() => import("./views/Forecast")),
  budget: lazy(() => import("./views/Budget")),
  reco: lazy(() => import("./views/Recommendations")),
  playbook: lazy(() => import("./views/Playbook")),
  carbon: lazy(() => import("./views/Carbon")),
  tagging: lazy(() => import("./views/Tagging")),
  ai: lazy(() => import("./views/AIExplain")),
  perf: lazy(() => import("./views/Perf")),
  lab: lazy(() => import("./views/Lab")),
  replay: lazy(() => import("./views/Replay")),
  raw: lazy(() => import("./views/RawData")),
};

function Content() {
  const { data, isLoading, isError, error } = useSnapshot();
  const location = useLocation();
  const reduced = useReducedMotion();

  // Auto-run the guided tour once, after the shell + KPI anchors exist.
  useEffect(() => {
    if (data) maybeAutoTour();
  }, [data]);

  return (
    <main className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-7xl p-6">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">
              Automated Cloud Cost Anomaly Detector
            </h1>
            <p className="text-sm text-muted-foreground">
              Project 13 · Cloud Computing · Spring 2025–2026
            </p>
          </div>
          <button
            onClick={startTour}
            className="flex shrink-0 items-center gap-1.5 rounded-md border border-border bg-card px-3 py-1.5 text-sm font-medium hover:bg-muted"
          >
            <HelpCircle size={15} /> Tour
          </button>
        </div>

        {isLoading && (
          <div className="mt-4 space-y-6">
            <KpiSkeleton />
            <ViewSkeleton />
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
            <div className="mt-4" data-tour="kpis">
              <KpiStrip kpis={data.kpis} />
            </div>
            <div className="mt-6">
              <AnimatePresence mode="wait">
                <motion.div
                  key={location.pathname}
                  initial={reduced ? false : { opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={reduced ? { opacity: 0 } : { opacity: 0, y: -10 }}
                  transition={{ duration: reduced ? 0 : 0.2, ease: "easeOut" }}
                >
                  <Suspense fallback={<ViewSkeleton />}>
                    <Routes location={location}>
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
                  </Suspense>
                </motion.div>
              </AnimatePresence>
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
