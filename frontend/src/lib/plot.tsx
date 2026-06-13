// Build a React Plotly component off the lightweight dist-min bundle (avoids
// pulling the full plotly.js source build and its TS friction).
import type { ComponentType, CSSProperties } from "react";
import factoryImport from "react-plotly.js/factory";
import plotlyImport from "plotly.js-dist-min";

// Both are CommonJS; Vite's interop can hand back a { default } namespace,
// so unwrap defensively before calling the factory.
const createPlotlyComponent = (
  (factoryImport as { default?: unknown }).default ?? factoryImport
) as (plotly: unknown) => unknown;
const Plotly = (plotlyImport as { default?: unknown }).default ?? plotlyImport;

const Plot = createPlotlyComponent(Plotly) as unknown as ComponentType<{
  data: unknown[];
  layout?: Record<string, unknown>;
  config?: Record<string, unknown>;
  style?: CSSProperties;
  className?: string;
  useResizeHandler?: boolean;
}>;

/** Shared Plotly defaults: clean light look, no modebar clutter, responsive. */
export const PLOT_CONFIG = { displayModeBar: false, responsive: true };

export const PLOT_LAYOUT_BASE: Record<string, unknown> = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { family: "Inter, system-ui, sans-serif", color: "#0f172a", size: 12 },
  margin: { t: 30, r: 16, b: 40, l: 56 },
  hovermode: "x unified",
  legend: { orientation: "h", y: -0.2 },
};

export default Plot;
