import { useState } from "react";
import { Waves } from "lucide-react";
import { useSnapshot } from "../hooks/useSnapshot";
import { Card, CardBody, SectionTitle, ModeToggle } from "../components/ui";
import { DataTable } from "../components/DataTable";
import Plot, { PLOT_CONFIG, PLOT_LAYOUT_BASE } from "../lib/plot";
import { PLOT3D_LAYOUT } from "../lib/threed";

interface DriftRow {
  service: string;
  change_date: string;
  direction: string;
  magnitude_pct: number;
  detector: string;
  confidence: number;
}

// Inspection tools on this chart: the statistic is the point, so give zoom/pan.
const PH_CONFIG = { ...PLOT_CONFIG, displayModeBar: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d"] };

export default function Drift() {
  const { data } = useSnapshot();
  const [mode, setMode] = useState<"3d" | "2d">("3d");
  const [svc, setSvc] = useState("");
  if (!data) return null;
  const drift = data.drift as unknown as DriftRow[];
  const signal = data.drift_signal ?? [];
  const services = data.meta.services;
  const COLOR = (dir: string) => (dir === "up" ? "#d97706" : "#2563eb");

  // Page-Hinkley statistic for one service: pick the first with a drift point.
  const flagCount = (s: string) => signal.filter((r) => r.service === s && r.flag).length;
  const flaggedServices = services.filter((s) => flagCount(s) > 0).sort((a, b) => flagCount(b) - flagCount(a));
  const activeSvc = services.includes(svc) ? svc : flaggedServices[0] ?? services[0] ?? "";
  const sigRows = signal.filter((r) => r.service === activeSvc);
  const threshold = sigRows[0]?.threshold ?? 50;
  const flags = sigRows.filter((r) => r.flag);

  // 2D events overlay: daily total with dashed drift markers.
  const shapes = drift.map((d) => ({
    type: "line", xref: "x", yref: "paper",
    x0: d.change_date, x1: d.change_date, y0: 0, y1: 1,
    line: { color: COLOR(d.direction), width: 1.5, dash: "dash" },
  }));

  return (
    <div>
      <SectionTitle
        icon={Waves}
        title="Concept drift - has the baseline itself shifted?"
        subtitle="Page-Hinkley + ADWIN flag baseline shifts. Below: the detector's own statistic over time, with the threshold and the change-points it fires on."
      />

      {signal.length > 0 && (
        <Card>
          <CardBody>
            <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
              <div className="text-sm text-muted-foreground">
                Page-Hinkley statistic for <span className="font-medium text-foreground">{activeSvc}</span>
                {" - "}
                <span className="font-medium text-foreground">{flags.length}</span> drift point{flags.length === 1 ? "" : "s"}
              </div>
              <select
                value={activeSvc}
                onChange={(e) => setSvc(e.target.value)}
                className="rounded-md border border-border bg-card px-2.5 py-1 text-xs font-medium text-foreground"
              >
                {services.map((s) => (
                  <option key={s} value={s}>
                    {s}{flaggedServices.includes(s) ? "  *" : ""}
                  </option>
                ))}
              </select>
            </div>
            <Plot
              data={[
                {
                  x: sigRows.map((r) => r.date),
                  y: sigRows.map((r) => r.ph_stat),
                  type: "scatter", mode: "lines", name: "PH statistic",
                  line: { color: "#6366f1", width: 2, shape: "spline", smoothing: 0.6 },
                  fill: "tozeroy", fillcolor: "rgba(99,102,241,0.12)",
                  hovertemplate: "PH = %{y:.1f}<extra></extra>",
                },
                {
                  x: flags.map((r) => r.date),
                  y: flags.map((r) => r.ph_stat),
                  type: "scatter", mode: "markers", name: "Drift point",
                  marker: { color: "#dc2626", size: 11, symbol: "diamond", line: { width: 1, color: "#fff" } },
                  hovertemplate: "Drift: %{x}<br>PH = %{y:.1f}<extra></extra>",
                },
              ]}
              layout={{
                ...PLOT_LAYOUT_BASE,
                height: 420,
                margin: { t: 44, r: 16, b: 30, l: 60 },
                legend: { orientation: "h", y: 1.1, x: 0 },
                shapes: [{
                  type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: threshold, y1: threshold,
                  line: { color: "#dc2626", width: 1.5, dash: "dash" },
                }],
                annotations: [{
                  xref: "paper", x: 0.005, xanchor: "left", yref: "y", y: threshold, yshift: 9,
                  text: `threshold ${threshold}`, showarrow: false, font: { size: 11, color: "#dc2626" },
                }],
                yaxis: { title: "Cumulative deviation", rangemode: "tozero" },
                xaxis: { type: "date", rangeslider: { visible: true } },
              }}
              config={PH_CONFIG}
              useResizeHandler
              style={{ width: "100%" }}
            />
            <p className="mt-2 text-xs text-muted-foreground">
              The curve is the cumulative deviation from the running mean. When it crosses the dashed threshold (red diamonds) the baseline has shifted - a drift point - then the statistic resets. Drag the slider under the axis to zoom into a window; use the toolbar to pan or download.
            </p>
          </CardBody>
        </Card>
      )}

      {drift.length ? (
        <>
          <Card className="mt-4">
            <CardBody>
              <div className="mb-2 flex items-center justify-between gap-2">
                <span className="text-sm text-muted-foreground">Drift events (gold = up, blue = down)</span>
                <ModeToggle mode={mode} onChange={setMode} />
              </div>
              {mode === "3d" ? (
                <Plot
                  data={[{
                    type: "scatter3d",
                    mode: "markers",
                    x: drift.map((d) => d.change_date),
                    y: drift.map((d) => services.indexOf(d.service)),
                    z: drift.map((d) => d.magnitude_pct),
                    marker: {
                      size: drift.map((d) => 6 + d.confidence * 8),
                      color: drift.map((d) => COLOR(d.direction)),
                      opacity: 0.9,
                      line: { width: 0.5, color: "#1e293b" },
                    },
                    text: drift.map((d) => `${d.service} · ${d.direction} ${d.magnitude_pct.toFixed(0)}% · ${d.detector} (conf ${d.confidence.toFixed(2)})`),
                    hoverinfo: "text",
                  }]}
                  layout={{
                    ...PLOT3D_LAYOUT,
                    height: 460,
                    scene: {
                      xaxis: { title: "Change date" },
                      yaxis: { tickvals: services.map((_, i) => i), ticktext: services, title: "" },
                      zaxis: { title: "Magnitude %" },
                      camera: { eye: { x: 1.7, y: 1.7, z: 1.05 } },
                      aspectmode: "cube",
                    },
                  }}
                  config={PLOT_CONFIG} useResizeHandler style={{ width: "100%" }}
                />
              ) : (
                <Plot
                  data={[{ x: data.daily.map((d) => d.date), y: data.daily.map((d) => d.cost), type: "scatter", mode: "lines", name: "Daily cost", line: { color: "#64748b", width: 2 } }]}
                  layout={{ ...PLOT_LAYOUT_BASE, height: 360, shapes, yaxis: { title: "Cost ($)" } }}
                  config={PLOT_CONFIG} useResizeHandler style={{ width: "100%" }}
                />
              )}
            </CardBody>
          </Card>
          <Card className="mt-4">
            <CardBody>
              <DataTable
                rows={data.drift}
                columns={[
                  { key: "service", label: "Service" },
                  { key: "change_date", label: "Change date" },
                  { key: "direction", label: "Direction" },
                  { key: "magnitude_pct", label: "Magnitude", align: "right", render: (r) => `${(r.magnitude_pct as number).toFixed(1)}%` },
                  { key: "detector", label: "Detector" },
                  { key: "confidence", label: "Confidence", align: "right", render: (r) => (r.confidence as number).toFixed(2) },
                ]}
              />
            </CardBody>
          </Card>
        </>
      ) : (
        <Card className="mt-4">
          <CardBody>
            <div className="py-8 text-center text-sm text-muted-foreground">No baseline drift events - workload is in a stable regime.</div>
          </CardBody>
        </Card>
      )}
    </div>
  );
}
