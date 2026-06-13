import { useState } from "react";
import { SlidersHorizontal } from "lucide-react";
import { useSnapshot } from "../hooks/useSnapshot";
import { Card, CardBody, SectionTitle, ModeToggle } from "../components/ui";
import Plot, { PLOT_CONFIG, PLOT_LAYOUT_BASE } from "../lib/plot";
import { PLOT3D_LAYOUT } from "../lib/threed";
import { DETECTOR_LABEL } from "../lib/utils";

function quantile(sorted: number[], q: number): number {
  if (!sorted.length) return 0;
  const i = Math.floor(q * (sorted.length - 1));
  return sorted[i];
}

export default function Lab() {
  const { data } = useSnapshot();
  const [det, setDet] = useState("");
  const [pct, setPct] = useState(0.9);
  const [mode, setMode] = useState<"3d" | "2d">("3d");
  if (!data) return null;

  const detector = det || data.detectors[0];
  const rows = data.detections[detector] ?? [];
  const scores = rows.map((r) => r.score).sort((a, b) => a - b);
  const threshold = quantile(scores, pct);
  const flaggedRows = rows.filter((r) => r.score >= threshold);
  const flaggedDates = [...new Set(flaggedRows.map((r) => r.date))];
  const dateToCost = new Map(data.daily.map((d) => [d.date, d.cost]));
  const services = data.meta.services;

  return (
    <div>
      <SectionTitle
        icon={SlidersHorizontal}
        title="Threshold sensitivity playground"
        subtitle="Re-threshold a detector's scores live. 3D plots every (date × service) point by score; red points clear the threshold. Slide to watch them light up."
      />
      <Card>
        <CardBody>
          <div className="flex flex-wrap items-center gap-4">
            <label className="text-sm">
              <span className="mr-2 text-muted-foreground">Detector</span>
              <select
                value={detector}
                onChange={(e) => setDet(e.target.value)}
                className="rounded-md border border-border bg-card px-2 py-1.5 text-sm"
              >
                {data.detectors.map((d) => (
                  <option key={d} value={d}>{DETECTOR_LABEL[d] ?? d}</option>
                ))}
              </select>
            </label>
            <label className="flex-1 text-sm">
              <span className="text-muted-foreground">Score percentile: {(pct * 100).toFixed(0)}%</span>
              <input
                type="range" min={0.5} max={0.99} step={0.01}
                value={pct}
                onChange={(e) => setPct(Number(e.target.value))}
                className="mt-1 w-full accent-[var(--color-primary)]"
              />
            </label>
            <div className="text-sm">
              <span className="text-muted-foreground">Flagged points: </span>
              <span className="text-lg font-semibold">{flaggedRows.length}</span>
            </div>
            <ModeToggle mode={mode} onChange={setMode} />
          </div>

          <div className="mt-4">
            {mode === "3d" ? (
              <Plot
                data={[{
                  type: "scatter3d",
                  mode: "markers",
                  x: rows.map((r) => r.date),
                  y: rows.map((r) => services.indexOf(r.service)),
                  z: rows.map((r) => r.score),
                  marker: {
                    size: rows.map((r) => (r.score >= threshold ? 5 : 3)),
                    color: rows.map((r) => (r.score >= threshold ? "#dc2626" : "#94a3b8")),
                    opacity: 0.85,
                  },
                  text: rows.map((r) => `${r.service} · ${r.date}: score ${r.score.toFixed(2)}${r.score >= threshold ? " · FLAGGED" : ""}`),
                  hoverinfo: "text",
                }]}
                layout={{
                  ...PLOT3D_LAYOUT,
                  height: 440,
                  scene: {
                    xaxis: { title: "Date" },
                    yaxis: { tickvals: services.map((_, i) => i), ticktext: services, title: "" },
                    zaxis: { title: "Anomaly score" },
                    camera: { eye: { x: 1.7, y: 1.7, z: 1.05 } },
                    aspectmode: "cube",
                  },
                }}
                config={PLOT_CONFIG} useResizeHandler style={{ width: "100%" }}
              />
            ) : (
              <Plot
                data={[
                  { x: data.daily.map((d) => d.date), y: data.daily.map((d) => d.cost), type: "scatter", mode: "lines", name: "Daily cost", line: { color: "#64748b", width: 2 } },
                  { x: flaggedDates, y: flaggedDates.map((d) => dateToCost.get(d) ?? null), type: "scatter", mode: "markers", name: "Flagged", marker: { color: "#dc2626", size: 9, symbol: "circle", line: { width: 1, color: "#1e293b" } } },
                ]}
                layout={{ ...PLOT_LAYOUT_BASE, height: 400, yaxis: { title: "Cost ($)" } }}
                config={PLOT_CONFIG} useResizeHandler style={{ width: "100%" }}
              />
            )}
          </div>
        </CardBody>
      </Card>
    </div>
  );
}
