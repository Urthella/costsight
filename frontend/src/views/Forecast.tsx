import { useState } from "react";
import { TrendingUp } from "lucide-react";
import { useSnapshot } from "../hooks/useSnapshot";
import { Card, CardBody, SectionTitle, ModeToggle } from "../components/ui";
import { DataTable } from "../components/DataTable";
import Plot, { PLOT_CONFIG, PLOT_LAYOUT_BASE } from "../lib/plot";
import { PLOT3D_LAYOUT } from "../lib/threed";
import { usd } from "../lib/utils";

interface FRow {
  date: string;
  service: string;
  kind: string;
  cost: number;
  lower: number;
  upper: number;
}

export default function Forecast() {
  const { data } = useSnapshot();
  const [mode, setMode] = useState<"3d" | "2d">("3d");
  const [svc, setSvc] = useState<string>("");
  if (!data) return null;

  const services = data.meta.services;
  const all = data.forecast as unknown as FRow[];

  // 3D: every service as a ribbon at its own depth - history solid, forecast dotted.
  const traces3d: Record<string, unknown>[] = [];
  services.forEach((s, i) => {
    const rows = all.filter((r) => r.service === s);
    const hist = rows.filter((r) => r.kind !== "forecast");
    const fc = rows.filter((r) => r.kind === "forecast");
    traces3d.push({
      type: "scatter3d", mode: "lines", name: s,
      x: hist.map((r) => r.date), y: hist.map(() => i), z: hist.map((r) => r.cost),
      line: { width: 5 },
    });
    traces3d.push({
      type: "scatter3d", mode: "lines", name: `${s} forecast`, showlegend: false,
      x: fc.map((r) => r.date), y: fc.map(() => i), z: fc.map((r) => r.cost),
      line: { width: 5, dash: "dot" },
    });
  });

  // 2D: one selected service with the 90% prediction-interval band.
  const active = svc || services[0];
  const rows = all.filter((r) => r.service === active);
  const hist = rows.filter((r) => r.kind !== "forecast");
  const fc = rows.filter((r) => r.kind === "forecast");
  const traces2d = [
    { x: fc.map((r) => r.date), y: fc.map((r) => r.upper), type: "scatter", mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
    { x: fc.map((r) => r.date), y: fc.map((r) => r.lower), type: "scatter", mode: "lines", fill: "tonexty", fillcolor: "rgba(30,64,175,0.15)", line: { width: 0 }, name: "90% PI" },
    { x: hist.map((r) => r.date), y: hist.map((r) => r.cost), type: "scatter", mode: "lines", name: "Actual", line: { color: "#64748b", width: 2 } },
    { x: fc.map((r) => r.date), y: fc.map((r) => r.cost), type: "scatter", mode: "lines", name: "Forecast", line: { color: "#1e40af", width: 2, dash: "dot" } },
  ];

  return (
    <div>
      <SectionTitle
        icon={TrendingUp}
        title="Holt-Winters forecast"
        subtitle="Per-service additive Holt-Winters with weekly seasonality. 3D shows every service as a ribbon (history solid, forecast dotted); 2D drills into one service with its 90% prediction band."
      />
      <Card>
        <CardBody>
          <div className="mb-2 flex items-center justify-between">
            {mode === "2d" ? (
              <select
                value={active}
                onChange={(e) => setSvc(e.target.value)}
                className="rounded-md border border-border bg-card px-3 py-1.5 text-sm"
              >
                {services.map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            ) : (
              <span className="text-xs text-muted-foreground">All services · drag to orbit</span>
            )}
            <ModeToggle mode={mode} onChange={setMode} />
          </div>
          {mode === "3d" ? (
            <Plot
              data={traces3d}
              layout={{
                ...PLOT3D_LAYOUT,
                height: 460,
                showlegend: true,
                scene: {
                  xaxis: { title: "Date" },
                  yaxis: { tickvals: services.map((_, i) => i), ticktext: services, title: "" },
                  zaxis: { title: "Cost ($)" },
                  camera: { eye: { x: 1.8, y: 1.4, z: 0.9 } },
                  aspectmode: "cube",
                },
              }}
              config={PLOT_CONFIG} useResizeHandler style={{ width: "100%" }}
            />
          ) : (
            <Plot
              data={traces2d}
              layout={{ ...PLOT_LAYOUT_BASE, height: 420, yaxis: { title: "Cost ($)" } }}
              config={PLOT_CONFIG} useResizeHandler style={{ width: "100%" }}
            />
          )}
        </CardBody>
      </Card>
      <Card className="mt-4">
        <CardBody>
          <div className="mb-2 text-sm font-medium">Projected monthly spend</div>
          <DataTable
            rows={data.projected_monthly}
            columns={[
              { key: "service", label: "Service" },
              { key: "daily_avg", label: "Daily avg", align: "right", render: (r) => usd(r.daily_avg as number, 2) },
              { key: "projected_monthly", label: "Projected / month", align: "right", render: (r) => usd(r.projected_monthly as number) },
            ]}
          />
        </CardBody>
      </Card>
    </div>
  );
}
