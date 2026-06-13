import { useState } from "react";
import { TrendingUp } from "lucide-react";
import { useSnapshot } from "../hooks/useSnapshot";
import { Card, CardBody, SectionTitle } from "../components/ui";
import { DataTable } from "../components/DataTable";
import Plot, { PLOT_CONFIG, PLOT_LAYOUT_BASE } from "../lib/plot";
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
  const [svc, setSvc] = useState<string>("");
  if (!data) return null;

  const services = data.meta.services;
  const active = svc || services[0];
  const rows = (data.forecast as unknown as FRow[]).filter((r) => r.service === active);
  const actual = rows.filter((r) => r.kind !== "forecast");
  const fc = rows.filter((r) => r.kind === "forecast");

  const traces = [
    { x: fc.map((r) => r.date), y: fc.map((r) => r.upper), type: "scatter", mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
    { x: fc.map((r) => r.date), y: fc.map((r) => r.lower), type: "scatter", mode: "lines", fill: "tonexty", fillcolor: "rgba(30,64,175,0.15)", line: { width: 0 }, name: "90% PI" },
    { x: actual.map((r) => r.date), y: actual.map((r) => r.cost), type: "scatter", mode: "lines", name: "Actual", line: { color: "#64748b", width: 2 } },
    { x: fc.map((r) => r.date), y: fc.map((r) => r.cost), type: "scatter", mode: "lines", name: "Forecast", line: { color: "#1e40af", width: 2, dash: "dot" } },
  ];

  return (
    <div>
      <SectionTitle
        icon={TrendingUp}
        title="Holt-Winters forecast"
        subtitle="Per-service additive Holt-Winters with weekly seasonality and a 90% prediction interval."
      />
      <div className="mb-3">
        <select
          value={active}
          onChange={(e) => setSvc(e.target.value)}
          className="rounded-md border border-border bg-card px-3 py-1.5 text-sm"
        >
          {services.map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
      </div>
      <Card>
        <CardBody>
          <Plot
            data={traces}
            layout={{ ...PLOT_LAYOUT_BASE, height: 400, yaxis: { title: "Cost ($)" } }}
            config={PLOT_CONFIG} useResizeHandler style={{ width: "100%" }}
          />
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
