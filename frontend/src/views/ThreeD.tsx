import { Boxes } from "lucide-react";
import { useSnapshot } from "../hooks/useSnapshot";
import { Card, CardBody, SectionTitle } from "../components/ui";
import { SpendSkyline } from "../components/SpendSkyline";
import Plot, { PLOT_CONFIG } from "../lib/plot";
import { usd } from "../lib/utils";

export default function ThreeD() {
  const { data } = useSnapshot();
  if (!data) return null;

  // Per-service totals for the WebGL skyline.
  const totals = new Map<string, number>();
  for (const s of data.series) totals.set(s.service, (totals.get(s.service) ?? 0) + s.cost);
  const skyline = [...totals.entries()]
    .map(([service, total]) => ({ service, total }))
    .sort((a, b) => b.total - a.total);

  // Per-service alert counts shown in the skyline hover labels.
  const alertCount: Record<string, number> = {};
  for (const a of data.alerts) alertCount[a.service] = (alertCount[a.service] ?? 0) + 1;
  const skylineInfo = Object.fromEntries(
    Object.entries(alertCount).map(([k, v]) => [k, `${v} alerts`]),
  );

  // Surface: services (y) × dates (x) → cost (z).
  const services = data.meta.services;
  const dates = [...new Set(data.series.map((s) => s.date))].sort();
  const lookup = new Map<string, number>();
  for (const s of data.series) lookup.set(`${s.service}|${s.date}`, s.cost);
  const z = services.map((svc) => dates.map((d) => lookup.get(`${svc}|${d}`) ?? 0));

  // Scatter3d: every point, plus anomalies highlighted.
  const anomKeys = new Set(data.alerts.map((a) => `${a.service}|${a.date}`));
  const base = data.series.filter((s) => !anomKeys.has(`${s.service}|${s.date}`));
  const anom = data.series.filter((s) => anomKeys.has(`${s.service}|${s.date}`));

  return (
    <div>
      <SectionTitle
        icon={Boxes}
        title="3D explorer"
        subtitle="WebGL spend skyline + interactive 3D surface and anomaly cloud. Drag to orbit, scroll to zoom."
      />

      <Card>
        <CardBody>
          <div className="mb-2 text-sm font-medium">Spend skyline (per-service total, auto-rotating)</div>
          <SpendSkyline data={skyline} info={skylineInfo} />
          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
            {skyline.map((b) => (
              <span key={b.service}>
                <span className="font-medium text-foreground">{b.service}</span> {usd(b.total)}
              </span>
            ))}
          </div>
        </CardBody>
      </Card>

      <div className="mt-4 grid grid-cols-1 gap-3 xl:grid-cols-2">
        <Card>
          <CardBody>
            <div className="mb-2 text-sm font-medium">Cost surface (service × day)</div>
            <Plot
              data={[{ type: "surface", x: dates, y: services, z, colorscale: "YlOrRd", showscale: false }]}
              layout={{
                height: 420,
                margin: { t: 0, r: 0, b: 0, l: 0 },
                scene: {
                  xaxis: { title: "Date" },
                  yaxis: { title: "Service" },
                  zaxis: { title: "Cost ($)" },
                },
                paper_bgcolor: "rgba(0,0,0,0)",
              }}
              config={PLOT_CONFIG}
              useResizeHandler
              style={{ width: "100%" }}
            />
          </CardBody>
        </Card>
        <Card>
          <CardBody>
            <div className="mb-2 text-sm font-medium">Anomaly cloud (red = flagged)</div>
            <Plot
              data={[
                {
                  type: "scatter3d",
                  mode: "markers",
                  name: "Normal",
                  x: base.map((s) => s.date),
                  y: base.map((s) => s.service),
                  z: base.map((s) => s.cost),
                  marker: { size: 2, color: "#94a3b8", opacity: 0.55 },
                },
                {
                  type: "scatter3d",
                  mode: "markers",
                  name: "Anomaly",
                  x: anom.map((s) => s.date),
                  y: anom.map((s) => s.service),
                  z: anom.map((s) => s.cost),
                  marker: { size: 4, color: "#dc2626" },
                },
              ]}
              layout={{
                height: 420,
                margin: { t: 0, r: 0, b: 0, l: 0 },
                legend: { orientation: "h", y: 0 },
                scene: {
                  xaxis: { title: "Date" },
                  yaxis: { title: "Service" },
                  zaxis: { title: "Cost ($)" },
                },
                paper_bgcolor: "rgba(0,0,0,0)",
              }}
              config={PLOT_CONFIG}
              useResizeHandler
              style={{ width: "100%" }}
            />
          </CardBody>
        </Card>
      </div>
    </div>
  );
}
