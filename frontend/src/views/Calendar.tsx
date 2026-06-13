import { useState } from "react";
import { CalendarDays } from "lucide-react";
import { useSnapshot } from "../hooks/useSnapshot";
import { Card, CardBody, SectionTitle, ModeToggle } from "../components/ui";
import Plot, { PLOT_CONFIG, PLOT_LAYOUT_BASE } from "../lib/plot";
import { PLOT3D_LAYOUT } from "../lib/threed";

export default function Calendar() {
  const { data } = useSnapshot();
  const [mode, setMode] = useState<"3d" | "2d">("3d");
  if (!data) return null;

  const services = data.meta.services;
  const dates = [...new Set(data.series.map((s) => s.date))].sort();
  const lookup = new Map<string, number>();
  for (const s of data.series) lookup.set(`${s.service}|${s.date}`, s.cost);
  const z = services.map((svc) => dates.map((d) => lookup.get(`${svc}|${d}`) ?? 0));

  return (
    <div>
      <SectionTitle
        icon={CalendarDays}
        title="Cost calendar"
        subtitle="Cost per (service, day). The 3D surface turns spikes into peaks and drift into rising ridges; toggle 2D for the flat heatmap."
      />
      <Card>
        <CardBody>
          <div className="mb-2 flex justify-end">
            <ModeToggle mode={mode} onChange={setMode} />
          </div>
          {mode === "3d" ? (
            <Plot
              data={[{ type: "surface", x: dates, y: services, z, colorscale: "YlOrRd", showscale: true, contours: { z: { show: true, usecolormap: true, project: { z: true } } } }]}
              layout={{
                ...PLOT3D_LAYOUT,
                height: 480,
                scene: {
                  xaxis: { title: "Date" },
                  yaxis: { title: "Service" },
                  zaxis: { title: "Cost ($)" },
                  camera: { eye: { x: 1.8, y: 1.6, z: 1.0 } },
                  aspectmode: "cube",
                },
              }}
              config={PLOT_CONFIG}
              useResizeHandler
              style={{ width: "100%" }}
            />
          ) : (
            <Plot
              data={[{ type: "heatmap", x: dates, y: services, z, colorscale: "YlOrRd", colorbar: { title: "$" } }]}
              layout={{
                ...PLOT_LAYOUT_BASE,
                height: 420,
                margin: { t: 20, r: 16, b: 60, l: 90 },
                xaxis: { title: "Date" },
                yaxis: { title: "Service" },
              }}
              config={PLOT_CONFIG}
              useResizeHandler
              style={{ width: "100%" }}
            />
          )}
        </CardBody>
      </Card>
    </div>
  );
}
