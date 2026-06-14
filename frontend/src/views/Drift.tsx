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

export default function Drift() {
  const { data } = useSnapshot();
  const [mode, setMode] = useState<"3d" | "2d">("3d");
  if (!data) return null;
  const drift = data.drift as unknown as DriftRow[];

  if (!drift.length)
    return (
      <div>
        <SectionTitle icon={Waves} title="Concept drift" />
        <Card><CardBody><div className="py-8 text-center text-sm text-muted-foreground">No baseline drift detected - workload is in a stable regime.</div></CardBody></Card>
      </div>
    );

  const services = data.meta.services;
  const COLOR = (dir: string) => (dir === "up" ? "#d97706" : "#2563eb");

  // 2D: daily total with dashed drift markers.
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
        subtitle="Page-Hinkley + ADWIN flag baseline shifts (gold = up, blue = down). 3D plots each event by date × service × magnitude; drag to orbit."
      />
      <Card>
        <CardBody>
          <div className="mb-2 flex justify-end"><ModeToggle mode={mode} onChange={setMode} /></div>
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
    </div>
  );
}
