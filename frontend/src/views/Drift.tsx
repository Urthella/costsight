import { Waves } from "lucide-react";
import { useSnapshot } from "../hooks/useSnapshot";
import { Card, CardBody, SectionTitle } from "../components/ui";
import { DataTable } from "../components/DataTable";
import Plot, { PLOT_CONFIG, PLOT_LAYOUT_BASE } from "../lib/plot";

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
  if (!data) return null;
  const drift = data.drift as unknown as DriftRow[];

  if (!drift.length)
    return (
      <div>
        <SectionTitle icon={Waves} title="Concept drift" />
        <Card><CardBody><div className="py-8 text-center text-sm text-muted-foreground">No baseline drift detected — workload is in a stable regime.</div></CardBody></Card>
      </div>
    );

  const shapes = drift.map((d) => ({
    type: "line",
    xref: "x",
    yref: "paper",
    x0: d.change_date,
    x1: d.change_date,
    y0: 0,
    y1: 1,
    line: { color: d.direction === "up" ? "#d97706" : "#2563eb", width: 1.5, dash: "dash" },
  }));

  return (
    <div>
      <SectionTitle
        icon={Waves}
        title="Concept drift — has the baseline itself shifted?"
        subtitle="Page-Hinkley + ADWIN flag baseline shifts (gold = up, blue = down), independent of the anomaly stream."
      />
      <Card>
        <CardBody>
          <Plot
            data={[
              { x: data.daily.map((d) => d.date), y: data.daily.map((d) => d.cost), type: "scatter", mode: "lines", name: "Daily cost", line: { color: "#64748b", width: 2 } },
            ]}
            layout={{ ...PLOT_LAYOUT_BASE, height: 360, shapes, yaxis: { title: "Cost ($)" } }}
            config={PLOT_CONFIG}
            useResizeHandler
            style={{ width: "100%" }}
          />
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
