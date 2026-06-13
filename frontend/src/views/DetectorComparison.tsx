import { useState } from "react";
import { BarChart3 } from "lucide-react";
import { useSnapshot } from "../hooks/useSnapshot";
import { Card, CardBody, SectionTitle, ModeToggle } from "../components/ui";
import { DataTable } from "../components/DataTable";
import Plot, { PLOT_CONFIG, PLOT_LAYOUT_BASE } from "../lib/plot";
import { bars3dTrace, barLabels, scene3d, PLOT3D_LAYOUT, type Bar3D } from "../lib/threed";
import { DETECTOR_COLOR, DETECTOR_LABEL } from "../lib/utils";

const TYPES = ["point_spike", "level_shift", "gradual_drift"];

export default function DetectorComparison() {
  const { data } = useSnapshot();
  const [mode, setMode] = useState<"3d" | "2d">("3d");
  if (!data) return null;

  const hasTruth = data.comparison.some((c) => c.f1 > 0 || c.tp > 0 || c.fn > 0);
  const dets = data.detectors;

  // 3D bars: anomaly_type (x) × detector (y) × F1 (height).
  const bars: Bar3D[] = [];
  const labels: string[] = [];
  TYPES.forEach((t, xi) =>
    dets.forEach((det, yi) => {
      const row = data.comparison.find((c) => c.detector === det && c.anomaly_type === t);
      const f1 = row ? row.f1 : 0;
      bars.push({ x: xi, y: yi, h: f1, color: DETECTOR_COLOR[det] ?? "#1e40af" });
      labels.push(`${DETECTOR_LABEL[det] ?? det} · ${t}: F1 ${f1.toFixed(3)}`);
    }),
  );

  const traces2d = dets.map((det) => ({
    x: TYPES,
    y: TYPES.map((t) => data.comparison.find((c) => c.detector === det && c.anomaly_type === t)?.f1 ?? 0),
    type: "bar",
    name: DETECTOR_LABEL[det] ?? det,
    marker: { color: DETECTOR_COLOR[det] ?? "#1e40af" },
  }));

  return (
    <div>
      <SectionTitle
        icon={BarChart3}
        title="Detector comparison"
        subtitle="F1 per anomaly type across detectors — no single detector wins everywhere. Drag the 3D chart to orbit."
      />
      {!hasTruth && (
        <div className="mb-3 rounded-lg border border-border bg-muted/50 p-3 text-sm text-muted-foreground">
          No ground-truth labels for this dataset, so P/R/F1 is blank.
        </div>
      )}
      <Card>
        <CardBody>
          <div className="mb-2 flex justify-end">
            <ModeToggle mode={mode} onChange={setMode} />
          </div>
          {mode === "3d" ? (
            <Plot
              data={[bars3dTrace(bars), barLabels(bars, labels)]}
              layout={{
                ...PLOT3D_LAYOUT,
                scene: scene3d(["point spike", "level shift", "gradual drift"], dets.map((d) => DETECTOR_LABEL[d] ?? d), "F1"),
              }}
              config={PLOT_CONFIG}
              useResizeHandler
              style={{ width: "100%" }}
            />
          ) : (
            <Plot
              data={traces2d}
              layout={{
                ...PLOT_LAYOUT_BASE,
                height: 380,
                barmode: "group",
                yaxis: { title: "F1", range: [0, 1] },
                xaxis: { title: "Anomaly type" },
                showlegend: true,
              }}
              config={PLOT_CONFIG}
              useResizeHandler
              style={{ width: "100%" }}
            />
          )}
        </CardBody>
      </Card>
      <Card className="mt-4">
        <CardBody>
          <DataTable
            rows={data.comparison as unknown as Record<string, unknown>[]}
            columns={[
              { key: "detector", label: "Detector", render: (r) => DETECTOR_LABEL[r.detector as string] ?? (r.detector as string) },
              { key: "anomaly_type", label: "Type" },
              { key: "precision", label: "Precision", align: "right", render: (r) => (r.precision as number).toFixed(3) },
              { key: "recall", label: "Recall", align: "right", render: (r) => (r.recall as number).toFixed(3) },
              { key: "f1", label: "F1", align: "right", render: (r) => (r.f1 as number).toFixed(3) },
            ]}
          />
        </CardBody>
      </Card>
    </div>
  );
}
