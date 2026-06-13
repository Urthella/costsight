import { BarChart3 } from "lucide-react";
import { useSnapshot } from "../hooks/useSnapshot";
import { Card, CardBody, SectionTitle } from "../components/ui";
import { DataTable } from "../components/DataTable";
import Plot, { PLOT_CONFIG, PLOT_LAYOUT_BASE } from "../lib/plot";
import { DETECTOR_COLOR, DETECTOR_LABEL } from "../lib/utils";

const TYPES = ["point_spike", "level_shift", "gradual_drift"];

export default function DetectorComparison() {
  const { data } = useSnapshot();
  if (!data) return null;

  const hasTruth = data.comparison.some((c) => c.f1 > 0 || c.tp > 0 || c.fn > 0);

  const traces = data.detectors.map((det) => ({
    x: TYPES,
    y: TYPES.map((t) => {
      const row = data.comparison.find(
        (c) => c.detector === det && c.anomaly_type === t,
      );
      return row ? row.f1 : 0;
    }),
    type: "bar",
    name: DETECTOR_LABEL[det] ?? det,
    marker: { color: DETECTOR_COLOR[det] ?? "#1e40af" },
  }));

  return (
    <div>
      <SectionTitle
        icon={BarChart3}
        title="Detector comparison"
        subtitle="Precision / Recall / F1 per anomaly type — no single detector wins everywhere."
      />
      {!hasTruth && (
        <div className="mb-3 rounded-lg border border-border bg-muted/50 p-3 text-sm text-muted-foreground">
          No ground-truth labels for this dataset, so P/R/F1 is blank.
        </div>
      )}
      <Card>
        <CardBody>
          <Plot
            data={traces}
            layout={{
              ...PLOT_LAYOUT_BASE,
              height: 380,
              barmode: "group",
              yaxis: { title: "F1", range: [0, 1] },
              xaxis: { title: "Anomaly type" },
            }}
            config={PLOT_CONFIG}
            useResizeHandler
            style={{ width: "100%" }}
          />
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
