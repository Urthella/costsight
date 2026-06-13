import { LineChart as LineIcon } from "lucide-react";
import { useSnapshot } from "../hooks/useSnapshot";
import { Card, CardBody, SectionTitle } from "../components/ui";
import Plot, { PLOT_CONFIG, PLOT_LAYOUT_BASE } from "../lib/plot";
import { DETECTOR_COLOR, DETECTOR_LABEL } from "../lib/utils";

const SYMBOL: Record<string, string> = {
  zscore: "circle",
  stl: "square",
  iforest: "triangle-up",
  ensemble: "star",
};

export default function CostTrend() {
  const { data } = useSnapshot();
  if (!data) return null;

  const dateToCost = new Map(data.daily.map((d) => [d.date, d.cost]));

  const traces: Record<string, unknown>[] = [
    {
      x: data.daily.map((d) => d.date),
      y: data.daily.map((d) => d.cost),
      type: "scatter",
      mode: "lines",
      name: "Daily cost",
      line: { color: "#64748b", width: 2 },
    },
  ];

  for (const det of data.detectors) {
    const flaggedDates = [
      ...new Set(
        (data.detections[det] ?? [])
          .filter((d) => d.is_anomaly)
          .map((d) => d.date),
      ),
    ];
    if (!flaggedDates.length) continue;
    traces.push({
      x: flaggedDates,
      y: flaggedDates.map((d) => dateToCost.get(d) ?? null),
      type: "scatter",
      mode: "markers",
      name: DETECTOR_LABEL[det] ?? det,
      marker: {
        color: DETECTOR_COLOR[det] ?? "#1e40af",
        size: 10,
        symbol: SYMBOL[det] ?? "circle",
        line: { width: 1, color: "#1e293b" },
      },
    });
  }

  // Ground-truth onset markers as faint vertical lines.
  const truthDates = [
    ...new Set(
      data.ground_truth
        .filter((r) => (r as { is_anomaly?: boolean }).is_anomaly)
        .map((r) => (r as { date: string }).date),
    ),
  ];
  const shapes = truthDates.map((d) => ({
    type: "line",
    xref: "x",
    yref: "paper",
    x0: d,
    x1: d,
    y0: 0,
    y1: 1,
    line: { color: "rgba(16,185,129,0.35)", width: 1 },
  }));

  // One line per service.
  const services = [...new Set(data.series.map((s) => s.service))];
  const perService = services.map((svc) => {
    const rows = data.series.filter((s) => s.service === svc);
    return {
      x: rows.map((r) => r.date),
      y: rows.map((r) => r.cost),
      type: "scatter",
      mode: "lines",
      name: svc,
    };
  });

  return (
    <div>
      <SectionTitle
        icon={LineIcon}
        title="Daily total cloud spend"
        subtitle="Markers = detector flags (by shape). Faint green lines = ground-truth anomaly days."
      />
      <Card>
        <CardBody>
          <Plot
            data={traces}
            layout={{
              ...PLOT_LAYOUT_BASE,
              height: 420,
              shapes,
              xaxis: { title: "Date" },
              yaxis: { title: "Cost ($)" },
            }}
            config={PLOT_CONFIG}
            useResizeHandler
            style={{ width: "100%" }}
          />
        </CardBody>
      </Card>

      <SectionTitle
        title="Per-service breakdown"
        subtitle="Where anomalies actually live — drift and level shifts are visible per service."
      />
      <Card>
        <CardBody>
          <Plot
            data={perService}
            layout={{ ...PLOT_LAYOUT_BASE, height: 420 }}
            config={PLOT_CONFIG}
            useResizeHandler
            style={{ width: "100%" }}
          />
        </CardBody>
      </Card>
    </div>
  );
}
