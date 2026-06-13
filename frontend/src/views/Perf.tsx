import { Zap } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { getPerf } from "../lib/api";
import { useDashboardParams } from "../state/params";
import { Card, CardBody, SectionTitle } from "../components/ui";
import { DataTable } from "../components/DataTable";
import Plot, { PLOT_CONFIG, PLOT_LAYOUT_BASE } from "../lib/plot";
import { DETECTOR_COLOR, DETECTOR_LABEL } from "../lib/utils";

export default function Perf() {
  const { params } = useDashboardParams();
  const { data, isLoading, isError } = useQuery({
    queryKey: ["perf", params.scenario, params.nDays, params.seed],
    queryFn: () => getPerf(params),
    staleTime: 5 * 60 * 1000,
  });

  return (
    <div>
      <SectionTitle
        icon={Zap}
        title="Detector performance — measured runtime"
        subtitle="Wall-clock per run and throughput across dataset sizes. Computed on demand (it re-runs each detector)."
      />
      {isLoading && <div className="animate-pulse text-sm text-muted-foreground">Benchmarking detectors…</div>}
      {isError && <div className="text-sm text-high">Failed to run the benchmark.</div>}
      {data && (() => {
        const rows = data.perf as {
          detector: string;
          n_days: number;
          rows_per_second: number;
        }[];
        const detectors = [...new Set(rows.map((r) => r.detector))];
        const sizes = [...new Set(rows.map((r) => r.n_days))].sort((a, b) => a - b);
        const traces = detectors.map((det) => ({
          x: sizes,
          y: sizes.map((n) => rows.find((r) => r.detector === det && r.n_days === n)?.rows_per_second ?? 0),
          type: "bar",
          name: DETECTOR_LABEL[det] ?? det,
          marker: { color: DETECTOR_COLOR[det] ?? "#1e40af" },
        }));
        return (
          <>
            <Card><CardBody>
              <div className="mb-1 text-xs text-muted-foreground">
                Throughput varies ~1000× across detectors (Z-Score vectorizes; Isolation
                Forest fits a model), so the axis is log-scaled — otherwise the slow
                detectors look like zero.
              </div>
              <Plot
                data={traces}
                layout={{
                  ...PLOT_LAYOUT_BASE,
                  height: 340,
                  barmode: "group",
                  xaxis: { title: "Dataset size (days)" },
                  yaxis: { title: "rows / second (log)", type: "log" },
                }}
                config={PLOT_CONFIG} useResizeHandler style={{ width: "100%" }}
              />
            </CardBody></Card>
            <Card className="mt-4"><CardBody>
              <DataTable
                rows={data.perf}
                columns={[
                  { key: "detector", label: "Detector", render: (r) => DETECTOR_LABEL[r.detector as string] ?? (r.detector as string) },
                  { key: "n_days", label: "Days", align: "right" },
                  { key: "seconds_per_run", label: "Sec / run", align: "right", render: (r) => (r.seconds_per_run as number).toFixed(4) },
                  { key: "rows_per_second", label: "Rows / sec", align: "right", render: (r) => Math.round(r.rows_per_second as number).toLocaleString() },
                ]}
              />
            </CardBody></Card>
          </>
        );
      })()}
    </div>
  );
}
