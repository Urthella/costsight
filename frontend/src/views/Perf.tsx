import { useState } from "react";
import { Zap } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { getPerf } from "../lib/api";
import { useDashboardParams } from "../state/params";
import { Card, CardBody, SectionTitle, ModeToggle } from "../components/ui";
import { DataTable } from "../components/DataTable";
import Plot, { PLOT_CONFIG, PLOT_LAYOUT_BASE } from "../lib/plot";
import { bars3dTrace, barLabels, scene3d, PLOT3D_LAYOUT, type Bar3D } from "../lib/threed";
import { DETECTOR_COLOR, DETECTOR_LABEL } from "../lib/utils";

interface PerfRow {
  detector: string;
  n_days: number;
  rows_per_second: number;
  seconds_per_run: number;
}

export default function Perf() {
  const { params } = useDashboardParams();
  const [mode, setMode] = useState<"3d" | "2d">("3d");
  const { data, isLoading, isError } = useQuery({
    queryKey: ["perf", params.scenario, params.nDays, params.seed],
    queryFn: () => getPerf(params),
    staleTime: 5 * 60 * 1000,
  });

  return (
    <div>
      <SectionTitle
        icon={Zap}
        title="Detector performance - measured runtime"
        subtitle="Throughput across dataset sizes. Heights are log10(rows/sec) so Isolation Forest (~100/s) and Z-Score (~80k/s) are both visible."
      />
      {isLoading && <div className="animate-pulse text-sm text-muted-foreground">Benchmarking detectors…</div>}
      {isError && <div className="text-sm text-high">Failed to run the benchmark.</div>}
      {data && (() => {
        const rows = data.perf as unknown as PerfRow[];
        const detectors = [...new Set(rows.map((r) => r.detector))];
        const sizes = [...new Set(rows.map((r) => r.n_days))].sort((a, b) => a - b);

        const bars: Bar3D[] = [];
        const labels: string[] = [];
        sizes.forEach((n, xi) =>
          detectors.forEach((det, yi) => {
            const rps = rows.find((r) => r.detector === det && r.n_days === n)?.rows_per_second ?? 1;
            bars.push({ x: xi, y: yi, h: Math.log10(Math.max(rps, 1)), color: DETECTOR_COLOR[det] ?? "#1e40af" });
            labels.push(`${DETECTOR_LABEL[det] ?? det} @ ${n}d: ${Math.round(rps).toLocaleString()} rows/s`);
          }),
        );

        const traces2d = detectors.map((det) => ({
          x: sizes,
          y: sizes.map((n) => rows.find((r) => r.detector === det && r.n_days === n)?.rows_per_second ?? 0),
          type: "bar",
          name: DETECTOR_LABEL[det] ?? det,
          marker: { color: DETECTOR_COLOR[det] ?? "#1e40af" },
        }));

        return (
          <>
            <Card><CardBody>
              <div className="mb-2 flex justify-end"><ModeToggle mode={mode} onChange={setMode} /></div>
              {mode === "3d" ? (
                <Plot
                  data={[bars3dTrace(bars), barLabels(bars, labels)]}
                  layout={{ ...PLOT3D_LAYOUT, scene: scene3d(sizes.map((s) => `${s}d`), detectors.map((d) => DETECTOR_LABEL[d] ?? d), "log₁₀ rows/s") }}
                  config={PLOT_CONFIG} useResizeHandler style={{ width: "100%" }}
                />
              ) : (
                <Plot
                  data={traces2d}
                  layout={{ ...PLOT_LAYOUT_BASE, height: 340, barmode: "group", xaxis: { title: "Dataset size (days)" }, yaxis: { title: "rows / second (log)", type: "log" }, showlegend: true }}
                  config={PLOT_CONFIG} useResizeHandler style={{ width: "100%" }}
                />
              )}
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
