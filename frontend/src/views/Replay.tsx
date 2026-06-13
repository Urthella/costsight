import { useEffect, useState } from "react";
import { Play, Pause } from "lucide-react";
import { useSnapshot } from "../hooks/useSnapshot";
import { Card, CardBody, SectionTitle } from "../components/ui";
import Plot, { PLOT_CONFIG, PLOT_LAYOUT_BASE } from "../lib/plot";

export default function Replay() {
  const { data } = useSnapshot();
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);

  const n = data?.daily.length ?? 0;

  useEffect(() => {
    if (!playing || n === 0) return;
    const t = setInterval(() => {
      setIdx((i) => {
        if (i >= n - 1) {
          setPlaying(false);
          return i;
        }
        return i + 1;
      });
    }, 120);
    return () => clearInterval(t);
  }, [playing, n]);

  if (!data) return null;
  const cur = Math.min(idx, n - 1);
  const upto = data.daily.slice(0, cur + 1);
  const curDate = data.daily[cur]?.date;

  // Alerts revealed so far (deduped by date+service), marked on the line.
  const dateToCost = new Map(data.daily.map((d) => [d.date, d.cost]));
  const revealed = [
    ...new Set(
      data.alerts.filter((a) => a.date <= (curDate ?? "")).map((a) => a.date),
    ),
  ];

  return (
    <div>
      <SectionTitle
        icon={Play}
        title="Day-by-day replay"
        subtitle="Walks the dataset one day at a time — anomalies appear as they would in real-life monitoring."
      />
      <Card>
        <CardBody>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setPlaying((p) => !p)}
              className="flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground"
            >
              {playing ? <Pause size={15} /> : <Play size={15} />}
              {playing ? "Pause" : "Play"}
            </button>
            <input
              type="range" min={0} max={Math.max(0, n - 1)} value={cur}
              onChange={(e) => { setIdx(Number(e.target.value)); setPlaying(false); }}
              className="flex-1 accent-[var(--color-primary)]"
            />
            <span className="w-24 text-right text-sm tabular-nums text-muted-foreground">{curDate}</span>
          </div>

          <div className="mt-4">
            <Plot
              data={[
                { x: upto.map((d) => d.date), y: upto.map((d) => d.cost), type: "scatter", mode: "lines", name: "Daily cost", line: { color: "#64748b", width: 2 } },
                { x: revealed, y: revealed.map((d) => dateToCost.get(d) ?? null), type: "scatter", mode: "markers", name: "Anomalies", marker: { color: "#dc2626", size: 11, symbol: "x", line: { width: 2, color: "#1e293b" } } },
              ]}
              layout={{
                ...PLOT_LAYOUT_BASE,
                height: 420,
                yaxis: { title: "Cost ($)" },
                xaxis: { range: [data.daily[0]?.date, data.daily[n - 1]?.date] },
              }}
              config={PLOT_CONFIG} useResizeHandler style={{ width: "100%" }}
            />
          </div>
        </CardBody>
      </Card>
    </div>
  );
}
