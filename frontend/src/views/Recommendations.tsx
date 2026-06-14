import { Lightbulb } from "lucide-react";
import { useSnapshot } from "../hooks/useSnapshot";
import { Card, CardBody, SectionTitle } from "../components/ui";
import Plot, { PLOT_CONFIG, PLOT_LAYOUT_BASE } from "../lib/plot";
import { usd } from "../lib/utils";

const CONF_DOT: Record<string, string> = {
  high: "bg-green-500",
  medium: "bg-amber-500",
  low: "bg-orange-500",
};

export default function Recommendations() {
  const { data } = useSnapshot();
  if (!data) return null;
  const recs = data.recommendations;
  if (!recs.length)
    return (
      <div>
        <SectionTitle icon={Lightbulb} title="Cost-optimization recommendations" />
        <Card><CardBody><div className="py-8 text-center text-sm text-muted-foreground">No optimization candidates - workload is already tight.</div></CardBody></Card>
      </div>
    );

  const total = recs.reduce((s, r) => s + r.impact_usd_per_month, 0);
  const byCat = new Map<string, number>();
  for (const r of recs) byCat.set(r.category, (byCat.get(r.category) ?? 0) + r.impact_usd_per_month);
  const cats = [...byCat.entries()].sort((a, b) => a[1] - b[1]);

  return (
    <div>
      <SectionTitle
        icon={Lightbulb}
        title="Cost-optimization recommendations"
        subtitle="Independent of anomalies - heuristics scan the CUR for the most common FinOps wins."
      />
      <div className="grid grid-cols-3 gap-3">
        <Card><CardBody><div className="text-xs text-muted-foreground">Findings</div><div className="mt-1 text-2xl font-semibold">{recs.length}</div></CardBody></Card>
        <Card><CardBody><div className="text-xs text-muted-foreground">Total monthly savings</div><div className="mt-1 text-2xl font-semibold text-primary">{usd(total)}</div></CardBody></Card>
        <Card><CardBody><div className="text-xs text-muted-foreground">Annualized</div><div className="mt-1 text-2xl font-semibold">{usd(total * 12)}</div></CardBody></Card>
      </div>

      {cats.length >= 2 && (
        <Card className="mt-4"><CardBody>
          <div className="mb-2 text-sm font-medium">Monthly savings by category</div>
          <Plot
            data={[{ x: cats.map((c) => c[1]), y: cats.map((c) => c[0]), type: "bar", orientation: "h", marker: { color: "#1e40af" } }]}
            layout={{ ...PLOT_LAYOUT_BASE, height: 280, xaxis: { title: "$ / month" }, margin: { t: 10, r: 16, b: 40, l: 120 } }}
            config={PLOT_CONFIG} useResizeHandler style={{ width: "100%" }}
          />
        </CardBody></Card>
      )}

      <div className="mt-4 space-y-3">
        {recs.map((r, i) => (
          <Card key={i}><CardBody>
            <div className="flex items-center gap-2">
              <span className={"h-2.5 w-2.5 rounded-full " + (CONF_DOT[r.confidence] ?? "bg-slate-400")} />
              <span className="font-medium">{r.category}</span>
              <span className="text-muted-foreground">· {r.service} · {r.region}</span>
              <span className="ml-auto font-semibold text-primary">{usd(r.impact_usd_per_month)}/mo</span>
            </div>
            <p className="mt-2 text-sm">{r.action}</p>
            <p className="mt-1 text-xs italic text-muted-foreground">{r.rationale}</p>
          </CardBody></Card>
        ))}
      </div>
    </div>
  );
}
