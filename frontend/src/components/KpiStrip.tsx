import type { Kpis } from "../types";
import { Card, CardBody } from "./ui";
import { usd, DETECTOR_LABEL } from "../lib/utils";

function Kpi({
  label,
  value,
  hint,
  accent,
}: {
  label: string;
  value: string;
  hint?: string;
  accent?: boolean;
}) {
  return (
    <Card className="flex-1">
      <CardBody className="p-3">
        <div className="text-xs text-muted-foreground">{label}</div>
        <div
          className={
            "mt-1 text-2xl font-semibold " + (accent ? "text-primary" : "")
          }
        >
          {value}
        </div>
        {hint && <div className="mt-0.5 text-xs text-muted-foreground">{hint}</div>}
      </CardBody>
    </Card>
  );
}

export function KpiStrip({ kpis }: { kpis: Kpis }) {
  const perDet = Object.entries(kpis.flags_per_detector)
    .map(([k, n]) => `${DETECTOR_LABEL[k] ?? k}: ${n}`)
    .join(" · ");
  return (
    <div className="flex flex-wrap gap-3">
      <Kpi label="Total spend" value={usd(kpis.total_spend)} />
      <Kpi label="Services" value={String(kpis.n_services)} />
      <Kpi label="Anomalies flagged" value={String(kpis.total_flags)} hint={perDet} />
      <Kpi label="Consensus alerts" value={String(kpis.consensus_alerts)} hint="≥2 detectors" />
      <Kpi
        label={`$ savable (${kpis.best_detector ?? "—"})`}
        value={usd(kpis.savable_usd)}
        hint={`${Math.round(kpis.leak_ratio * 100)}% of leak`}
        accent
      />
    </div>
  );
}
