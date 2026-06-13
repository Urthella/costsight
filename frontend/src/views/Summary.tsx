import { LayoutDashboard } from "lucide-react";
import { useSnapshot } from "../hooks/useSnapshot";
import { Card, CardBody, SectionTitle, SeverityBadge } from "../components/ui";
import Plot, { PLOT_CONFIG, PLOT_LAYOUT_BASE } from "../lib/plot";
import { usd } from "../lib/utils";
import type { Alert } from "../types";

interface Incident {
  date: string;
  service: string;
  severity: string;
  cost: number;
  detectors: number;
  score: number;
}

function dedupeIncidents(alerts: Alert[]): Incident[] {
  const map = new Map<string, Incident>();
  for (const a of alerts) {
    const k = `${a.date}|${a.service}`;
    const cur = map.get(k);
    if (!cur) {
      map.set(k, {
        date: a.date,
        service: a.service,
        severity: a.severity,
        cost: a.cost,
        detectors: 1,
        score: a.severity_score,
      });
    } else {
      cur.detectors += 1;
      cur.cost = Math.max(cur.cost, a.cost);
      if (a.severity_score > cur.score) {
        cur.score = a.severity_score;
        cur.severity = a.severity;
      }
    }
  }
  return [...map.values()].sort((a, b) => b.score - a.score);
}

export default function Summary() {
  const { data } = useSnapshot();
  if (!data) return null;

  const incidents = dedupeIncidents(data.alerts);
  const top = incidents.slice(0, 8);
  const mix = { HIGH: 0, MEDIUM: 0, LOW: 0 } as Record<string, number>;
  for (const i of incidents) mix[i.severity] = (mix[i.severity] ?? 0) + 1;
  const rec = data.recommendations[0];

  return (
    <div>
      <SectionTitle
        icon={LayoutDashboard}
        title="Executive summary"
        subtitle={`${data.meta.scenario} · ${data.meta.dataset_days} days · ${data.meta.n_services} services`}
      />

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <Card>
          <CardBody>
            <div className="text-xs text-muted-foreground">Distinct anomalies</div>
            <div className="mt-1 text-3xl font-semibold">{incidents.length}</div>
            <div className="mt-1 text-xs text-muted-foreground">
              unique (date, service) across all detectors
            </div>
          </CardBody>
        </Card>
        <Card>
          <CardBody>
            <div className="text-xs text-muted-foreground">Carbon footprint</div>
            <div className="mt-1 text-3xl font-semibold">
              {data.carbon.kg_co2?.toLocaleString("en-US", {
                maximumFractionDigits: 0,
              })}{" "}
              <span className="text-base font-normal">kgCO₂e</span>
            </div>
            <div className="mt-1 text-xs text-muted-foreground">
              ≈ {data.carbon.km_driven_equiv?.toLocaleString("en-US", {
                maximumFractionDigits: 0,
              })}{" "}
              km driven
            </div>
          </CardBody>
        </Card>
        <Card>
          <CardBody>
            <div className="text-xs text-muted-foreground">Top savings</div>
            <div className="mt-1 text-3xl font-semibold text-primary">
              {rec ? `${usd(rec.impact_usd_per_month)}/mo` : "—"}
            </div>
            <div className="mt-1 text-xs text-muted-foreground">
              {rec ? `${rec.category} · ${rec.service}` : "no candidates"}
            </div>
          </CardBody>
        </Card>
      </div>

      <div className="mt-4 grid grid-cols-1 gap-3 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardBody>
            <div className="mb-2 text-sm font-medium">Highest-severity incidents</div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-left text-muted-foreground">
                    <th className="py-1.5 pr-4 font-medium">Date</th>
                    <th className="py-1.5 pr-4 font-medium">Service</th>
                    <th className="py-1.5 pr-4 font-medium">Severity</th>
                    <th className="py-1.5 pr-4 text-right font-medium">Cost</th>
                    <th className="py-1.5 text-right font-medium"># det.</th>
                  </tr>
                </thead>
                <tbody>
                  {top.map((i) => (
                    <tr key={`${i.date}-${i.service}`} className="border-b border-border/60">
                      <td className="py-1.5 pr-4">{i.date}</td>
                      <td className="py-1.5 pr-4">{i.service}</td>
                      <td className="py-1.5 pr-4">
                        <SeverityBadge severity={i.severity} />
                      </td>
                      <td className="py-1.5 pr-4 text-right">{usd(i.cost)}</td>
                      <td className="py-1.5 text-right">{i.detectors}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardBody>
        </Card>
        <Card>
          <CardBody>
            <div className="mb-2 text-sm font-medium">Severity mix</div>
            <Plot
              data={[
                {
                  type: "pie",
                  hole: 0.55,
                  labels: ["HIGH", "MEDIUM", "LOW"],
                  values: [mix.HIGH, mix.MEDIUM, mix.LOW],
                  marker: { colors: ["#dc2626", "#d97706", "#2563eb"] },
                  textinfo: "label+percent",
                },
              ]}
              layout={{ ...PLOT_LAYOUT_BASE, height: 260, showlegend: false, margin: { t: 10, r: 10, b: 10, l: 10 } }}
              config={PLOT_CONFIG}
              useResizeHandler
              style={{ width: "100%" }}
            />
          </CardBody>
        </Card>
      </div>

      {rec && (
        <Card className="mt-4">
          <CardBody>
            <div className="text-sm font-medium text-primary">
              Recommended next action
            </div>
            <div className="mt-1 text-sm">{rec.action}</div>
            <div className="mt-1 text-xs italic text-muted-foreground">
              {rec.rationale} (confidence: {rec.confidence})
            </div>
          </CardBody>
        </Card>
      )}
    </div>
  );
}
