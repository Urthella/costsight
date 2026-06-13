import { useState } from "react";
import { Bot } from "lucide-react";
import { useMutation } from "@tanstack/react-query";
import { postExplain } from "../lib/api";
import { useSnapshot } from "../hooks/useSnapshot";
import { Card, CardBody, SectionTitle, SeverityBadge } from "../components/ui";
import { usd } from "../lib/utils";

export default function AIExplain() {
  const { data } = useSnapshot();
  const [idx, setIdx] = useState(0);
  const mutation = useMutation({ mutationFn: postExplain });
  if (!data) return null;

  // De-duplicate alerts to (date, service), highest severity first.
  const seen = new Set<string>();
  const alerts = data.alerts.filter((a) => {
    const k = `${a.date}|${a.service}`;
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
  const alert = alerts[idx];
  const attr = data.attribution.find(
    (r) => (r as { date: string }).date === alert?.date && (r as { service: string }).service === alert?.service,
  ) as { top_dimension?: string; top_value?: string } | undefined;

  return (
    <div>
      <SectionTitle
        icon={Bot}
        title="AI-powered root-cause explanation"
        subtitle="Claude turns an alert + its attribution into a plain-English root-cause note (deterministic template without an API key)."
      />
      {alerts.length === 0 ? (
        <Card><CardBody><div className="py-8 text-center text-sm text-muted-foreground">No alerts to explain.</div></CardBody></Card>
      ) : (
        <Card><CardBody>
          <div className="flex flex-wrap items-center gap-3">
            <select
              value={idx}
              onChange={(e) => { setIdx(Number(e.target.value)); mutation.reset(); }}
              className="rounded-md border border-border bg-card px-3 py-1.5 text-sm"
            >
              {alerts.slice(0, 50).map((a, i) => (
                <option key={`${a.date}-${a.service}`} value={i}>
                  {a.date} · {a.service} · {a.severity}
                </option>
              ))}
            </select>
            <SeverityBadge severity={alert.severity} />
            <span className="text-sm text-muted-foreground">{usd(alert.cost)}</span>
            <button
              onClick={() =>
                mutation.mutate({
                  service: alert.service,
                  date: alert.date,
                  severity: alert.severity,
                  cost: alert.cost,
                  flagged_by: alert.detector,
                  top_dimension: attr?.top_dimension ?? "",
                  top_value: attr?.top_value ?? "",
                })
              }
              disabled={mutation.isPending}
              className="ml-auto rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground disabled:opacity-50"
            >
              {mutation.isPending ? "Explaining…" : "Explain this alert"}
            </button>
          </div>

          {mutation.isError && (
            <div className="mt-4 text-sm text-high">Couldn't generate an explanation.</div>
          )}
          {mutation.data && (
            <div className="mt-4 whitespace-pre-wrap rounded-md border border-border bg-muted/50 p-4 text-sm">
              {mutation.data}
            </div>
          )}
        </CardBody></Card>
      )}
    </div>
  );
}
