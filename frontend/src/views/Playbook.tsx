import { BookOpen } from "lucide-react";
import { useSnapshot } from "../hooks/useSnapshot";
import { Card, CardBody, SectionTitle } from "../components/ui";

interface Play {
  headline?: string;
  checks?: string;
  owner?: string;
  sla?: string;
  remediation?: string;
}

const TITLE: Record<string, string> = {
  point_spike: "Point spike",
  level_shift: "Level shift",
  gradual_drift: "Gradual drift",
  multi_detector_consensus: "Multi-detector consensus",
};

export default function Playbook() {
  const { data } = useSnapshot();
  if (!data) return null;
  const books = data.playbooks as Record<string, Play>;
  return (
    <div>
      <SectionTitle
        icon={BookOpen}
        title="Anomaly playbook"
        subtitle="Triage runbook per anomaly type — owner, SLA, and the exact checks to run."
      />
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        {Object.entries(books).map(([key, p]) => (
          <Card key={key}>
            <CardBody>
              <div className="flex items-center justify-between">
                <h3 className="font-semibold">{TITLE[key] ?? key}</h3>
                <div className="flex gap-2 text-xs text-muted-foreground">
                  {p.owner && <span className="rounded bg-muted px-2 py-0.5">{p.owner}</span>}
                  {p.sla && <span className="rounded bg-muted px-2 py-0.5">SLA {p.sla}</span>}
                </div>
              </div>
              {p.headline && <p className="mt-1 text-sm">{p.headline}</p>}
              {p.checks && (
                <pre className="mt-2 whitespace-pre-wrap rounded-md bg-muted/60 p-3 text-xs text-foreground/80">
                  {p.checks}
                </pre>
              )}
              {p.remediation && (
                <p className="mt-2 text-xs text-muted-foreground">
                  <span className="font-medium text-foreground">Remediation: </span>
                  {p.remediation}
                </p>
              )}
            </CardBody>
          </Card>
        ))}
      </div>
    </div>
  );
}
