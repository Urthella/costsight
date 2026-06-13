import { useState } from "react";
import { Bell } from "lucide-react";
import { useSnapshot } from "../hooks/useSnapshot";
import { Card, CardBody, SectionTitle, SeverityBadge } from "../components/ui";
import { DataTable } from "../components/DataTable";
import { usd, DETECTOR_LABEL } from "../lib/utils";
import type { Severity } from "../types";

const BANDS: Severity[] = ["HIGH", "MEDIUM", "LOW"];

export default function AlertLog() {
  const { data } = useSnapshot();
  const [bands, setBands] = useState<Severity[]>(["HIGH", "MEDIUM", "LOW"]);
  if (!data) return null;

  const rows = data.alerts.filter((a) => bands.includes(a.severity));

  return (
    <div>
      <SectionTitle
        icon={Bell}
        title="Alert log"
        subtitle="Severity = deviation × duration × $ impact, banded LOW / MEDIUM / HIGH."
      />
      <div className="mb-3 flex gap-2">
        {BANDS.map((b) => (
          <button
            key={b}
            onClick={() =>
              setBands((s) =>
                s.includes(b) ? s.filter((x) => x !== b) : [...s, b],
              )
            }
            className={
              "rounded-full border px-3 py-1 text-xs font-medium transition-colors " +
              (bands.includes(b)
                ? "border-primary bg-primary/10 text-primary"
                : "border-border text-muted-foreground")
            }
          >
            {b}
          </button>
        ))}
      </div>
      <Card>
        <CardBody>
          <DataTable
            rows={rows as unknown as Record<string, unknown>[]}
            columns={[
              { key: "date", label: "Date" },
              { key: "service", label: "Service" },
              {
                key: "severity",
                label: "Severity",
                render: (r) => <SeverityBadge severity={r.severity as string} />,
              },
              {
                key: "cost",
                label: "Cost",
                align: "right",
                render: (r) => usd(r.cost as number),
              },
              {
                key: "detector",
                label: "Detector",
                render: (r) => DETECTOR_LABEL[r.detector as string] ?? (r.detector as string),
              },
              {
                key: "score",
                label: "Score",
                align: "right",
                render: (r) => (r.score as number).toFixed(2),
              },
            ]}
          />
        </CardBody>
      </Card>
    </div>
  );
}
