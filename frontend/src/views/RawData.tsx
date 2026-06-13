import { useState } from "react";
import { Table2 } from "lucide-react";
import { useSnapshot } from "../hooks/useSnapshot";
import { Card, CardBody, SectionTitle } from "../components/ui";
import { DataTable } from "../components/DataTable";
import { usd } from "../lib/utils";

export default function RawData() {
  const { data } = useSnapshot();
  const [tab, setTab] = useState<"series" | "truth">("series");
  if (!data) return null;

  return (
    <div>
      <SectionTitle icon={Table2} title="Raw data" subtitle="The underlying long-format records." />
      <div className="mb-3 flex gap-2">
        {(["series", "truth"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={
              "rounded-md px-3 py-1 text-sm transition-colors " +
              (tab === t ? "bg-primary/10 font-medium text-primary" : "text-muted-foreground hover:bg-muted")
            }
          >
            {t === "series" ? "Per-service cost" : "Ground-truth labels"}
          </button>
        ))}
      </div>
      <Card>
        <CardBody>
          {tab === "series" ? (
            <DataTable
              rows={data.series as unknown as Record<string, unknown>[]}
              maxRows={100}
              columns={[
                { key: "date", label: "Date" },
                { key: "service", label: "Service" },
                { key: "cost", label: "Cost", align: "right", render: (r) => usd(r.cost as number, 2) },
              ]}
            />
          ) : (
            <DataTable
              rows={data.ground_truth.filter((r) => (r as { is_anomaly?: boolean }).is_anomaly)}
              maxRows={100}
              empty="No ground-truth anomalies in this scenario."
              columns={[
                { key: "date", label: "Date" },
                { key: "service", label: "Service" },
                { key: "anomaly_type", label: "Type" },
              ]}
            />
          )}
        </CardBody>
      </Card>
    </div>
  );
}
