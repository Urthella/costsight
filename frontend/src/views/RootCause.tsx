import { Search } from "lucide-react";
import { useSnapshot } from "../hooks/useSnapshot";
import { Card, CardBody, SectionTitle, SeverityBadge } from "../components/ui";
import { DataTable } from "../components/DataTable";
import { usd } from "../lib/utils";

export default function RootCause() {
  const { data } = useSnapshot();
  if (!data) return null;
  return (
    <div>
      <SectionTitle
        icon={Search}
        title="Root-cause attribution"
        subtitle="For each alert: which CUR dimension (region / usage-type / tag) drove the spend above baseline."
      />
      <Card>
        <CardBody>
          <DataTable
            rows={data.attribution}
            maxRows={50}
            empty="No alerts to attribute."
            columns={[
              { key: "date", label: "Date" },
              { key: "service", label: "Service" },
              {
                key: "severity",
                label: "Severity",
                render: (r) => <SeverityBadge severity={r.severity as string} />,
              },
              {
                key: "delta",
                label: "Δ vs baseline",
                align: "right",
                render: (r) => usd(r.delta as number),
              },
              { key: "top_dimension", label: "Driver" },
              { key: "top_value", label: "Value" },
              {
                key: "top_value_share",
                label: "Share",
                align: "right",
                render: (r) => `${Math.round((r.top_value_share as number) * 100)}%`,
              },
            ]}
          />
        </CardBody>
      </Card>
    </div>
  );
}
