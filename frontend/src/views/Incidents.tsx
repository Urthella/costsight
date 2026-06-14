import { Network } from "lucide-react";
import { useSnapshot } from "../hooks/useSnapshot";
import { Card, CardBody, SectionTitle, SeverityBadge } from "../components/ui";
import { DataTable } from "../components/DataTable";

export default function Incidents() {
  const { data } = useSnapshot();
  if (!data) return null;
  return (
    <div>
      <SectionTitle
        icon={Network}
        title="Incidents"
        subtitle="DBSCAN groups close-in-time alerts into incidents - rows become triageable events."
      />
      <Card>
        <CardBody>
          <DataTable
            rows={data.incidents}
            empty="No multi-alert incidents - every alert is a singleton."
            columns={[
              { key: "incident_id", label: "ID" },
              { key: "n_alerts", label: "Alerts", align: "right" },
              { key: "first_date", label: "First" },
              { key: "last_date", label: "Last" },
              { key: "services", label: "Services" },
              {
                key: "max_severity",
                label: "Max severity",
                render: (r) => <SeverityBadge severity={r.max_severity as string} />,
              },
              { key: "summary", label: "Summary" },
            ]}
          />
        </CardBody>
      </Card>
    </div>
  );
}
