import { Tag } from "lucide-react";
import { useSnapshot } from "../hooks/useSnapshot";
import { Card, CardBody, SectionTitle } from "../components/ui";
import { usd } from "../lib/utils";

export default function Tagging() {
  const { data } = useSnapshot();
  if (!data) return null;
  const t = data.tagging;
  const coverage = (t.coverage ?? []) as {
    tag: string;
    covered_pct: number;
    untagged_usd: number;
  }[];

  return (
    <div>
      <SectionTitle
        icon={Tag}
        title="Tag governance"
        subtitle="Untagged spend has no owner → no chargeback. This quantifies tag debt and where it lives."
      />

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        {coverage.map((c) => (
          <Card key={c.tag}>
            <CardBody>
              <div className="text-xs text-muted-foreground">{c.tag}</div>
              <div className="mt-1 text-3xl font-semibold">
                {c.covered_pct.toFixed(0)}%{" "}
                <span className="text-base font-normal text-muted-foreground">covered</span>
              </div>
              <div
                className={
                  "mt-1 text-xs " +
                  (c.untagged_usd > 0 ? "text-high" : "text-muted-foreground")
                }
              >
                {c.untagged_usd > 0
                  ? `${usd(c.untagged_usd)} untagged`
                  : "fully tagged"}
              </div>
            </CardBody>
          </Card>
        ))}
      </div>

      <Card className="mt-4">
        <CardBody>
          <div className="mb-1 text-sm font-medium">Policy-as-code stub (AWS Config rule)</div>
          <pre className="overflow-x-auto rounded-md bg-muted/60 p-3 text-xs">
            {t.policy_yaml}
          </pre>
        </CardBody>
      </Card>
    </div>
  );
}
