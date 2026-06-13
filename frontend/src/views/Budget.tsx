import { useState } from "react";
import { Wallet } from "lucide-react";
import { useSnapshot } from "../hooks/useSnapshot";
import { Card, CardBody, SectionTitle } from "../components/ui";
import Plot, { PLOT_CONFIG, PLOT_LAYOUT_BASE } from "../lib/plot";
import { usd } from "../lib/utils";

export default function Budget() {
  const { data } = useSnapshot();
  const [budget, setBudget] = useState<number | null>(null);
  if (!data) return null;

  const projected = (data.projected_monthly as { projected_monthly: number }[]).reduce(
    (s, r) => s + (r.projected_monthly ?? 0),
    0,
  );
  const fallback = Math.max(1000, Math.ceil(projected / 1000) * 1000);
  const monthlyBudget = budget ?? fallback;
  const over = projected - monthlyBudget;
  const pct = monthlyBudget > 0 ? (projected / monthlyBudget) * 100 : 0;

  return (
    <div>
      <SectionTitle
        icon={Wallet}
        title="What-if budget tracker"
        subtitle="Set a monthly budget and compare it against the forecast-driven projection."
      />
      <Card>
        <CardBody>
          <label className="block max-w-xs">
            <span className="text-xs text-muted-foreground">Monthly budget ($)</span>
            <input
              type="number"
              min={0}
              step={500}
              value={monthlyBudget}
              onChange={(e) => setBudget(Number(e.target.value))}
              className="mt-1 w-full rounded-md border border-border bg-background px-2 py-1.5 text-sm"
            />
          </label>
          <div className="mt-4 grid grid-cols-3 gap-3">
            <div><div className="text-xs text-muted-foreground">Projected / month</div><div className="mt-1 text-2xl font-semibold">{usd(projected)}</div></div>
            <div><div className="text-xs text-muted-foreground">Budget</div><div className="mt-1 text-2xl font-semibold">{usd(monthlyBudget)}</div></div>
            <div>
              <div className="text-xs text-muted-foreground">{over >= 0 ? "Over budget" : "Headroom"}</div>
              <div className={"mt-1 text-2xl font-semibold " + (over >= 0 ? "text-high" : "text-primary")}>
                {usd(Math.abs(over))}
              </div>
            </div>
          </div>
          <div className="mt-4">
            <Plot
              data={[
                { type: "bar", orientation: "h", x: [projected], y: ["Projected"], marker: { color: over >= 0 ? "#dc2626" : "#1e40af" }, name: "Projected" },
                { type: "bar", orientation: "h", x: [monthlyBudget], y: ["Budget"], marker: { color: "#94a3b8" }, name: "Budget" },
              ]}
              layout={{ ...PLOT_LAYOUT_BASE, height: 200, showlegend: false, xaxis: { title: "$ / month" }, margin: { t: 10, r: 16, b: 40, l: 90 } }}
              config={PLOT_CONFIG} useResizeHandler style={{ width: "100%" }}
            />
          </div>
          <p className="mt-2 text-sm text-muted-foreground">
            Forecast burns <span className="font-medium text-foreground">{pct.toFixed(0)}%</span> of the budget.
          </p>
        </CardBody>
      </Card>
    </div>
  );
}
