import { useState } from "react";
import { Wallet } from "lucide-react";
import { useSnapshot } from "../hooks/useSnapshot";
import { Card, CardBody, SectionTitle, ModeToggle } from "../components/ui";
import Plot, { PLOT_CONFIG, PLOT_LAYOUT_BASE } from "../lib/plot";
import { bars3dTrace, barLabels, scene3d, PLOT3D_LAYOUT, type Bar3D } from "../lib/threed";
import { usd } from "../lib/utils";

interface ProjRow {
  service: string;
  projected_monthly: number;
}

export default function Budget() {
  const { data } = useSnapshot();
  const [budget, setBudget] = useState<number | null>(null);
  const [mode, setMode] = useState<"3d" | "2d">("3d");
  if (!data) return null;

  const proj = data.projected_monthly as unknown as ProjRow[];
  const projected = proj.reduce((s, r) => s + (r.projected_monthly ?? 0), 0);
  const fallback = Math.max(1000, Math.ceil(projected / 1000) * 1000);
  const monthlyBudget = budget ?? fallback;
  const over = projected - monthlyBudget;
  const pct = monthlyBudget > 0 ? (projected / monthlyBudget) * 100 : 0;

  // 3D: per-service projected spend; bars over their fair share turn red.
  const fair = proj.length ? monthlyBudget / proj.length : 0;
  const bars: Bar3D[] = proj.map((r, i) => ({
    x: i,
    y: 0,
    h: r.projected_monthly,
    color: r.projected_monthly > fair ? "#dc2626" : "#1e40af",
  }));

  return (
    <div>
      <SectionTitle
        icon={Wallet}
        title="What-if budget tracker"
        subtitle="Set a monthly budget and compare it against the forecast-driven projection. 3D shows per-service spend - red towers exceed an equal share of the budget."
      />
      <Card>
        <CardBody>
          <div className="flex items-end justify-between gap-3">
            <label className="block max-w-xs flex-1">
              <span className="text-xs text-muted-foreground">Monthly budget ($)</span>
              <input
                type="number" min={0} step={500} value={monthlyBudget}
                onChange={(e) => setBudget(Number(e.target.value))}
                className="mt-1 w-full rounded-md border border-border bg-background px-2 py-1.5 text-sm"
              />
            </label>
            <ModeToggle mode={mode} onChange={setMode} />
          </div>

          <div className="mt-4 grid grid-cols-3 gap-3">
            <div><div className="text-xs text-muted-foreground">Projected / month</div><div className="mt-1 text-2xl font-semibold">{usd(projected)}</div></div>
            <div><div className="text-xs text-muted-foreground">Budget</div><div className="mt-1 text-2xl font-semibold">{usd(monthlyBudget)}</div></div>
            <div>
              <div className="text-xs text-muted-foreground">{over >= 0 ? "Over budget" : "Headroom"}</div>
              <div className={"mt-1 text-2xl font-semibold " + (over >= 0 ? "text-high" : "text-primary")}>{usd(Math.abs(over))}</div>
            </div>
          </div>

          <div className="mt-4">
            {mode === "3d" ? (
              <Plot
                data={[bars3dTrace(bars, 0.38, 0.38), barLabels(bars, proj.map((r) => `${r.service}: ${usd(r.projected_monthly)}/mo`))]}
                layout={{ ...PLOT3D_LAYOUT, scene: scene3d(proj.map((r) => r.service), [""], "$ / month") }}
                config={PLOT_CONFIG} useResizeHandler style={{ width: "100%" }}
              />
            ) : (
              <Plot
                data={[
                  { type: "bar", orientation: "h", x: [projected], y: ["Projected"], marker: { color: over >= 0 ? "#dc2626" : "#1e40af" } },
                  { type: "bar", orientation: "h", x: [monthlyBudget], y: ["Budget"], marker: { color: "#94a3b8" } },
                ]}
                layout={{ ...PLOT_LAYOUT_BASE, height: 200, showlegend: false, xaxis: { title: "$ / month" }, margin: { t: 10, r: 16, b: 40, l: 90 } }}
                config={PLOT_CONFIG} useResizeHandler style={{ width: "100%" }}
              />
            )}
          </div>
          <p className="mt-2 text-sm text-muted-foreground">
            Forecast burns <span className="font-medium text-foreground">{pct.toFixed(0)}%</span> of the budget.
          </p>
        </CardBody>
      </Card>
    </div>
  );
}
