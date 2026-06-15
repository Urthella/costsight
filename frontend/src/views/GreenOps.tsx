import { Sprout } from "lucide-react";
import { useSnapshot } from "../hooks/useSnapshot";
import { Card, CardBody, SectionTitle } from "../components/ui";
import { DataTable } from "../components/DataTable";
import Plot, { PLOT_CONFIG, PLOT_LAYOUT_BASE } from "../lib/plot";

const usd = (n: number) => `$${Math.round(n).toLocaleString()}`;
const kg = (n: number) => `${Math.round(n).toLocaleString()} kg`;

function Stat({ label, value, sub, tone }: { label: string; value: string; sub?: string; tone?: "low" | "high" }) {
  const accent = tone === "high" ? "text-high" : tone === "low" ? "text-low" : "text-foreground";
  return (
    <Card>
      <CardBody className="p-3">
        <div className="text-xs text-muted-foreground">{label}</div>
        <div className={`mt-0.5 text-xl font-bold ${accent}`}>{value}</div>
        {sub && <div className="text-xs text-muted-foreground">{sub}</div>}
      </CardBody>
    </Card>
  );
}

export default function GreenOps() {
  const { data } = useSnapshot();
  if (!data) return null;
  const g = data.green_ops;
  if (!g || !g.savings || !g.inaction || (!g.savings.length && !g.inaction.horizons.length))
    return (
      <div>
        <SectionTitle icon={Sprout} title="GreenOps" />
        <Card><CardBody><div className="py-8 text-center text-sm text-muted-foreground">No savings or active leaks to translate into carbon right now.</div></CardBody></Card>
      </div>
    );

  const top = g.savings.slice(0, 8).reverse(); // horizontal bar: biggest on top
  const h30 = g.inaction.horizons.find((h) => h.days === 30);

  return (
    <div>
      <SectionTitle
        icon={Sprout}
        title="GreenOps - savings ranked by carbon, not just dollars"
        subtitle="A dollar saved in a coal grid (ap-south-1) is ~50x the CO2 of a dollar in a hydro grid (eu-north-1) - so the greenest fix is rarely the biggest-dollar fix. And every day an anomaly leaks burns both money and carbon."
      />

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Green savings / mo" value={usd(g.savings_total_usd)} sub={`${kg(g.savings_total_co2)} CO2`} tone="low" />
        <Stat label="CO2 avoidable / mo" value={kg(g.savings_total_co2)} sub={`= ${Math.round(g.savings_total_co2 / 21)} tree-years`} tone="low" />
        <Stat label="Leak rate / day" value={usd(g.inaction.daily_usd)} sub={`${g.inaction.daily_co2_kg.toFixed(1)} kg CO2 / day`} tone="high" />
        <Stat label="Cost of 30d inaction" value={h30 ? usd(h30.usd) : "-"} sub={h30 ? `${kg(h30.co2_kg)} CO2` : ""} tone="high" />
      </div>

      <Card className="mt-4">
        <CardBody>
          <div className="mb-1 text-sm font-medium">Greenest savings first (kgCO2 / month, $ on each bar)</div>
          <Plot
            data={[{
              type: "bar", orientation: "h",
              y: top.map((s) => `${s.service} · ${s.region}`),
              x: top.map((s) => s.co2_kg_per_month),
              marker: { color: "#16a34a" },
              text: top.map((s) => usd(s.usd_per_month) + "/mo"),
              textposition: "auto",
              hovertemplate: "%{y}<br>%{x:.0f} kg CO2/mo · %{text}<extra></extra>",
            }]}
            layout={{ ...PLOT_LAYOUT_BASE, height: 320, margin: { t: 16, r: 16, b: 36, l: 150 }, xaxis: { title: "kg CO2 / month" } }}
            config={PLOT_CONFIG} useResizeHandler style={{ width: "100%" }}
          />
        </CardBody>
      </Card>

      <Card className="mt-4">
        <CardBody>
          <DataTable
            rows={g.savings as unknown as Record<string, unknown>[]}
            columns={[
              { key: "category", label: "Category" },
              { key: "service", label: "Service" },
              { key: "region", label: "Region" },
              { key: "usd_per_month", label: "$/mo", align: "right", render: (r) => usd(r.usd_per_month as number) },
              { key: "co2_kg_per_month", label: "kgCO2/mo", align: "right", render: (r) => kg(r.co2_kg_per_month as number) },
              { key: "km_equiv", label: "km driving", align: "right", render: (r) => (r.km_equiv as number).toLocaleString() },
            ]}
          />
        </CardBody>
      </Card>

      <Card className="mt-4">
        <CardBody>
          <div className="mb-1 text-sm font-medium">Cost of inaction - if today's leaks stay unfixed</div>
          <Plot
            data={[
              { type: "bar", name: "$ wasted", x: g.inaction.horizons.map((h) => `${h.days} days`), y: g.inaction.horizons.map((h) => h.usd), marker: { color: "#dc2626" }, yaxis: "y", hovertemplate: "%{x}: $%{y:,.0f}<extra></extra>" },
              { type: "bar", name: "kg CO2", x: g.inaction.horizons.map((h) => `${h.days} days`), y: g.inaction.horizons.map((h) => h.co2_kg), marker: { color: "#0ea5e9" }, yaxis: "y2", hovertemplate: "%{x}: %{y:,.0f} kg<extra></extra>" },
            ]}
            layout={{
              ...PLOT_LAYOUT_BASE, height: 320, barmode: "group",
              margin: { t: 20, r: 56, b: 40, l: 60 },
              yaxis: { title: "$ wasted" },
              yaxis2: { title: "kg CO2", overlaying: "y", side: "right", showgrid: false },
            }}
            config={PLOT_CONFIG} useResizeHandler style={{ width: "100%" }}
          />
          {g.inaction.by_service.length > 0 && (
            <p className="mt-2 text-xs text-muted-foreground">
              Worst leaker: <span className="font-medium text-foreground">{g.inaction.by_service[0].service}</span> at {usd(g.inaction.by_service[0].daily_usd)}/day - {kg(g.inaction.by_service[0].co2_kg_30d)} CO2 over 30 days if left unfixed.
            </p>
          )}
        </CardBody>
      </Card>
    </div>
  );
}
