import { Leaf } from "lucide-react";
import { useSnapshot } from "../hooks/useSnapshot";
import { Card, CardBody, SectionTitle } from "../components/ui";
import Plot, { PLOT_CONFIG, PLOT_LAYOUT_BASE } from "../lib/plot";
import { usd } from "../lib/utils";

const n = (v: number) => v.toLocaleString("en-US", { maximumFractionDigits: 0 });

export default function Carbon() {
  const { data } = useSnapshot();
  if (!data) return null;
  const c = data.carbon;
  const byService = (c.by_service ?? []) as { service: string; kg_co2: number }[];
  const byRegion = (c.by_region ?? []) as { region: string; kg_co2: number }[];

  return (
    <div>
      <SectionTitle
        icon={Leaf}
        title="Carbon footprint of cloud spend"
        subtitle="USD → kWh (per-service intensity) → kgCO₂e (per-region grid carbon). Snapshot dated 2024-12-15."
      />
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Card><CardBody><div className="text-xs text-muted-foreground">CO₂e</div><div className="mt-1 text-2xl font-semibold">{n(c.kg_co2)} <span className="text-sm font-normal">kg</span></div></CardBody></Card>
        <Card><CardBody><div className="text-xs text-muted-foreground">≈ km driven</div><div className="mt-1 text-2xl font-semibold">{n(c.km_driven_equiv)}</div></CardBody></Card>
        <Card><CardBody><div className="text-xs text-muted-foreground">≈ tree-years</div><div className="mt-1 text-2xl font-semibold">{c.tree_years_equiv.toFixed(1)}</div></CardBody></Card>
        <Card><CardBody><div className="text-xs text-muted-foreground">Spend basis</div><div className="mt-1 text-2xl font-semibold">{usd(c.cost_usd)}</div></CardBody></Card>
      </div>

      <div className="mt-4 grid grid-cols-1 gap-3 lg:grid-cols-2">
        <Card><CardBody>
          <div className="mb-2 text-sm font-medium">CO₂e by service</div>
          <Plot
            data={[{ x: byService.map((s) => s.service), y: byService.map((s) => s.kg_co2), type: "bar", marker: { color: "#10b981" } }]}
            layout={{ ...PLOT_LAYOUT_BASE, height: 320, yaxis: { title: "kgCO₂e" } }}
            config={PLOT_CONFIG} useResizeHandler style={{ width: "100%" }}
          />
        </CardBody></Card>
        <Card><CardBody>
          <div className="mb-2 text-sm font-medium">CO₂e by region</div>
          <Plot
            data={[{ x: byRegion.map((s) => s.region), y: byRegion.map((s) => s.kg_co2), type: "bar", marker: { color: "#0ea5e9" } }]}
            layout={{ ...PLOT_LAYOUT_BASE, height: 320, yaxis: { title: "kgCO₂e" } }}
            config={PLOT_CONFIG} useResizeHandler style={{ width: "100%" }}
          />
        </CardBody></Card>
      </div>
    </div>
  );
}
