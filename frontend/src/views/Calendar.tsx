import { CalendarDays } from "lucide-react";
import { useSnapshot } from "../hooks/useSnapshot";
import { Card, CardBody, SectionTitle } from "../components/ui";
import Plot, { PLOT_CONFIG, PLOT_LAYOUT_BASE } from "../lib/plot";

export default function Calendar() {
  const { data } = useSnapshot();
  if (!data) return null;

  const services = data.meta.services;
  const dates = [...new Set(data.series.map((s) => s.date))].sort();
  const lookup = new Map<string, number>();
  for (const s of data.series) lookup.set(`${s.service}|${s.date}`, s.cost);
  const z = services.map((svc) => dates.map((d) => lookup.get(`${svc}|${d}`) ?? 0));

  return (
    <div>
      <SectionTitle
        icon={CalendarDays}
        title="Cost calendar heatmap"
        subtitle="One cell per (service, day). Colour intensity = daily cost — drift and weekend seasonality jump out."
      />
      <Card>
        <CardBody>
          <Plot
            data={[
              {
                type: "heatmap",
                x: dates,
                y: services,
                z,
                colorscale: "YlOrRd",
                colorbar: { title: "$" },
              },
            ]}
            layout={{
              ...PLOT_LAYOUT_BASE,
              height: 420,
              margin: { t: 20, r: 16, b: 60, l: 90 },
              xaxis: { title: "Date" },
              yaxis: { title: "Service" },
            }}
            config={PLOT_CONFIG}
            useResizeHandler
            style={{ width: "100%" }}
          />
        </CardBody>
      </Card>
    </div>
  );
}
