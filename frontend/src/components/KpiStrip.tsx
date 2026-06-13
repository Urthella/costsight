import { useEffect } from "react";
import {
  motion,
  useMotionValue,
  useTransform,
  animate,
  useReducedMotion,
  type Variants,
} from "framer-motion";
import type { Kpis } from "../types";
import { usd, DETECTOR_LABEL } from "../lib/utils";

const intFmt = (n: number) => Math.round(n).toLocaleString("en-US");

function CountUp({
  value,
  format,
}: {
  value: number;
  format: (n: number) => string;
}) {
  const reduced = useReducedMotion();
  const mv = useMotionValue(reduced ? value : 0);
  const out = useTransform(mv, format);
  useEffect(() => {
    if (reduced) {
      mv.set(value);
      return;
    }
    const controls = animate(mv, value, { duration: 0.7, ease: "easeOut" });
    return () => controls.stop();
  }, [value, reduced, mv]);
  return <motion.span>{out}</motion.span>;
}

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.06 } },
} satisfies Variants;
const item = {
  hidden: { opacity: 0, y: 10 },
  show: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 260, damping: 24 } },
} satisfies Variants;

function Tile({
  label,
  value,
  hint,
  accent,
}: {
  label: string;
  value: React.ReactNode;
  hint?: string;
  accent?: boolean;
}) {
  return (
    <motion.div
      variants={item}
      className="flex-1 rounded-xl border border-border bg-card p-3 shadow-sm"
    >
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className={"mt-1 text-2xl font-semibold " + (accent ? "text-primary" : "")}>
        {value}
      </div>
      {hint && <div className="mt-0.5 text-xs text-muted-foreground">{hint}</div>}
    </motion.div>
  );
}

export function KpiStrip({ kpis }: { kpis: Kpis }) {
  const perDet = Object.entries(kpis.flags_per_detector)
    .map(([k, n]) => `${DETECTOR_LABEL[k] ?? k}: ${n}`)
    .join(" · ");
  return (
    <motion.div
      className="flex flex-wrap gap-3"
      variants={container}
      initial="hidden"
      animate="show"
    >
      <Tile label="Total spend" value={<CountUp value={kpis.total_spend} format={(v) => usd(v)} />} />
      <Tile label="Services" value={<CountUp value={kpis.n_services} format={intFmt} />} />
      <Tile
        label="Anomalies flagged"
        value={<CountUp value={kpis.total_flags} format={intFmt} />}
        hint={perDet}
      />
      <Tile
        label="Consensus alerts"
        value={<CountUp value={kpis.consensus_alerts} format={intFmt} />}
        hint="≥2 detectors"
      />
      <Tile
        label={`$ savable (${kpis.best_detector ?? "—"})`}
        value={<CountUp value={kpis.savable_usd} format={(v) => usd(v)} />}
        hint={`${Math.round(kpis.leak_ratio * 100)}% of leak`}
        accent
      />
    </motion.div>
  );
}
