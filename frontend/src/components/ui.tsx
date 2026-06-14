import type { ReactNode } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { cn } from "../lib/utils";

// Every Card animates in (fade + lift) and lifts its shadow on hover - this is
// the "motion on every page" primitive. Honors prefers-reduced-motion.
export function Card({
  className,
  children,
  dataTour,
}: {
  className?: string;
  children: ReactNode;
  dataTour?: string;
}) {
  const reduced = useReducedMotion();
  return (
    <motion.div
      data-tour={dataTour}
      initial={reduced ? false : { opacity: 0, y: 10 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.15 }}
      whileHover={reduced ? undefined : { boxShadow: "0 10px 28px rgba(15,23,42,0.10)" }}
      transition={{ type: "spring", stiffness: 240, damping: 26 }}
      className={cn(
        "rounded-xl border border-border bg-card shadow-sm",
        className,
      )}
    >
      {children}
    </motion.div>
  );
}

export function CardBody({
  className,
  children,
}: {
  className?: string;
  children: ReactNode;
}) {
  return <div className={cn("p-4", className)}>{children}</div>;
}

export function SectionTitle({
  icon: Icon,
  title,
  subtitle,
}: {
  icon?: React.ComponentType<{ size?: number; className?: string }>;
  title: string;
  subtitle?: string;
}) {
  return (
    <div className="mb-4">
      <h2 className="flex items-center gap-2 text-xl font-semibold text-foreground">
        {Icon && <Icon size={22} className="text-primary" />}
        {title}
      </h2>
      {subtitle && (
        <p className="mt-1 text-sm text-muted-foreground">{subtitle}</p>
      )}
    </div>
  );
}

export function ModeToggle({
  mode,
  onChange,
}: {
  mode: "3d" | "2d";
  onChange: (m: "3d" | "2d") => void;
}) {
  return (
    <div className="inline-flex overflow-hidden rounded-md border border-border text-xs">
      {(["3d", "2d"] as const).map((m) => (
        <button
          key={m}
          onClick={() => onChange(m)}
          className={cn(
            "px-2.5 py-1 font-medium transition-colors",
            mode === m ? "bg-primary text-primary-foreground" : "bg-card text-muted-foreground hover:bg-muted",
          )}
        >
          {m.toUpperCase()}
        </button>
      ))}
    </div>
  );
}

export function KpiSkeleton() {
  return (
    <div className="flex flex-wrap gap-3">
      {[0, 1, 2, 3, 4].map((i) => (
        <div key={i} className="h-[88px] flex-1 animate-pulse rounded-xl bg-muted" />
      ))}
    </div>
  );
}

export function ViewSkeleton() {
  return (
    <div className="space-y-3">
      <div className="h-7 w-72 animate-pulse rounded bg-muted" />
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        {[0, 1, 2].map((i) => (
          <div key={i} className="h-24 animate-pulse rounded-xl bg-muted" />
        ))}
      </div>
      <div className="h-80 animate-pulse rounded-xl bg-muted" />
    </div>
  );
}

const SEV_CLASS: Record<string, string> = {
  HIGH: "bg-high/10 text-high border-high/30",
  MEDIUM: "bg-medium/10 text-medium border-medium/30",
  LOW: "bg-low/10 text-low border-low/30",
};

export function SeverityBadge({ severity }: { severity: string }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium",
        SEV_CLASS[severity] ?? "bg-muted text-muted-foreground border-border",
      )}
    >
      {severity}
    </span>
  );
}
