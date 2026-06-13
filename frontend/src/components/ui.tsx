import type { ReactNode } from "react";
import { cn } from "../lib/utils";

export function Card({
  className,
  children,
}: {
  className?: string;
  children: ReactNode;
}) {
  return (
    <div
      className={cn(
        "rounded-xl border border-border bg-card shadow-sm",
        className,
      )}
    >
      {children}
    </div>
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
