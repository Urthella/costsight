import { NavLink } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Cloud } from "lucide-react";
import { NAV } from "../nav";
import { getScenarios } from "../lib/api";
import { useDashboardParams } from "../state/params";
import { cn } from "../lib/utils";

export function Sidebar() {
  const { params, setParams } = useDashboardParams();
  const { data: scenarios } = useQuery({
    queryKey: ["scenarios"],
    queryFn: getScenarios,
    staleTime: Infinity,
  });

  return (
    <aside className="flex h-full w-64 shrink-0 flex-col border-r border-border bg-card">
      <div className="flex items-center gap-2 border-b border-border px-4 py-4">
        <Cloud className="text-primary" size={24} />
        <div>
          <div className="text-sm font-semibold leading-tight">costsight</div>
          <div className="text-xs text-muted-foreground">Cloud cost anomalies</div>
        </div>
      </div>

      <nav className="flex-1 overflow-y-auto px-2 py-3">
        {NAV.map((group) => (
          <div key={group.label} className="mb-3">
            <div className="flex items-center gap-1.5 px-2 pb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              <group.icon size={13} />
              {group.label}
            </div>
            {group.items.map((it) => (
              <NavLink
                key={it.key}
                to={it.path}
                end={it.path === "/"}
                className={({ isActive }) =>
                  cn(
                    "flex items-center gap-2 rounded-md px-2 py-1.5 text-sm transition-colors",
                    isActive
                      ? "bg-primary/10 font-medium text-primary"
                      : "text-foreground/80 hover:bg-muted",
                  )
                }
              >
                <it.icon size={16} />
                {it.label}
              </NavLink>
            ))}
          </div>
        ))}
      </nav>

      <div className="space-y-3 border-t border-border px-3 py-3 text-sm">
        <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Configuration
        </div>
        <label className="block">
          <span className="text-xs text-muted-foreground">Scenario</span>
          <select
            value={params.scenario}
            onChange={(e) => setParams({ scenario: e.target.value })}
            className="mt-1 w-full rounded-md border border-border bg-background px-2 py-1.5 text-sm"
          >
            {(scenarios ?? [{ key: "default", description: "" }]).map((s) => (
              <option key={s.key} value={s.key}>
                {s.key}
              </option>
            ))}
          </select>
        </label>
        <label className="block">
          <span className="text-xs text-muted-foreground">
            Days of history: {params.nDays}
          </span>
          <input
            type="range"
            min={30}
            max={180}
            step={15}
            value={params.nDays}
            onChange={(e) => setParams({ nDays: Number(e.target.value) })}
            className="mt-1 w-full accent-[var(--color-primary)]"
          />
        </label>
        <label className="block">
          <span className="text-xs text-muted-foreground">Random seed</span>
          <input
            type="number"
            min={0}
            value={params.seed}
            onChange={(e) => setParams({ seed: Number(e.target.value) })}
            className="mt-1 w-full rounded-md border border-border bg-background px-2 py-1.5 text-sm"
          />
        </label>
      </div>
    </aside>
  );
}
