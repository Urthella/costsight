import { useRef } from "react";
import { NavLink } from "react-router-dom";
import { motion } from "framer-motion";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Cloud, Upload, X } from "lucide-react";
import { NAV } from "../nav";
import { getScenarios, postUpload } from "../lib/api";
import { useDashboardParams } from "../state/params";
import { cn } from "../lib/utils";

export function Sidebar() {
  const { params, setParams, uploaded, uploadName, setUploaded } = useDashboardParams();
  const fileRef = useRef<HTMLInputElement>(null);
  const { data: scenarios } = useQuery({
    queryKey: ["scenarios"],
    queryFn: getScenarios,
    staleTime: Infinity,
  });
  const upload = useMutation({
    mutationFn: postUpload,
    onSuccess: (snap, file) => setUploaded(snap, file.name),
  });

  return (
    <aside className="flex h-full w-64 shrink-0 flex-col border-r border-border bg-card">
      <div
        data-tour="brand"
        className="flex items-center gap-2 border-b border-border px-4 py-4"
      >
        <Cloud className="text-primary" size={24} />
        <div>
          <div className="text-sm font-semibold leading-tight">costsight</div>
          <div className="text-xs text-muted-foreground">Cloud cost anomalies</div>
        </div>
      </div>

      <nav data-tour="nav" className="flex-1 overflow-y-auto px-2 py-3">
        {NAV.map((group) => (
          <div
            key={group.label}
            data-tour={`nav-${group.label.toLowerCase().replace(/[^a-z]+/g, "-")}`}
            className="mb-3"
          >
            <div className="flex items-center gap-1.5 px-2 pb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              <group.icon size={13} />
              {group.label}
            </div>
            {group.items.map((it) => (
              <NavLink
                key={it.key}
                to={it.path}
                end={it.path === "/"}
                data-tour={it.key === "threed" ? "threed" : undefined}
                className="relative flex items-center gap-2 rounded-md px-2 py-1.5 text-sm hover:bg-muted/60"
              >
                {({ isActive }) => (
                  <>
                    {isActive && (
                      <motion.span
                        layoutId="nav-pill"
                        className="absolute inset-0 rounded-md bg-primary/10"
                        transition={{ type: "spring", stiffness: 400, damping: 32 }}
                      />
                    )}
                    <it.icon
                      size={16}
                      className={cn("relative z-10", isActive && "text-primary")}
                    />
                    <span
                      className={cn(
                        "relative z-10",
                        isActive ? "font-medium text-primary" : "text-foreground/80",
                      )}
                    >
                      {it.label}
                    </span>
                  </>
                )}
              </NavLink>
            ))}
          </div>
        ))}
      </nav>

      <div className="space-y-3 border-t border-border px-3 py-3 text-sm">
        <div data-tour="upload" className="space-y-3">
          <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Data source
          </div>

          <input
            ref={fileRef}
            type="file"
            accept=".csv"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) upload.mutate(f);
              e.target.value = "";
            }}
          />

          {uploaded ? (
            <div className="rounded-md border border-primary/40 bg-primary/5 p-2">
              <div className="flex items-center gap-1.5 text-xs font-medium text-primary">
                <Upload size={13} />
                <span className="truncate">{uploadName ?? "uploaded CUR"}</span>
              </div>
              <div className="mt-1 text-[11px] text-muted-foreground">
                Real data - Detector comparison stays blank (no labels).
              </div>
              <button
                onClick={() => setUploaded(null)}
                className="mt-2 flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
              >
                <X size={12} /> Back to synthetic
              </button>
            </div>
          ) : (
            <button
              onClick={() => fileRef.current?.click()}
              disabled={upload.isPending}
              className="flex w-full items-center justify-center gap-1.5 rounded-md border border-border bg-background px-2 py-1.5 text-xs font-medium hover:bg-muted disabled:opacity-50"
            >
              <Upload size={13} />
              {upload.isPending ? "Parsing…" : "Upload AWS CUR (.csv)"}
            </button>
          )}
          {upload.isError && (
            <div className="text-[11px] text-high">{String((upload.error as Error).message)}</div>
          )}
          {!uploaded && (
            <div className="text-[11px] leading-snug text-muted-foreground">
              No file handy? The bundled{" "}
              <code className="rounded bg-muted px-1">examples/cur_*.csv</code>{" "}
              files (60-90 days) work great - try cur_default_90d.csv.
            </div>
          )}
        </div>

        <div data-tour="synthetic" className="space-y-3">
          <div className="pt-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Synthetic data
          </div>
          <label data-tour="scenario" className="block">
            <span className="text-xs text-muted-foreground">Scenario</span>
            <select
              value={params.scenario}
              disabled={!!uploaded}
              onChange={(e) => setParams({ scenario: e.target.value })}
              className="mt-1 w-full rounded-md border border-border bg-background px-2 py-1.5 text-sm disabled:opacity-50"
            >
              {(scenarios ?? [{ key: "default", description: "" }]).map((s) => (
                <option key={s.key} value={s.key}>
                  {s.key}
                </option>
              ))}
            </select>
          </label>
          <label data-tour="days" className="block">
            <span className="text-xs text-muted-foreground">
              Days of history: {params.nDays}
            </span>
            <input
              type="range"
              min={30}
              max={180}
              step={15}
              value={params.nDays}
              disabled={!!uploaded}
              onChange={(e) => setParams({ nDays: Number(e.target.value) })}
              className="mt-1 w-full accent-[var(--color-primary)] disabled:opacity-50"
            />
          </label>
          <label data-tour="seed" className="block">
            <span className="text-xs text-muted-foreground">Random seed</span>
            <input
              type="number"
              min={0}
              value={params.seed}
              disabled={!!uploaded}
              onChange={(e) => setParams({ seed: Number(e.target.value) })}
              className="mt-1 w-full rounded-md border border-border bg-background px-2 py-1.5 text-sm disabled:opacity-50"
            />
          </label>
        </div>
      </div>
    </aside>
  );
}
