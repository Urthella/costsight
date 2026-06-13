import { createContext, useContext, useState, type ReactNode } from "react";
import type { DashboardParams } from "../types";

interface Ctx {
  params: DashboardParams;
  setParams: (patch: Partial<DashboardParams>) => void;
}

const ParamsContext = createContext<Ctx | null>(null);

export function ParamsProvider({ children }: { children: ReactNode }) {
  const [params, setState] = useState<DashboardParams>({
    scenario: "default",
    nDays: 90,
    seed: 42,
  });
  const setParams = (patch: Partial<DashboardParams>) =>
    setState((s) => ({ ...s, ...patch }));
  return (
    <ParamsContext.Provider value={{ params, setParams }}>
      {children}
    </ParamsContext.Provider>
  );
}

// eslint-disable-next-line react-refresh/only-export-components
export function useDashboardParams(): Ctx {
  const ctx = useContext(ParamsContext);
  if (!ctx) throw new Error("useDashboardParams must be inside ParamsProvider");
  return ctx;
}
