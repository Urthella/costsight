import { createContext, useContext, useState, type ReactNode } from "react";
import type { DashboardParams, Snapshot } from "../types";

interface Ctx {
  params: DashboardParams;
  setParams: (patch: Partial<DashboardParams>) => void;
  // When the user uploads a real CUR, its snapshot overrides the synthetic one.
  uploaded: Snapshot | null;
  uploadName: string | null;
  setUploaded: (snap: Snapshot | null, name?: string | null) => void;
}

const ParamsContext = createContext<Ctx | null>(null);

export function ParamsProvider({ children }: { children: ReactNode }) {
  const [params, setState] = useState<DashboardParams>({
    scenario: "default",
    nDays: 90,
    seed: 42,
  });
  const [uploaded, setUp] = useState<Snapshot | null>(null);
  const [uploadName, setUploadName] = useState<string | null>(null);

  const setParams = (patch: Partial<DashboardParams>) =>
    setState((s) => ({ ...s, ...patch }));
  const setUploaded = (snap: Snapshot | null, name: string | null = null) => {
    setUp(snap);
    setUploadName(name);
  };

  return (
    <ParamsContext.Provider value={{ params, setParams, uploaded, uploadName, setUploaded }}>
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
