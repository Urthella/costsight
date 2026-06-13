import { useQuery } from "@tanstack/react-query";
import { getSnapshot } from "../lib/api";
import { useDashboardParams } from "../state/params";

/** One cached fetch per (scenario, nDays, seed) shared across every view. */
export function useSnapshot() {
  const { params } = useDashboardParams();
  return useQuery({
    queryKey: ["snapshot", params.scenario, params.nDays, params.seed],
    queryFn: () => getSnapshot(params),
    staleTime: 5 * 60 * 1000,
  });
}
