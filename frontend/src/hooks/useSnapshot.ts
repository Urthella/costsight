import { useQuery } from "@tanstack/react-query";
import { getSnapshot } from "../lib/api";
import { useDashboardParams } from "../state/params";
import type { Snapshot } from "../types";

interface SnapshotState {
  data: Snapshot | undefined;
  isLoading: boolean;
  isError: boolean;
  error: unknown;
}

/** One cached fetch per (scenario, nDays, seed) shared across every view.
 *  An uploaded CUR snapshot, when present, overrides the synthetic fetch. */
export function useSnapshot(): SnapshotState {
  const { params, uploaded } = useDashboardParams();
  const q = useQuery({
    queryKey: ["snapshot", params.scenario, params.nDays, params.seed],
    queryFn: () => getSnapshot(params),
    staleTime: 5 * 60 * 1000,
    enabled: !uploaded,
  });
  if (uploaded) {
    return { data: uploaded, isLoading: false, isError: false, error: null };
  }
  return { data: q.data, isLoading: q.isLoading, isError: q.isError, error: q.error };
}
