import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/** Tailwind-aware className joiner (the shadcn `cn` helper). */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

export const usd = (v: number, digits = 0): string =>
  `$${v.toLocaleString("en-US", { maximumFractionDigits: digits })}`;

export const SEVERITY_COLOR: Record<string, string> = {
  HIGH: "#dc2626",
  MEDIUM: "#d97706",
  LOW: "#2563eb",
};

export const DETECTOR_COLOR: Record<string, string> = {
  zscore: "#3b82f6",
  stl: "#f59e0b",
  iforest: "#a855f7",
  ensemble: "#10b981",
};

export const DETECTOR_LABEL: Record<string, string> = {
  zscore: "Z-Score",
  stl: "STL",
  iforest: "Isolation Forest",
  ensemble: "Ensemble",
};
