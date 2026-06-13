import {
  LayoutDashboard,
  LineChart,
  Boxes,
  CalendarDays,
  Bell,
  Search,
  BarChart3,
  Network,
  Waves,
  TrendingUp,
  Wallet,
  Lightbulb,
  BookOpen,
  Leaf,
  Tag,
  Bot,
  Zap,
  SlidersHorizontal,
  Play,
  Table2,
  Radar,
  PiggyBank,
  FlaskConical,
  type LucideIcon,
} from "lucide-react";

export interface NavItem {
  key: string;
  label: string;
  path: string;
  icon: LucideIcon;
}
export interface NavGroup {
  label: string;
  icon: LucideIcon;
  items: NavItem[];
}

const item = (key: string, label: string, icon: LucideIcon): NavItem => ({
  key,
  label,
  icon,
  path: key === "summary" ? "/" : `/${key}`,
});

export const NAV: NavGroup[] = [
  {
    label: "Overview",
    icon: LayoutDashboard,
    items: [
      item("summary", "Summary", LayoutDashboard),
      item("trend", "Cost trend", LineChart),
      item("calendar", "Calendar", CalendarDays),
      item("threed", "3D explorer", Boxes),
    ],
  },
  {
    label: "Detection",
    icon: Radar,
    items: [
      item("alerts", "Alert log", Bell),
      item("rootcause", "Root-cause", Search),
      item("comparison", "Detector comparison", BarChart3),
      item("incidents", "Incidents", Network),
      item("drift", "Drift", Waves),
    ],
  },
  {
    label: "FinOps",
    icon: PiggyBank,
    items: [
      item("forecast", "Forecast", TrendingUp),
      item("budget", "Budget", Wallet),
      item("reco", "Recommendations", Lightbulb),
      item("playbook", "Playbook", BookOpen),
    ],
  },
  {
    label: "Sustainability",
    icon: Leaf,
    items: [
      item("carbon", "Carbon", Leaf),
      item("tagging", "Tagging", Tag),
    ],
  },
  {
    label: "Lab & Data",
    icon: FlaskConical,
    items: [
      item("ai", "AI Explain", Bot),
      item("perf", "Perf", Zap),
      item("lab", "Lab", SlidersHorizontal),
      item("replay", "Replay", Play),
      item("raw", "Raw data", Table2),
    ],
  },
];

export const ALL_ITEMS: NavItem[] = NAV.flatMap((g) => g.items);
