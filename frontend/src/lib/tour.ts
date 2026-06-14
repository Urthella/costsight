import { driver } from "driver.js";
import "driver.js/dist/driver.css";

// Bump when the tour content changes so returning users see it once more.
const SEEN_KEY = "costsight_tour_v3";

/** A detailed, plain-language guided tour - assumes zero prior knowledge, so
 *  anyone (reviewer, teammate, first-time user) understands every part of the
 *  app. It walks the left control rail top-to-bottom first (brand, the five
 *  view groups, the data-source controls), then the main Summary content.
 *  Runs on the Summary view, where all the anchored elements exist. */
export function startTour(): void {
  const d = driver({
    showProgress: true,
    overlayOpacity: 0.6,
    nextBtnText: "Next →",
    prevBtnText: "← Back",
    doneBtnText: "Done",
    steps: [
      {
        popover: {
          title: "Welcome to costsight ☁️",
          description:
            "costsight watches your <b>cloud bill</b> and flags <b>cost anomalies</b> - days where a service costs far more than it should. It catches three kinds: a <b>point spike</b> (one bad day), a <b>level shift</b> (a permanent step up) and <b>gradual drift</b> (a slow creep). This tour explains every part of the screen. We'll start with the <b>left rail</b>, then the main dashboard.",
        },
      },

      // ---- Left rail: brand + the five view groups -----------------------
      {
        element: '[data-tour="brand"]',
        popover: {
          title: "The left rail is mission control",
          description:
            "This whole strip on the left does two jobs: the <b>top</b> is the menu of every view, and the <b>bottom</b> is where you choose the data (your real bill, or a built-in scenario). Everything on the right is just a <b>view of whatever you pick down here</b>.",
          side: "right",
          align: "start",
        },
      },
      {
        element: '[data-tour="nav"]',
        popover: {
          title: "The menu: every view in 5 groups",
          description:
            "All the analysis lives here, grouped into five sections - <b>Overview</b>, <b>Detection</b>, <b>FinOps</b>, <b>Sustainability</b> and <b>Lab & Data</b>. Click any row to open it. Let's go through what each group gives you.",
          side: "right",
          align: "start",
        },
      },
      {
        element: '[data-tour="nav-overview"]',
        popover: {
          title: "Group 1 · Overview",
          description:
            "The at-a-glance pages. <b>Summary</b> (this page) is the executive dashboard; <b>Cost trend</b> plots daily spend over time with anomaly markers (in 2D, or a 3D line forest); <b>Calendar</b> is a heatmap of daily spend per service (hot cells = expensive days, plus a 3D surface mode); <b>3D explorer</b> turns the whole cost history into a rotatable 3D landscape.",
          side: "right",
          align: "start",
        },
      },
      {
        element: '[data-tour="nav-detection"]',
        popover: {
          title: "Group 2 · Detection",
          description:
            "The core of the project. <b>Alert log</b> lists the severity-scored alerts, filterable by HIGH / MEDIUM / LOW; <b>Root-cause</b> tells you <i>which</i> region, usage type or team tag drove each alert; <b>Detector comparison</b> scores the four methods against each other on F1, precision and recall; <b>Incidents</b> clusters related alerts into single events; <b>Drift</b> catches slow baseline shifts a spike-detector would miss.",
          side: "right",
          align: "start",
        },
      },
      {
        element: '[data-tour="nav-finops"]',
        popover: {
          title: "Group 3 · FinOps",
          description:
            "Turning detection into money decisions. <b>Forecast</b> projects each service's spend forward with a confidence band; <b>Budget</b> compares each service against its fair share of the monthly budget; <b>Recommendations</b> proposes concrete savings actions with a dollar figure; <b>Playbook</b> is the triage reference (owner, SLA, checks and fix) for each anomaly type.",
          side: "right",
          align: "start",
        },
      },
      {
        element: '[data-tour="nav-sustainability"]',
        popover: {
          title: "Group 4 · Sustainability",
          description:
            "The green angle. <b>Carbon</b> translates dollars into kgCO₂e (and km driven) using per-service energy and per-region grid intensity; <b>Tagging</b> shows how much of your spend is properly tagged for cost allocation, and where the gaps are.",
          side: "right",
          align: "start",
        },
      },
      {
        element: '[data-tour="nav-lab-data"]',
        popover: {
          title: "Group 5 · Lab & Data",
          description:
            "The hands-on tools. <b>AI Explain</b> writes a plain-English root-cause summary of any alert; <b>Perf</b> benchmarks how fast each detector runs; <b>Lab</b> lets you move the detector thresholds and watch results change live; <b>Replay</b> animates the cost trend with anomalies appearing over time; <b>Raw data</b> is the underlying table behind everything.",
          side: "right",
          align: "start",
        },
      },

      // ---- Left rail: data-source controls -------------------------------
      {
        element: '[data-tour="upload"]',
        popover: {
          title: "Run it on YOUR bill",
          description:
            "This isn't limited to demo data. Drop a real <b>AWS Cost & Usage Report</b> (the .csv AWS exports to your S3 bucket) here and the <b>entire app</b> - detection, alerts, forecast, carbon - recomputes on your numbers. No file handy? The bundled <b>examples/cur_*.csv</b> files work great. Once a file is loaded, <b>Back to synthetic</b> returns you to the demo data.",
          side: "right",
          align: "start",
        },
      },
      {
        element: '[data-tour="scenario"]',
        popover: {
          title: "Or try a built-in scenario",
          description:
            "Seven synthetic mixes, each stress-testing a different pattern: <b>spike storm</b> (many sharp spikes), <b>stealth leak</b> (a slow hidden drift), <b>multi-region</b>, <b>weekend camouflage</b>, <b>calm</b> (almost nothing - tests the quiet case) and the canonical <b>default</b>. Switching the scenario instantly recomputes every view.",
          side: "right",
          align: "start",
        },
      },
      {
        element: '[data-tour="days"]',
        popover: {
          title: "How much history to simulate",
          description:
            "Drag to generate between 30 and 180 days of data. The detectors model a <b>weekly cycle</b>, so they need roughly two weeks before they can reliably tell an anomaly from normal weekend/weekday variation - longer windows give cleaner detection.",
          side: "right",
          align: "start",
        },
      },
      {
        element: '[data-tour="seed"]',
        popover: {
          title: "Random seed = reproducibility",
          description:
            "The synthetic data is random, but <b>seeded</b>: the same seed always produces the exact same bill. That's how our benchmark numbers stay reproducible - change the seed to roll a fresh-but-repeatable dataset.",
          side: "right",
          align: "start",
        },
      },

      // ---- Main content --------------------------------------------------
      {
        element: '[data-tour="kpis"]',
        popover: {
          title: "The headline numbers",
          description:
            "Always on top, and they animate as the data changes: <b>Total spend</b> in the window · <b>Services</b> tracked · <b>Anomalies flagged</b> (suspicious points found) · <b>Consensus alerts</b> (flagged by 2+ detectors, so high-confidence) · <b>$ savable</b> (money you'd recover by acting fast).",
          side: "bottom",
        },
      },
      {
        element: '[data-tour="skyline"]',
        popover: {
          title: "The 3D spend landscape",
          description:
            "A real WebGL chart: each <b>tower is one service</b>, its <b>height is how much it costs</b>, and colour goes blue→amber with spend. <b>Drag to rotate</b>, and <b>hover a tower</b> to see its exact spend and alert count. This is the kind of 3D you'll see throughout the app.",
          side: "top",
        },
      },
      {
        element: '[data-tour="highlights"]',
        popover: {
          title: "The three takeaways",
          description:
            "<b>Distinct anomalies</b> = unique (day, service) problems found. <b>Carbon footprint</b> = the same spend translated to kgCO₂e and km driven. <b>Top savings</b> = the single biggest money-saving action we detected.",
          side: "top",
        },
      },
      {
        element: '[data-tour="incidents"]',
        popover: {
          title: "Highest-severity incidents",
          description:
            "The worst anomalies, ranked. <b>Severity</b> (LOW / MEDIUM / HIGH) combines how far off it is, how long it lasted and the dollar impact. The <b># det.</b> column shows <b>how many of the 4 detectors agreed</b> - more agreement means more confidence.",
          side: "top",
          align: "start",
        },
      },
      {
        element: '[data-tour="severity"]',
        popover: {
          title: "Severity mix",
          description:
            "How the anomalies split across HIGH / MEDIUM / LOW. A FinOps engineer who only chases MEDIUM and HIGH sees almost no false alarms - those bands are about 100% accurate in our benchmark.",
          side: "left",
        },
      },
      {
        element: '[data-tour="recommendation"]',
        popover: {
          title: "What to do next",
          description:
            "Beyond detecting problems, costsight suggests a concrete fix (e.g. re-route cross-region transfers) with the expected monthly saving and a confidence level - so the insight turns into action.",
          side: "top",
        },
      },
      {
        popover: {
          title: "How detection actually works",
          description:
            "Four detectors run in parallel: <b>Z-Score</b> (simple, perfect on sudden spikes), <b>STL</b> (handles weekly seasonality + drift, strongest overall), <b>Isolation Forest</b> (a machine-learning outlier model) and an <b>Ensemble</b> that takes a majority vote. The whole point of the project: <b>no single method catches everything</b>, so we compare all four.",
        },
      },
      {
        element: '[data-tour="threed"]',
        popover: {
          title: "The 3D explorer",
          description:
            "Open this for the showcase: an auto-rotating <b>spend skyline</b>, a cost <b>surface</b> (spikes become peaks, drift becomes ridges) and a 3D <b>anomaly cloud</b>, all draggable. This view is pure 3D, but <b>most other chart views</b> default to 3D and carry a <b>3D｜2D toggle</b> in the corner, so you can drop to 2D any time you want precise reading.",
          side: "right",
        },
      },
      {
        element: '[data-tour="tourbtn"]',
        popover: {
          title: "Replay this any time",
          description:
            "That's the whole app. Click <b>Tour</b> here whenever you want to walk through it again. Enjoy exploring costsight! 🎉",
          side: "left",
        },
      },
    ],
  });
  d.drive();
}

/** Auto-run the tour once, the first time a user lands (per tour version). */
export function maybeAutoTour(): void {
  if (localStorage.getItem(SEEN_KEY)) return;
  localStorage.setItem(SEEN_KEY, "1");
  // Small delay so the shell + Summary anchors are mounted.
  setTimeout(startTour, 700);
}
