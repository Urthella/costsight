import { driver } from "driver.js";
import "driver.js/dist/driver.css";

// Bump when the tour content changes so returning users see it once more.
const SEEN_KEY = "costsight_tour_v2";

/** A detailed, plain-language guided tour — assumes zero prior knowledge, so
 *  anyone (reviewer, teammate, first-time user) understands every part of the
 *  app. Runs on the Summary view, where all the anchored elements exist. */
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
            "costsight watches your <b>cloud bill</b> and flags <b>cost anomalies</b> — days where a service costs far more than it should. It catches three kinds: a <b>point spike</b> (one bad day), a <b>level shift</b> (a permanent step up), and <b>gradual drift</b> (a slow creep). This 90-second tour explains every part of the screen.",
        },
      },
      {
        element: '[data-tour="nav"]',
        popover: {
          title: "1 · The 19 views, in 5 groups",
          description:
            "<b>Overview</b> (summary + charts), <b>Detection</b> (alerts, root-cause, detector comparison, drift), <b>FinOps</b> (forecast, budget, savings), <b>Sustainability</b> (carbon, tagging) and <b>Lab & Data</b> (AI explain, performance, raw data). Click any item to open that view.",
          side: "right",
          align: "start",
        },
      },
      {
        element: '[data-tour="kpis"]',
        popover: {
          title: "2 · The headline numbers",
          description:
            "Always visible, and they animate as the data changes: <b>Total spend</b> in the window · <b>Services</b> tracked · <b>Anomalies flagged</b> (suspicious points found) · <b>Consensus alerts</b> (flagged by 2+ detectors, so high-confidence) · <b>$ savable</b> (money you'd recover by acting fast).",
          side: "bottom",
        },
      },
      {
        element: '[data-tour="skyline"]',
        popover: {
          title: "3 · The 3D spend landscape",
          description:
            "A real WebGL chart: each <b>tower is one service</b>, its <b>height is how much it costs</b>, colour goes blue→amber with spend. <b>Drag to rotate</b>, and <b>hover a tower</b> to see its exact spend and how many alerts it has. This is the kind of 3D you'll see throughout the app.",
          side: "top",
        },
      },
      {
        element: '[data-tour="highlights"]',
        popover: {
          title: "4 · The three takeaways",
          description:
            "<b>Distinct anomalies</b> = unique (day, service) problems found. <b>Carbon footprint</b> = the same spend translated to kgCO₂e (and km driven), the sustainability angle. <b>Top savings</b> = the single biggest money-saving action we detected.",
          side: "top",
        },
      },
      {
        element: '[data-tour="incidents"]',
        popover: {
          title: "5 · Highest-severity incidents",
          description:
            "The worst anomalies, ranked. <b>Severity</b> (LOW/MEDIUM/HIGH) combines how far off it is, how long it lasted, and the dollar impact. The <b># det.</b> column shows <b>how many of the 4 detectors agreed</b> — more agreement means more confidence.",
          side: "top",
          align: "start",
        },
      },
      {
        element: '[data-tour="severity"]',
        popover: {
          title: "6 · Severity mix",
          description:
            "How the anomalies split across HIGH / MEDIUM / LOW. A FinOps engineer who only chases MEDIUM and HIGH sees almost no false alarms — those bands are ~100% accurate in our benchmark.",
          side: "left",
        },
      },
      {
        element: '[data-tour="recommendation"]',
        popover: {
          title: "7 · What to do next",
          description:
            "Beyond detecting problems, costsight suggests a concrete fix (e.g. re-route cross-region transfers) with the expected monthly saving and a confidence level — so the insight turns into action.",
          side: "top",
        },
      },
      {
        popover: {
          title: "8 · How detection actually works",
          description:
            "Four detectors run in parallel: <b>Z-Score</b> (simple, perfect on sudden spikes), <b>STL</b> (handles weekly seasonality + drift, strongest overall), <b>Isolation Forest</b> (machine-learning outlier model), and an <b>Ensemble</b> that takes a majority vote. The point of the project: <b>no single method catches everything</b>, so we compare all four.",
        },
      },
      {
        element: '[data-tour="threed"]',
        popover: {
          title: "9 · The 3D explorer + 2D toggle",
          description:
            "Open this for the showcase: a rotatable cost <b>surface</b> (spikes become peaks, drift becomes ridges) and a 3D <b>anomaly cloud</b>. Most chart views default to 3D, and every one has a <b>3D｜2D toggle</b> in the corner — switch to 2D any time you want precise reading.",
          side: "right",
        },
      },
      {
        element: '[data-tour="datasource"]',
        popover: {
          title: "10 · Run it on YOUR bill",
          description:
            "This isn't limited to demo data. Drop a real <b>AWS Cost & Usage Report</b> (the .csv AWS gives you) here and the entire app — detection, alerts, forecast, carbon — recomputes on your numbers.",
          side: "right",
        },
      },
      {
        element: '[data-tour="scenario"]',
        popover: {
          title: "11 · Or try built-in scenarios",
          description:
            "No bill handy? Pick a synthetic <b>scenario</b> (spike storm, stealth leak, calm, …), change the <b>days of history</b> or the <b>random seed</b>, and watch the detectors respond. Great for stress-testing.",
          side: "right",
        },
      },
      {
        element: '[data-tour="tourbtn"]',
        popover: {
          title: "12 · Replay this any time",
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
