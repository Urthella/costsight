import { driver } from "driver.js";
import "driver.js/dist/driver.css";

const SEEN_KEY = "costsight_tour_v1";

/** Guided product tour — great for demos ("here's what's where"). */
export function startTour(): void {
  const d = driver({
    showProgress: true,
    overlayOpacity: 0.55,
    nextBtnText: "Next →",
    prevBtnText: "← Back",
    doneBtnText: "Done",
    steps: [
      {
        popover: {
          title: "Welcome to costsight ☁️",
          description:
            "Automated cloud-cost anomaly detection — four detectors, severity scoring, root-cause, forecasting, carbon and more. Here's a 60-second tour.",
        },
      },
      {
        element: '[data-tour="nav"]',
        popover: {
          title: "19 analyses, 5 groups",
          description:
            "Overview, Detection, FinOps, Sustainability and Lab & Data. Jump to any view from here.",
          side: "right",
        },
      },
      {
        element: '[data-tour="kpis"]',
        popover: {
          title: "Live headline metrics",
          description:
            "Total spend, anomalies flagged, consensus alerts and the dollars you could save — they recompute (and count up) whenever the data changes.",
          side: "bottom",
        },
      },
      {
        element: '[data-tour="threed"]',
        popover: {
          title: "3D, front and centre",
          description:
            "Most charts render in 3D by default — drag to orbit, and flip the 3D｜2D switch any time. This 3D explorer is the showcase view.",
          side: "right",
        },
      },
      {
        element: '[data-tour="datasource"]',
        popover: {
          title: "Bring your own bill",
          description:
            "Upload a real AWS Cost & Usage Report (.csv) and every view runs on your own data — detection, alerts, forecast, carbon, the lot.",
          side: "right",
        },
      },
      {
        element: '[data-tour="scenario"]',
        popover: {
          title: "Or explore scenarios",
          description:
            "Switch synthetic anomaly scenarios, history length and random seed to stress-test the detectors.",
          side: "right",
        },
      },
      {
        popover: {
          title: "That's the tour 🎉",
          description: "Replay it any time with the Tour button at the top right.",
        },
      },
    ],
  });
  d.drive();
}

/** Auto-run the tour once, the first time a user lands. */
export function maybeAutoTour(): void {
  if (localStorage.getItem(SEEN_KEY)) return;
  localStorage.setItem(SEEN_KEY, "1");
  // Small delay so the shell + KPI anchors are mounted.
  setTimeout(startTour, 700);
}
