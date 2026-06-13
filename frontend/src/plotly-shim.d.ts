// plotly.js-dist-min and the react-plotly.js factory ship no bundled types;
// we drive them through a thin wrapper (lib/plot.tsx) so `any` here is fine.
declare module "plotly.js-dist-min";
declare module "react-plotly.js/factory";
