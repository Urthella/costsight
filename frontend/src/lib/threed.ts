// Helpers for true 3D charts in Plotly: real extruded bars (mesh3d cuboids)
// and a categorical 3D scene, so the dashboard can lead with 3D.

export interface Bar3D {
  x: number;
  y: number;
  h: number;
  color: string;
}

// 12 triangles (2 per cube face) over the 8 corners of each bar.
const FACES = [
  [0, 1, 2], [0, 2, 3], // bottom
  [4, 6, 5], [4, 7, 6], // top
  [0, 5, 1], [0, 4, 5], // front
  [1, 5, 6], [1, 6, 2], // right
  [2, 6, 7], [2, 7, 3], // back
  [3, 7, 4], [3, 4, 0], // left
];

/** Combine many cuboids into one mesh3d trace - a 3D bar chart. */
export function bars3dTrace(bars: Bar3D[], w = 0.32, d = 0.32): Record<string, unknown> {
  const X: number[] = [], Y: number[] = [], Z: number[] = [];
  const I: number[] = [], J: number[] = [], K: number[] = [], FC: string[] = [];
  bars.forEach((b, n) => {
    const x0 = b.x - w, x1 = b.x + w, y0 = b.y - d, y1 = b.y + d;
    const z0 = 0, z1 = Math.max(b.h, 1e-6);
    const corners = [
      [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
      [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ];
    corners.forEach((c) => { X.push(c[0]); Y.push(c[1]); Z.push(c[2]); });
    const base = n * 8;
    FACES.forEach((f) => { I.push(base + f[0]); J.push(base + f[1]); K.push(base + f[2]); FC.push(b.color); });
  });
  return {
    type: "mesh3d",
    x: X, y: Y, z: Z, i: I, j: J, k: K,
    facecolor: FC,
    flatshading: true,
    hoverinfo: "skip",
    lighting: { ambient: 0.65, diffuse: 0.8, roughness: 0.5, specular: 0.15 },
  };
}

/** Marker+text trace sitting on top of bars so hover shows the real value. */
export function barLabels(
  bars: Bar3D[],
  labels: string[],
): Record<string, unknown> {
  return {
    type: "scatter3d",
    mode: "markers",
    x: bars.map((b) => b.x),
    y: bars.map((b) => b.y),
    z: bars.map((b) => b.h),
    marker: { size: 3, color: bars.map((b) => b.color) },
    text: labels,
    hoverinfo: "text",
    showlegend: false,
  };
}

/** A categorical 3D scene with tick labels and a pleasant default camera. */
export function scene3d(
  xticks: string[],
  yticks: string[],
  zlabel: string,
): Record<string, unknown> {
  return {
    xaxis: { tickvals: xticks.map((_, i) => i), ticktext: xticks, title: "" },
    yaxis: { tickvals: yticks.map((_, i) => i), ticktext: yticks, title: "" },
    zaxis: { title: zlabel },
    camera: { eye: { x: 1.7, y: 1.7, z: 1.05 } },
    aspectmode: "cube",
  };
}

export const PLOT3D_LAYOUT: Record<string, unknown> = {
  height: 420,
  margin: { t: 10, r: 10, b: 10, l: 10 },
  paper_bgcolor: "rgba(0,0,0,0)",
  showlegend: false,
};
