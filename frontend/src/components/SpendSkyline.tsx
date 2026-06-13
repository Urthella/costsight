import { useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Html } from "@react-three/drei";
import { useReducedMotion } from "framer-motion";

export interface SkylineBar {
  service: string;
  total: number;
}

function Bars({
  data,
  info,
  hover,
  selected,
  setHover,
  setSelected,
}: {
  data: SkylineBar[];
  info?: Record<string, string>;
  hover: number | null;
  selected: number | null;
  setHover: (i: number | null) => void;
  setSelected: (i: number | null) => void;
}) {
  const max = Math.max(...data.map((d) => d.total), 1);
  const n = data.length;
  return (
    <group position={[0, -1.2, 0]}>
      {data.map((d, i) => {
        const h = (d.total / max) * 4 + 0.25;
        const x = (i - (n - 1) / 2) * 1.25;
        const t = d.total / max;
        const color = `hsl(${220 - t * 180}, 72%, 52%)`;
        const active = hover === i || selected === i;
        return (
          <group key={d.service}>
            <mesh
              position={[x, h / 2, 0]}
              scale={active ? [1.08, 1.02, 1.08] : [1, 1, 1]}
              castShadow
              onPointerOver={(e) => {
                e.stopPropagation();
                setHover(i);
                document.body.style.cursor = "pointer";
              }}
              onPointerOut={() => {
                setHover(null);
                document.body.style.cursor = "auto";
              }}
              onClick={(e) => {
                e.stopPropagation();
                setSelected(selected === i ? null : i);
              }}
            >
              <boxGeometry args={[0.72, h, 0.72]} />
              <meshStandardMaterial
                color={color}
                emissive={color}
                emissiveIntensity={active ? 0.4 : 0}
                roughness={0.45}
                metalness={0.1}
              />
            </mesh>
            {active && (
              <Html position={[x, h + 0.6, 0]} center distanceFactor={9}>
                <div
                  style={{
                    whiteSpace: "nowrap",
                    background: "#0f172a",
                    color: "white",
                    padding: "4px 8px",
                    borderRadius: 6,
                    fontSize: 12,
                    fontWeight: 600,
                  }}
                >
                  {d.service} · $
                  {d.total.toLocaleString("en-US", { maximumFractionDigits: 0 })}
                  {info?.[d.service] ? ` · ${info[d.service]}` : ""}
                </div>
              </Html>
            )}
          </group>
        );
      })}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]} receiveShadow>
        <planeGeometry args={[n * 1.6, 5]} />
        <meshStandardMaterial color="#e2e8f0" roughness={1} />
      </mesh>
    </group>
  );
}

/** WebGL hero: per-service total spend as an interactive 3D skyline.
 *  Hover/click a bar to highlight it (and pause auto-rotate); drag to orbit. */
export function SpendSkyline({
  data,
  info,
}: {
  data: SkylineBar[];
  info?: Record<string, string>;
}) {
  const reduced = useReducedMotion();
  const [hover, setHover] = useState<number | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  return (
    <div style={{ height: 320, width: "100%" }}>
      <Canvas camera={{ position: [0, 3.2, 9], fov: 50 }} dpr={[1, 2]} shadows>
        <ambientLight intensity={0.65} />
        <directionalLight position={[5, 9, 5]} intensity={1.15} castShadow />
        <Bars
          data={data}
          info={info}
          hover={hover}
          selected={selected}
          setHover={setHover}
          setSelected={setSelected}
        />
        <OrbitControls
          enablePan={false}
          enableZoom={false}
          autoRotate={!reduced && hover === null && selected === null}
          autoRotateSpeed={1.1}
          minPolarAngle={0.7}
          maxPolarAngle={1.45}
        />
      </Canvas>
    </div>
  );
}
