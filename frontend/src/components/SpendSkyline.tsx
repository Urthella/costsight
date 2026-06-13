import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useReducedMotion } from "framer-motion";

export interface SkylineBar {
  service: string;
  total: number;
}

function Bars({ data }: { data: SkylineBar[] }) {
  const max = Math.max(...data.map((d) => d.total), 1);
  const n = data.length;
  return (
    <group position={[0, -1.2, 0]}>
      {data.map((d, i) => {
        const h = (d.total / max) * 4 + 0.25;
        const x = (i - (n - 1) / 2) * 1.25;
        const t = d.total / max;
        // Blue (low) → amber (high): hue 220° → 40°.
        const color = `hsl(${220 - t * 180}, 72%, 52%)`;
        return (
          <mesh key={d.service} position={[x, h / 2, 0]} castShadow>
            <boxGeometry args={[0.72, h, 0.72]} />
            <meshStandardMaterial color={color} roughness={0.45} metalness={0.1} />
          </mesh>
        );
      })}
      {/* Ground plane for spatial context */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]} receiveShadow>
        <planeGeometry args={[n * 1.6, 5]} />
        <meshStandardMaterial color="#e2e8f0" roughness={1} />
      </mesh>
    </group>
  );
}

/** WebGL hero: per-service total spend as an auto-rotating 3D "skyline".
 *  Bar height ∝ spend, colour ramps blue→amber; drag to orbit. */
export function SpendSkyline({ data }: { data: SkylineBar[] }) {
  const reduced = useReducedMotion();
  return (
    <div style={{ height: 300, width: "100%" }}>
      <Canvas camera={{ position: [0, 3.2, 9], fov: 50 }} dpr={[1, 2]} shadows>
        <ambientLight intensity={0.65} />
        <directionalLight position={[5, 9, 5]} intensity={1.15} castShadow />
        <Bars data={data} />
        <OrbitControls
          enablePan={false}
          enableZoom={false}
          autoRotate={!reduced}
          autoRotateSpeed={1.1}
          minPolarAngle={0.7}
          maxPolarAngle={1.45}
        />
      </Canvas>
    </div>
  );
}
