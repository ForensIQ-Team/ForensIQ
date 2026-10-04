import React, { Suspense, useRef, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Line, Html } from '@react-three/drei';
import * as THREE from 'three';
import { Network, Share2, Globe, Radio } from 'lucide-react';

interface NodeData {
  id: number;
  label: string;
  category: string;
  targetPos: [number, number, number];
  startPos: [number, number, number];
  color: string;
  size: number;
}

const NODES_DATA: NodeData[] = [
  { id: 0, label: 'ORIGIN (Source Image)', category: 'Source', targetPos: [0, 0, 0], startPos: [0, 0, 0], color: '#dc2626', size: 0.45 },
  { id: 1, label: 'NewsPortal.com', category: 'News', targetPos: [-2.2, 1.4, 0.5], startPos: [-4, 3, 2], color: '#e11d48', size: 0.25 },
  { id: 2, label: 'SocialMediaMirror', category: 'Social', targetPos: [2.0, 1.8, -0.8], startPos: [4, 4, -2], color: '#e11d48', size: 0.25 },
  { id: 3, label: 'ForumArchive.net', category: 'Forum', targetPos: [-1.8, -1.6, -0.5], startPos: [-3, -4, -1], color: '#b91c1c', size: 0.22 },
  { id: 4, label: 'BlogHost.org', category: 'Blog', targetPos: [2.5, -1.2, 0.4], startPos: [5, -3, 1], color: '#b91c1c', size: 0.22 },
  { id: 5, label: 'Account_Sockpuppet1', category: 'Account', targetPos: [-3.2, 0.2, -1.2], startPos: [-6, 1, -3], color: '#991b1b', size: 0.18 },
  { id: 6, label: 'StockImageRepo', category: 'Archive', targetPos: [3.4, 0.5, 1.1], startPos: [6, 2, 3], color: '#991b1b', size: 0.2 },
  { id: 7, label: 'MediaCloudMirror', category: 'Cloud', targetPos: [0.8, 2.6, 0.8], startPos: [2, 5, 2], color: '#e11d48', size: 0.22 },
  { id: 8, label: 'FactCheckPortal', category: 'News', targetPos: [-1.2, 2.5, -1.5], startPos: [-2, 5, -3], color: '#10b981', size: 0.25 },
  { id: 9, label: 'DeepFakeForumThread', category: 'Forum', targetPos: [1.2, -2.4, -1.2], startPos: [3, -5, -3], color: '#dc2626', size: 0.25 },
  { id: 10, label: 'TelegramChannel', category: 'Social', targetPos: [-2.8, -0.8, 1.2], startPos: [-5, -2, 3], color: '#e11d48', size: 0.2 },
  { id: 11, label: 'PublicRepository', category: 'Archive', targetPos: [0.2, -3.0, 0.6], startPos: [1, -6, 2], color: '#991b1b', size: 0.2 },
  { id: 12, label: 'UnverifiedHost', category: 'Web', targetPos: [3.6, -2.2, -0.6], startPos: [6, -4, -2], color: '#b91c1c', size: 0.18 },
  { id: 13, label: 'NewsWireFeed', category: 'News', targetPos: [-3.5, 2.2, -0.2], startPos: [-6, 4, -1], color: '#e11d48', size: 0.22 },
  { id: 14, label: 'ImageSharerBot', category: 'Account', targetPos: [1.8, 3.2, -0.4], startPos: [3, 6, -1], color: '#991b1b', size: 0.18 },
  { id: 15, label: 'MirrorNode_EU', category: 'Cloud', targetPos: [-0.5, -2.2, -2.0], startPos: [-1, -5, -4], color: '#991b1b', size: 0.18 },
];

function SceneContents({ isVisible }: { isVisible: boolean }) {
  const groupRef = useRef<THREE.Group>(null);
  const centerSphereRef = useRef<THREE.Mesh>(null);
  const progress = useRef(0);

  useFrame(({ pointer, clock }) => {
    const time = clock.getElapsedTime();

    // Lerp progress when visible
    progress.current = THREE.MathUtils.lerp(progress.current, isVisible ? 1 : 0, 0.04);

    // Subtle pulse for central ORIGIN node
    if (centerSphereRef.current) {
      const scale = 1 + Math.sin(time * 3) * 0.12;
      centerSphereRef.current.scale.set(scale, scale, scale);
    }

    // Parallax rotation based on mouse pointer
    if (groupRef.current) {
      groupRef.current.rotation.y = THREE.MathUtils.lerp(groupRef.current.rotation.y, pointer.x * 0.35, 0.05);
      groupRef.current.rotation.x = THREE.MathUtils.lerp(groupRef.current.rotation.x, -pointer.y * 0.25, 0.05);
    }
  });

  return (
    <group ref={groupRef}>
      {/* 3D Nodes */}
      {NODES_DATA.map((node) => {
        const isOrigin = node.id === 0;
        // Interpolate position from startPos to targetPos
        const currentPos: [number, number, number] = [
          THREE.MathUtils.lerp(node.startPos[0], node.targetPos[0], progress.current),
          THREE.MathUtils.lerp(node.startPos[1], node.targetPos[1], progress.current),
          THREE.MathUtils.lerp(node.startPos[2], node.targetPos[2], progress.current),
        ];

        return (
          <group key={node.id} position={currentPos}>
            <mesh ref={isOrigin ? centerSphereRef : undefined}>
              <sphereGeometry args={[node.size, 24, 24]} />
              <meshBasicMaterial color={node.color} />
            </mesh>

            {/* Glowing Ring around Origin */}
            {isOrigin && (
              <mesh rotation={[Math.PI / 2, 0, 0]}>
                <ringGeometry args={[0.6, 0.68, 32]} />
                <meshBasicMaterial color="#ef4444" side={THREE.DoubleSide} transparent opacity={0.8} />
              </mesh>
            )}

            {/* Node HTML Label */}
            <Html distanceFactor={12} position={[0, node.size + 0.25, 0]} center>
              <div className="bg-stone-900/90 text-stone-100 text-xs font-mono px-2 py-0.5 rounded border border-stone-700 whitespace-nowrap backdrop-blur-xs shadow-md pointer-events-none">
                <span className={isOrigin ? 'text-red-400 font-bold' : 'text-stone-300'}>
                  {node.label}
                </span>
              </div>
            </Html>
          </group>
        );
      })}

      {/* Connecting Lines from Origin (0,0,0) to all target nodes */}
      {NODES_DATA.slice(1).map((node) => {
        const currentTarget: [number, number, number] = [
          THREE.MathUtils.lerp(node.startPos[0], node.targetPos[0], progress.current),
          THREE.MathUtils.lerp(node.startPos[1], node.targetPos[1], progress.current),
          THREE.MathUtils.lerp(node.startPos[2], node.targetPos[2], progress.current),
        ];

        return (
          <Line
            key={`line-${node.id}`}
            points={[[0, 0, 0], currentTarget]}
            color={node.color}
            lineWidth={1.5}
            transparent
            opacity={0.45 * progress.current}
          />
        );
      })}
    </group>
  );
}

export const PropagationMap: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
        }
      },
      { threshold: 0.2 }
    );

    if (containerRef.current) {
      observer.observe(containerRef.current);
    }

    return () => observer.disconnect();
  }, []);

  return (
    <section id="propagation" ref={containerRef} className="py-20 px-4 sm:px-6 lg:px-12 max-w-7xl mx-auto">
      <div className="text-center max-w-2xl mx-auto mb-10 space-y-3">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-red-100 border border-red-200 text-red-900 font-mono text-xs font-bold uppercase">
          <Share2 className="w-3.5 h-3.5 text-red-700" />
          <span>REACT THREE FIBER (3D ENGINE)</span>
        </div>
        <h2 className="text-3xl sm:text-4xl font-serif font-bold text-stone-900 tracking-tight">
          3D Dissemination & Propagation Tracking
        </h2>
        <p className="text-stone-950 text-sm leading-relaxed">
          Interactive 3D network visualizing how compromised media spreads from source origin across news hosts, social mirrors, and unverified forums.
        </p>
      </div>

      {/* R3F 3D Canvas Stage */}
      <div className="relative w-full h-[480px] sm:h-[560px] bg-[#0c1017] rounded-2xl border-2 border-stone-800 shadow-2xl overflow-hidden">
        {/* Top Overlay Stats Header */}
        <div className="absolute top-4 left-4 right-4 z-20 flex flex-wrap items-center justify-between gap-3 text-stone-300 font-mono text-xs bg-stone-950/80 p-3 rounded-xl border border-stone-800 backdrop-blur-md">
          <div className="flex items-center gap-2">
            <Radio className="w-4 h-4 text-red-500 animate-pulse" />
            <span className="font-bold text-stone-100">LIVE GRAPH: 16 INDEXED NODES</span>
          </div>
          <div className="flex items-center gap-4 text-xs">
            <span>ORIGIN SPREAD VELOCITY: <strong className="text-red-400">4.2x / hr</strong></span>
            <span>PROVENANCE RESIDUAL: <strong className="text-emerald-400">100% RECOVERED</strong></span>
          </div>
        </div>

        {/* 3D Canvas */}
        <Suspense
          fallback={
            <div className="w-full h-full flex items-center justify-center text-red-400 font-mono text-xs">
              Loading 3D Propagation Canvas...
            </div>
          }
        >
          <Canvas camera={{ position: [0, 0, 8.5], fov: 50 }}>
            <ambientLight intensity={0.8} />
            <pointLight position={[10, 10, 10]} intensity={1.2} />
            <SceneContents isVisible={isVisible} />
          </Canvas>
        </Suspense>

        {/* Bottom Legend Overlay */}
        <div className="absolute bottom-4 left-4 right-4 z-20 flex flex-wrap items-center justify-between text-xs font-mono text-stone-950 bg-stone-950/70 p-2.5 rounded-lg border border-stone-800/80 backdrop-blur-xs">
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-red-600 inline-block" /> Central Origin</span>
            <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-rose-500 inline-block" /> Social Mirror</span>
            <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-emerald-500 inline-block" /> Fact Check</span>
          </div>
          <span>MOUSE PARALLAX ACTIVE</span>
        </div>
      </div>
    </section>
  );
};
