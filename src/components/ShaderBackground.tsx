import React, { Suspense } from 'react';
import { ShaderGradientCanvas, ShaderGradient } from '@shadergradient/react';

export const ShaderBackground: React.FC = () => {
  const gradientProps: any = {
    animate: 'on',
    axesHelper: 'off',
    brightness: 0.6,
    cAzimuthAngle: 180,
    cDistance: 3.6,
    cPolarAngle: 90,
    cameraZoom: 1,
    color1: '#a173ff',
    color2: '#dbd6d4',
    color3: '#e1a0d6',
    destination: 'localFile',
    embedMode: 'off',
    envPreset: 'city',
    format: 'webm',
    frameRate: 10,
    gizmoHelper: 'hide',
    grain: 'off',
    lightType: '3d',
    loop: 'on',
    loopDuration: 19.8,
    pixelDensity: 1,
    positionX: -1.4,
    positionY: 0,
    positionZ: 0,
    range: 'enabled',
    rangeEnd: 19.8,
    rangeStart: 0,
    reflection: 0.1,
    rotationX: 0,
    rotationY: 10,
    rotationZ: 50,
    shader: 'defaults',
    toggleAxis: false,
    type: 'plane',
    uAmplitude: 1,
    uDensity: 1.3,
    uFrequency: 5.5,
    uSpeed: 0.2,
    uStrength: 4,
    uTime: 1.68,
    wireframe: false,
    zoomOut: false,
  };

  return (
    <div
      className="fixed inset-0 z-0 pointer-events-none overflow-hidden"
      style={{ position: 'fixed', inset: 0, zIndex: 0, pointerEvents: 'none' }}
    >
      <Suspense fallback={<div className="fixed inset-0 bg-[#f5f0e6]" />}>
        <ShaderGradientCanvas
          pixelDensity={1}
          pointerEvents="none"
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            pointerEvents: 'none',
          }}
        >
          <ShaderGradient {...gradientProps} />
        </ShaderGradientCanvas>
      </Suspense>
    </div>
  );
};
