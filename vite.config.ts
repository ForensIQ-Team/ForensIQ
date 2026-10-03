import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import {defineConfig} from 'vite';

export default defineConfig(() => {
  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
    optimizeDeps: {
      include: ['react', 'react-dom', 'lucide-react', 'three', '@react-three/fiber', '@react-three/drei'],
    },
    server: {
      port: 3000,
      host: true,
      hmr: process.env.DISABLE_HMR !== 'true',
      watch: {
        ignored: [
          '**/ForensIQ-ML/**',
          '**/node_modules/**',
          '**/.git/**',
          '**/*.py',
          '**/*.json',
          '**/*.jpg',
          '**/*.jpeg',
          '**/*.png',
        ],
      },
    },
  };
});
