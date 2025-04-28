import { defineConfig } from 'vite';
import { qwikVite } from '@builder.io/qwik/optimizer';
import { qwikCity } from '@builder.io/qwik-city/vite';
import { resolve } from 'path';

export default defineConfig(() => {
  return {
    plugins: [
      qwikCity(),
      qwikVite(),
    ],
    preview: {
      headers: {
        'Cache-Control': 'public, max-age=600',
      },
    },
    server: {
      port: 5173,
      host: 'localhost',
      headers: {
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'require-corp',
      },
    },
    resolve: {
      alias: {
        '~': resolve(__dirname, 'src'),
      },
    },
    optimizeDeps: {
      exclude: ['onnxruntime-web'],
    },
    build: {
      commonjsOptions: {
        include: [/onnxruntime-web/, /node_modules/],
      },
    },
    assetsInclude: ['**/*.wasm'],
  };
});
