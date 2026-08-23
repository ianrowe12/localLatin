/// <reference types="vitest" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  test: {
    // Component tests (VariantSelector, feedback wiring) need a DOM; the pure
    // helper tests run happily under jsdom too.
    environment: 'jsdom',
    globals: false,
    setupFiles: ['./src/test/setup.ts'],
  },
  base: process.env.VITE_BASE_PATH || '/',
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes('node_modules')) return undefined
          if (id.includes('/react-dom/') || id.includes('/react/')) return 'react'
          if (id.includes('/framer-motion/')) return 'motion'
          if (id.includes('/@floating-ui/')) return 'floating-ui'
          if (id.includes('/react-virtuoso/')) return 'virtuoso'
          return 'vendor'
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8001',
        changeOrigin: true,
      },
    },
  },
})
