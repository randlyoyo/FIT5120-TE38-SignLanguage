import { fileURLToPath } from 'node:url'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      // @mediapipe/tasks-vision@0.10.33 ships a package.json "exports" map
      // that mixes top-level condition keys with subpath keys, which both
      // Node's and Rolldown's resolvers reject outright ("exports" cannot
      // contain some keys starting with '.' and some not) -- the build
      // fails on this import with no workaround via tsconfig/vite options
      // alone. Aliasing straight to the real entry file sidesteps package
      // "exports" resolution entirely. Do not bump this package version to
      // "fix" it without re-reading recognition/API.md §2 -- the version
      // must match the exact bundle the training keypoints were extracted
      // with.
      '@mediapipe/tasks-vision': fileURLToPath(
        new URL('./node_modules/@mediapipe/tasks-vision/vision_bundle.mjs', import.meta.url)
      ),
    },
  },
  server: {
    proxy: {
      '/api': { target: 'http://localhost:4000', changeOrigin: true },
    },
  },
})
