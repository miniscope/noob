import { defineConfig } from 'vite'
import { resolve } from 'node:path'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  build: {
    target: "baseline-widely-available",
    lib: {
      entry: resolve(__dirname, 'src/main_docs.ts'),
      name: "noob_js",
      fileName: 'js/noob-js',
      cssFileName: "css/noob-js",
    },
    outDir: resolve(__dirname, '../docs/_static'),
    emptyOutDir: false,
    minify: false
  },
  plugins: [
    react({
      babel: {
        plugins: [['babel-plugin-react-compiler']],
      },
    }),
  ],
  define: {
    'process.env': {}
  }
})