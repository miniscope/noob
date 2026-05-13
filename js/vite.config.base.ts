import { defineConfig } from "vite";
import react, { reactCompilerPreset } from "@vitejs/plugin-react";
import babel from "@rolldown/plugin-babel";
import { resolve } from "node:path";

export default defineConfig({
  build: {
    target: "baseline-widely-available",
    lib: {
      entry: resolve(__dirname, "src/main_docs.ts"),
      name: "noob_js",
      fileName: "js/noob-js",
      cssFileName: "css/noob-js",
      formats: ["es"],
    },
    outDir: resolve(__dirname, "/dist"),
    emptyOutDir: false,
    minify: false,
    sourcemap: true,
    rolldownOptions: {
      output: {
        codeSplitting: {
          groups: [
            {
              test: /node_modules\/react.*/,
              name: "js/react",
            },
            {
              test: /node_modules\/elkjs/,
              name: "js/elkjs",
            },
            {
              test: /node_modules\/@xyflow/,
              name: "js/reactflow",
            },
          ],
        },
      },
    },
  },
  plugins: [react(), babel({ presets: [reactCompilerPreset()] })],
  define: {
    "process.env.NODE_ENV": '"development"',
  },
});
