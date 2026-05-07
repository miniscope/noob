import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
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
  },
  plugins: [
    react({
      babel: {
        plugins: [["babel-plugin-react-compiler"]],
      },
    }),
  ],
  define: {
    "process.env": {},
  },
});
