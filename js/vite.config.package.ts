import { mergeConfig } from "vite";

import PackageConfig from "./vite.config.package.dev.ts";

export default mergeConfig(PackageConfig, {
  build: {
    emptyOutDir: true,
    minify: true,
    sourcemap: false,
  },
  mode: "production",
  define: {
    "process.env.NODE_ENV": '"production"',
  },
});
