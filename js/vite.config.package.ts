import { mergeConfig } from "vite";

import PackageConfig from "./vite.config.package.dev.ts";

export default mergeConfig(PackageConfig, {
  build: {
    minify: true,
  },
  mode: "production",
  define: {
    "process.env.NODE_ENV": '"production"',
  },
});
