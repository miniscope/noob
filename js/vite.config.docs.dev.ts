import { mergeConfig } from "vite";
import { resolve } from "node:path";

import baseConfig from "./vite.config.base.ts";

export default mergeConfig(baseConfig, {
  build: {
    lib: {
      entry: resolve(__dirname, "src/main_docs.ts"),
    },
    outDir: resolve(__dirname, "../docs/_static"),
  },
});
