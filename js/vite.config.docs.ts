import { mergeConfig } from 'vite'

import DocsConfig from './vite.config.dev.docs'

// https://vite.dev/config/
export default mergeConfig(DocsConfig, {
  build: {
    minify: true,
    sourcemap: true,
  }
})