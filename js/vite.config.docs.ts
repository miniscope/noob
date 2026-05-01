import { mergeConfig } from 'vite'

import DocsConfig from './vite.config.docs.dev.ts'

// https://vite.dev/config/
export default mergeConfig(DocsConfig, {
  build: {
    minify: true,
  }
})