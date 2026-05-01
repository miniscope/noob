import { mergeConfig } from 'vite'

import DocsConfig from './vite.config.docs.dev.ts'

export default mergeConfig(DocsConfig, {
  build: {
    minify: true,
  }
})