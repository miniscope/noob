import { mergeConfig } from 'vite'

import PackageConfig from './vite.config.package.dev.ts'

// https://vite.dev/config/
export default mergeConfig(PackageConfig, {
  build: {
    minify: true,
  }
})