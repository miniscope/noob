import { mergeConfig } from 'vite'
import { resolve } from 'node:path'

import baseConfig from './vite.config.base.ts'

// https://vite.dev/config/
export default mergeConfig(baseConfig, {
  build: {
    lib: {
      entry: resolve(__dirname, 'src/main_package.ts'),
    },
    outDir: resolve(__dirname, '../packages/noob/src/noob/_js'),
  },
})