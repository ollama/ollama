import type { ForgeConfig } from '@electron-forge/shared-types'
import { MakerSquirrel } from '@electron-forge/maker-squirrel'
import { MakerZIP } from '@electron-forge/maker-zip'
import { MakerDeb } from '@electron-forge/maker-deb'
import { MakerRpm } from '@electron-forge/maker-rpm'
import { PublisherGithub } from '@electron-forge/publisher-github'
import { AutoUnpackNativesPlugin } from '@electron-forge/plugin-auto-unpack-natives'
import { WebpackPlugin } from '@electron-forge/plugin-webpack'

import { mainConfig } from './webpack.main.config'
import { rendererConfig } from './webpack.renderer.config'

const config: ForgeConfig = {
  packagerConfig: {
    asar: true,
    icon: './images/icon',
    extraResource: ['../dist/ollama'],
    ...(process.env.SIGN
      ? {
          osxSign: {
            identity: process.env.APPLE_IDENTITY,
          },
          osxNotarize: {
            tool: 'notarytool',
            appleId: process.env.APPLE_ID || '',
            appleIdPassword: process.env.APPLE_PASSWORD || '',
            teamId: process.env.APPLE_TEAM_ID || '',
          },
        }
      : {}),
  },
  rebuildConfig: {},
  makers: [new MakerSquirrel({}), new MakerZIP({}, ['darwin']), new MakerRpm({}), new MakerDeb({})],
  publishers: [
    new PublisherGithub({
      repository: {
        name: 'ollama',
        owner: 'jmorganca',
      },
      draft: false,
      prerelease: true,
    }),
  ],
  plugins: [
    new AutoUnpackNativesPlugin({}),
    new WebpackPlugin({
      mainConfig,
      devContentSecurityPolicy: `default-src * 'unsafe-eval' 'unsafe-inline'`,
      renderer: {
        config: rendererConfig,
        nodeIntegration: true,
        entryPoints: [
          {
            html: './src/index.html',
            js: './src/renderer.tsx',
            name: 'main_window',
            preload: {
              js: './src/preload.ts',
            },
          },
        ],
      },
    }),
  ],
}

export default config
