import type { ForgeConfig } from '@electron-forge/shared-types'
import { MakerSquirrel } from '@electron-forge/maker-squirrel'
import { MakerZIP } from '@electron-forge/maker-zip'
import { PublisherGithub } from '@electron-forge/publisher-github'
import { AutoUnpackNativesPlugin } from '@electron-forge/plugin-auto-unpack-natives'
import { WebpackPlugin } from '@electron-forge/plugin-webpack'
import * as path from 'path'
import * as fs from 'fs'

import { mainConfig } from './webpack.main.config'
import { rendererConfig } from './webpack.renderer.config'

const packageJson = JSON.parse(fs.readFileSync(path.resolve(__dirname, './package.json'), 'utf8'))

const config: ForgeConfig = {
  packagerConfig: {
    appVersion: process.env.VERSION || packageJson.version,
    asar: true,
    icon: './assets/icon.icns',
    extraResource: [
      '../ollama',
      path.join(__dirname, './assets/ollama_icon_16x16Template.png'),
      path.join(__dirname, './assets/ollama_icon_16x16Template@2x.png'),
      ...(process.platform === 'darwin' ? ['../llama/ggml-metal.metal'] : []),
    ],
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
  makers: [new MakerSquirrel({}), new MakerZIP({}, ['darwin'])],
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
  hooks: {
    readPackageJson: async (_, packageJson) => {
      return { ...packageJson, version: process.env.VERSION || packageJson.version }
    },
  },
  plugins: [
    new AutoUnpackNativesPlugin({}),
    new WebpackPlugin({
      mainConfig,
      devContentSecurityPolicy: `default-src * 'unsafe-eval' 'unsafe-inline'; img-src data: 'self'`,
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
