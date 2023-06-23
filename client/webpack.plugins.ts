import type IForkTsCheckerWebpackPlugin from 'fork-ts-checker-webpack-plugin'
import * as path from 'path'
import PermissionsPlugin from './permissions-plugin'

// eslint-disable-next-line @typescript-eslint/no-var-requires
const ForkTsCheckerWebpackPlugin: typeof IForkTsCheckerWebpackPlugin = require('fork-ts-checker-webpack-plugin')
const CopyWebpackPlugin = require('copy-webpack-plugin')

export const plugins = [
  new ForkTsCheckerWebpackPlugin({
    logger: 'webpack-infrastructure',
  }),
  new CopyWebpackPlugin({
    patterns: [{ from: 'resources', to: 'resources' }],
  }),
  new PermissionsPlugin({
    resourcePath: '.webpack/renderer/resources/server',
  }),
]
