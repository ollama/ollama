import webpack from 'webpack'
import ForkTsCheckerWebpackPlugin from 'fork-ts-checker-webpack-plugin'
// When running under native ESM, named imports from CommonJS packages (like 'webpack')
// are not available directly. We import the namespace/default and destructure.
const { DefinePlugin } = webpack

export const plugins = [
  new ForkTsCheckerWebpackPlugin({
    logger: 'webpack-infrastructure',
  }),
  new DefinePlugin({
    'process.env.TELEMETRY_WRITE_KEY': JSON.stringify(process.env.TELEMETRY_WRITE_KEY),
  }),
]
