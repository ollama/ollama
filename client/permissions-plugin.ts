import chmodr from 'chmodr'
import * as path from 'path'

interface PluginOptions {
  resourcePath: string
}

class PermissionsPlugin {
  options: PluginOptions

  constructor(options: PluginOptions) {
    this.options = options
  }

  apply(compiler: any) {
    compiler.hooks.afterEmit.tap('PermissionsPlugin', () => {
      chmodr(path.join(this.options.resourcePath), 0o755, err => {
        // this fails on the first call to suppress the error
      })
    })
  }
}

export default PermissionsPlugin
