import * as fs from 'fs'
import { exec as cbExec } from 'child_process'
import * as path from 'path'
import { promisify } from 'util'

const app = process && process.type === 'renderer' ? require('@electron/remote').app : require('electron').app
const ollama = app.isPackaged ? path.join(process.resourcesPath, 'ollama') : path.resolve(process.cwd(), '..', 'ollama')
const exec = promisify(cbExec)
const defaultSymlinkPath = '/usr/local/bin/ollama'

export type PathType = 'default' | 'custom'

export async function isInstalled(pathType: PathType = 'default') {
  try {
    if (pathType === 'default') {
      return fs.existsSync(defaultSymlinkPath) && fs.readlinkSync(defaultSymlinkPath) === ollama
    } else if (pathType === 'custom') {
      // User has opted for custom path
      try {
        const result = await exec('which ollama')
        const customPath = result.stdout.trim()
        
        // Check if the custom path exists and is executable
        if (customPath && fs.existsSync(customPath)) {
          return true
        }
      } catch (error) {
        // 'which' command returned non-zero exit code (not found)
        return false
      }
    }
    return false
  } catch (error) {
    return false
  }
}

export async function install() {
  const command = `do shell script "mkdir -p ${path.dirname(
    defaultSymlinkPath
  )} && ln -F -s \\"${ollama}\\" \\"${defaultSymlinkPath}\\"" with administrator privileges`

  await exec(`osascript -e '${command}'`)
}
