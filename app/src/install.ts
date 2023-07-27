import * as fs from 'fs'
import { exec as cbExec } from 'child_process'
import * as path from 'path'
import { promisify } from 'util'

const app = process && process.type === 'renderer' ? require('@electron/remote').app : require('electron').app
const ollama = app.isPackaged ? path.join(process.resourcesPath, 'ollama') : path.resolve(process.cwd(), '..', 'ollama')
const exec = promisify(cbExec)
const symlinkPath = '/usr/local/bin/ollama'

export function installed() {
  return fs.existsSync(symlinkPath) && fs.readlinkSync(symlinkPath) === ollama
}

export async function install() {
  const command = `do shell script "mkdir -p ${path.dirname(
    symlinkPath
  )} && ln -F -s \\"${ollama}\\" \\"${symlinkPath}\\"" with administrator privileges`

  await exec(`osascript -e '${command}'`)
}
