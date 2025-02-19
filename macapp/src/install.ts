import * as fs from 'fs'
import { spawn } from 'child_process'
import * as path from 'path'

const app = process && process.type === 'renderer' ? require('@electron/remote').app : require('electron').app
const ollama = app.isPackaged ? path.join(process.resourcesPath, 'ollama') : path.resolve(process.cwd(), '..', 'ollama')
const symlinkPath = '/usr/local/bin/ollama'

export function installed(): boolean {
  return fs.existsSync(symlinkPath) && fs.readlinkSync(symlinkPath) === ollama
}

function validPath(targetPath: string): boolean {
  const normalized = path.normalize(targetPath)
  return !(/[;&|`$(){}[\]<>]/.test(normalized) || normalized.includes('..'))
}

export async function install(): Promise<void> {
  if (!validPath(ollama) || !validPath(symlinkPath)) {
    throw new Error('Invalid path format')
  }

  await fs.promises.mkdir(path.dirname(symlinkPath), { recursive: true })
    .catch(err => err.code === 'EEXIST' ? null : Promise.reject(err))

  const process = spawn('osascript', [
    '-e',
    `do shell script "ln -F -s '${path.normalize(ollama)}' '${path.normalize(symlinkPath)}'" with administrator privileges`
  ])

  await new Promise<void>((resolve, reject) => {
    process.on('error', reject)
    process.on('close', code => code === 0 ? resolve() : reject(new Error(`Failed with code ${code}`)))
  })
}
