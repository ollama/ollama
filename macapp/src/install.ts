import { exec as cbExec, spawn } from 'child_process'
import * as path from 'path'
import { promisify } from 'util'

const app = process && process.type === 'renderer' ? require('@electron/remote').app : require('electron').app
const ollama = app.isPackaged ? path.join(process.resourcesPath, 'ollama') : path.resolve(process.cwd(), '..', 'ollama')
const exec = promisify(cbExec)
const symlinkPath = '/usr/local/bin/ollama'

export async function installed() {
  const shells = ['/bin/zsh', '/bin/bash', '/usr/local/bin/fish'];
  const checks = shells.map(shell => 
    new Promise(resolve => {
      const proc = spawn(shell, ['-l', '-c', `which ollama`]);

      proc.on('error', () => {
        resolve(false); // if the shell isn't found, this will resolve false here
      });

      proc.on('close', code => {
        if (code === 0) {
          resolve(true); // ollama found, resolve true immediately
        } else {
          resolve(false);
        }
      });
    })
  );

  const results = await Promise.allSettled(checks)
  return results.some(result => result.status === 'fulfilled' && result.value === true)
}

export async function install() {
  const command = `do shell script "mkdir -p ${path.dirname(
    symlinkPath
  )} && ln -F -s \\"${ollama}\\" \\"${symlinkPath}\\"" with administrator privileges`

  await exec(`osascript -e '${command}'`)
}
