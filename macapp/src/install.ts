import * as fs from 'fs';
import { exec as cbExec } from 'child_process';
import * as path from 'path';
import { promisify } from 'util';

const app = process && process.type === 'renderer' ? require('@electron/remote').app : require('electron').app;
const ollama = app.isPackaged ? path.join(process.resourcesPath, 'ollama') : path.resolve(process.cwd(), '..', 'ollama');
const exec = promisify(cbExec);
const symlinkPath = '/usr/local/bin/ollama';

// Check if Ollama CLI is in the user's PATH
function isOllamaInPath(): boolean {
  try {
    const pathEnv = process.env.PATH || '';
    const paths = pathEnv.split(path.delimiter);
    return paths.some(p => fs.existsSync(path.join(p, 'ollama')));
  } catch {
    return false;
  }
}

// Check if the symlink is correctly set up
export function installed() {
  return fs.existsSync(symlinkPath) && fs.readlinkSync(symlinkPath) === ollama;
}

// Main installation logic with user prompt handling
export async function install() {

  // Check if the symlink already exists and is correct
  if (installed()) {
    console.log('Ollama CLI is already symlinked at the default location.');
    return;
  }

  // Check if Ollama CLI is already in PATH
  if (isOllamaInPath()) {
    console.log('Ollama CLI is already available in the PATH.');
    return;
  }
  await createSymlink();
}


// Create the symlink
async function createSymlink() {
  const command = `do shell script "mkdir -p ${path.dirname(
    symlinkPath
  )} && ln -F -s \\"${ollama}\\" \\"${symlinkPath}\\"" with administrator privileges`;

  try {
    await exec(`osascript -e '${command}'`);
    console.log('Symlink created successfully.');
  } catch (error) {
    console.error('Error creating symlink:', error);
  }
}
