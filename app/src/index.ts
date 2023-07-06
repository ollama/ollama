import { spawn, exec } from 'child_process'
import { app, autoUpdater, dialog, Tray, Menu, nativeTheme, Notification } from 'electron'
import * as path from 'path'
import * as fs from 'fs'

require('@electron/remote/main').initialize()

let tray: Tray | null = null
let secondInstance: Boolean = false
const SingleInstanceLock = app.requestSingleInstanceLock()

const createSystemtray = () => {
  let brightModeIconPath = path.join(__dirname, '..', '..', 'assets', 'ollama_icon_dark_16x16.png')
  let darkModeIconPath = path.join(__dirname, '..', '..', 'assets', 'ollama_icon_bright_16x16.png')

  if (app.isPackaged) {
    brightModeIconPath = path.join(process.resourcesPath, 'ollama_icon_dark_16x16@2x.png')
    darkModeIconPath = path.join(process.resourcesPath, 'ollama_icon_bright_16x16@2x.png')
  }

  tray = new Tray(brightModeIconPath)

  if (process.platform === 'darwin') {
    tray.setImage(nativeTheme.shouldUseDarkColors ? darkModeIconPath : brightModeIconPath)
    nativeTheme.on('updated', () => {
      tray.setImage(nativeTheme.shouldUseDarkColors ? darkModeIconPath : brightModeIconPath)
    })
  }


  const contextMenu = Menu.buildFromTemplate([
    { role: 'quit', label: 'Quit Ollama', accelerator: 'Command+Q' },
  ])

  tray.setContextMenu(contextMenu)
  tray.setToolTip('Ollama')
}

// Handle creating/removing shortcuts on Windows when installing/uninstalling.
if (require('electron-squirrel-startup')) {
  app.quit()
}

const ollama = path.join(process.resourcesPath, 'ollama')

// if the app is packaged then run the server
if (app.isPackaged) {
  // Start the executable
  console.log(`Starting server`)
  const proc = spawn(ollama, ['serve'])
  proc.stdout.on('data', data => {
    console.log(`server: ${data}`)
  })
  proc.stderr.on('data', data => {
    console.error(`server: ${data}`)
  })

  process.on('exit', () => {
    proc.kill()
  })
}

function server() {
  const binary = app.isPackaged
    ? path.join(process.resourcesPath, 'ollama')
    : path.resolve(__dirname, '..', '..', 'ollama')

  console.log(`Starting server`)
  const proc = spawn(binary, ['serve'])
  proc.stdout.on('data', data => {
    console.log(`server: ${data}`)
  })
  proc.stderr.on('data', data => {
    console.error(`server: ${data}`)
  })

  process.on('exit', () => {
    proc.kill()
  })
}

function installCLI() {
  const symlinkPath = '/usr/local/bin/ollama'

  if (fs.existsSync(symlinkPath) && fs.readlinkSync(symlinkPath) === ollama) {
    return
  }

  dialog
    .showMessageBox({
      type: 'info',
      title: 'Ollama CLI installation',
      message: 'To install the Ollama CLI, we need to ask you for administrator privileges.',
      buttons: ['OK'],
    })
    .then(result => {
      if (result.response === 0) {
        const command = `
    do shell script "ln -F -s ${ollama} /usr/local/bin/ollama" with administrator privileges
    `
        exec(`osascript -e '${command}'`, (error: Error | null, stdout: string, stderr: string) => {
          if (error) {
            console.error(`exec error: ${error}`)
            return
          }
          console.log(`stdout: ${stdout}`)
          console.error(`stderr: ${stderr}`)
        })
      }
    })
}

if (!SingleInstanceLock) {  
  app.quit()
} else {
  app.on('ready', () => {
    if (process.platform === 'darwin') {
      app.dock.hide()
  
      if (!app.isInApplicationsFolder()) {
        const chosen = dialog.showMessageBoxSync({
          type: 'question',
          buttons: ['Move to Applications', 'Do Not Move'],
          message: 'Move Ollama to the Applications directory?',
          defaultId: 0,
          cancelId: 1
        })
  
        if (chosen === 0) {
          try {
            app.moveToApplicationsFolder({
              conflictHandler: (conflictType) => {
                if (conflictType === 'existsAndRunning') {
                  dialog.showMessageBoxSync({
                    type: 'info',
                    message: 'Cannot move to Applications directory',
                    detail: 'Another version of Ollama is currently running from your Applications directory. Close it first and try again.'
                  })
                }
                return true
              }
            })
            return
          } catch (e) {
            console.error('Failed to move to applications folder')
            console.error(e)
          }
        }
      }
    }
  
    createSystemtray()
  
    if (app.isPackaged) {
      installCLI()
    }
  })
}


// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

// In this file you can include the rest of your app's specific main process
// code. You can also put them in separate files and import them here.
autoUpdater.setFeedURL({
  url: `https://ollama.ai/api/update?os=${process.platform}&arch=${process.arch}&version=${app.getVersion()}`,
})

if (app.isPackaged) {
  autoUpdater.checkForUpdates()
  setInterval(() => {
    autoUpdater.checkForUpdates()
  }, 60000)
}

autoUpdater.on('error', e => {
  console.error('update check failed', e)
})

autoUpdater.on('update-downloaded', (event, releaseNotes, releaseName) => {
  dialog
    .showMessageBox({
      type: 'info',
      buttons: ['Restart Now', 'Later'],
      title: 'New update available',
      message: process.platform === 'win32' ? releaseNotes : releaseName,
      detail: 'A new version of Ollama is available. Restart to apply the update.',
    })
    .then(returnValue => {
      if (returnValue.response === 0) autoUpdater.quitAndInstall()
    })
})
