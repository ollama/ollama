import { spawn } from 'child_process'
import { app, autoUpdater, dialog, Tray, Menu, BrowserWindow, nativeTheme } from 'electron'
import Store from 'electron-store'
import winston from 'winston'
import 'winston-daily-rotate-file'
import * as path from 'path'

import { analytics, id } from './telemetry'
import { installed } from './install'

require('@electron/remote/main').initialize()

const store = new Store()
let tray: Tray | null = null
let welcomeWindow: BrowserWindow | null = null

declare const MAIN_WINDOW_WEBPACK_ENTRY: string

const logger = winston.createLogger({
  transports: [
    new winston.transports.Console(),
    new winston.transports.File({
      filename: path.join(app.getPath('home'), '.ollama', 'logs', 'server.log'),
      maxsize: 1024 * 1024 * 20,
      maxFiles: 5,
    }),
  ],
  format: winston.format.printf(info => info.message),
})

const SingleInstanceLock = app.requestSingleInstanceLock()
if (!SingleInstanceLock) {
  app.quit()
}

function firstRunWindow() {
  // Create the browser window.
  welcomeWindow = new BrowserWindow({
    width: 400,
    height: 500,
    frame: false,
    fullscreenable: false,
    resizable: false,
    movable: true,
    show: false,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
    alwaysOnTop: true,
  })

  require('@electron/remote/main').enable(welcomeWindow.webContents)

  // and load the index.html of the app.
  welcomeWindow.loadURL(MAIN_WINDOW_WEBPACK_ENTRY)

  welcomeWindow.on('ready-to-show', () => welcomeWindow.show())

  // for debugging
  // welcomeWindow.webContents.openDevTools()

  if (process.platform === 'darwin') {
    app.dock.hide()
  }
}

function createSystemtray() {
  let iconPath = nativeTheme.shouldUseDarkColors
    ? path.join(__dirname, '..', '..', 'assets', 'ollama_icon_16x16Template.png') 
    : path.join(__dirname, '..', '..', 'assets', 'ollama_outline_icon_16x16Template.png')

  if (app.isPackaged) {
    iconPath = nativeTheme.shouldUseDarkColors
    ? path.join(process.resourcesPath, 'ollama_icon_16x16Template.png') 
    : path.join(process.resourcesPath, 'ollama_outline_icon_16x16Template.png')
  }

  tray = new Tray(iconPath)

  nativeTheme.on('updated', function theThemeHasChanged () {
    if (nativeTheme.shouldUseDarkColors) {
      app.isPackaged 
        ? tray.setImage(path.join(process.resourcesPath, 'ollama_icon_16x16Template.png')) 
        : tray.setImage(path.join(__dirname, '..', '..', 'assets', 'ollama_icon_16x16Template.png'))
    } else {
      app.isPackaged 
        ? tray.setImage(path.join(process.resourcesPath, 'ollama_outline_icon_16x16Template.png')) 
        : tray.setImage(path.join(__dirname, '..', '..', 'assets', 'ollama_outline_icon_16x16Template.png'))
    }
  })

  const contextMenu = Menu.buildFromTemplate([{ role: 'quit', label: 'Quit Ollama', accelerator: 'Command+Q' }])

  tray.setContextMenu(contextMenu)
  tray.setToolTip('Ollama')
}

if (require('electron-squirrel-startup')) {
  app.quit()
}

function server() {
  const binary = app.isPackaged
  ? path.join(process.resourcesPath, 'ollama')
  : path.resolve(process.cwd(), '..', 'ollama');

  const proc = spawn(binary, ['serve']);

  proc.stdout.on('data', data => {
    logger.info(data.toString().trim());
  });

  proc.stderr.on('data', data => {
    logger.error(data.toString().trim());
  });
    

  proc.on('exit', (code, signal) => {
    if (code === 0 || code === null) {
      logger.info('Server has stopped.');
      setTimeout(server, 5000);
    } else {
      logger.error(`Server exited with code: ${code}, signal: ${signal}`);
      setTimeout(server, 3000);
    }
  });

  app.on('before-quit', () => {
    proc.off('exit', server);
    proc.kill();
  });
}

if (process.platform === 'darwin') {
  app.dock.hide()
}

app.on('ready', () => {
  if (process.platform === 'darwin') {
    if (app.isPackaged) {
      if (!app.isInApplicationsFolder()) {
        const chosen = dialog.showMessageBoxSync({
          type: 'question',
          buttons: ['Move to Applications', 'Do Not Move'],
          message: 'Ollama works best when run from the Applications directory.',
          defaultId: 0,
          cancelId: 1,
        })

        if (chosen === 0) {
          try {
            app.moveToApplicationsFolder({
              conflictHandler: conflictType => {
                if (conflictType === 'existsAndRunning') {
                  dialog.showMessageBoxSync({
                    type: 'info',
                    message: 'Cannot move to Applications directory',
                    detail:
                      'Another version of Ollama is currently running from your Applications directory. Close it first and try again.',
                  })
                }
                return true
              },
            })
            return
          } catch (e) {
            logger.error(`[Move to Applications] Failed to move to applications folder - ${e.message}}`)
          }
        }
      }
    }
  }

  createSystemtray()
  server()

  if (store.get('first-time-run') && installed()) {
    app.setLoginItemSettings({ openAtLogin: app.getLoginItemSettings().openAtLogin })
    return
  }

  // This is the first run or the CLI is no longer installed
  app.setLoginItemSettings({ openAtLogin: true })
  firstRunWindow()
})

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

async function heartbeat() {
  analytics.track({
    anonymousId: id(),
    event: 'heartbeat',
    properties: {
      version: app.getVersion(),
    },
  })
}

if (app.isPackaged) {
  heartbeat()
  autoUpdater.checkForUpdates()
  setInterval(() => {
    heartbeat()
    autoUpdater.checkForUpdates()
  }, 60 * 60 * 1000)
}

autoUpdater.on('error', e => {
  logger.error(`update check failed - ${e.message}`)
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
