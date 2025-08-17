import { spawn, ChildProcess } from 'child_process'
import { app, autoUpdater, dialog, Tray, Menu, BrowserWindow, MenuItemConstructorOptions, nativeTheme } from 'electron'
import Store from 'electron-store'
import winston from 'winston'
import 'winston-daily-rotate-file'
import * as path from 'path'

import { v4 as uuidv4 } from 'uuid'
import { installed } from './install'

require('@electron/remote/main').initialize()

if (require('electron-squirrel-startup')) {
  app.quit()
}

const store = new Store()

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

app.on('ready', () => {
  // Ensure renderer side electron-store IPC is initialized (avoids blank window if renderer Store() instantiation blocks)
  try {
    // @ts-ignore - initRenderer is a static we intentionally invoke before creating BrowserWindow(s)
    if (typeof (Store as any).initRenderer === 'function') {
      ;(Store as any).initRenderer()
    }
  } catch (e) {
    console.error('[bootstrap] Store.initRenderer failed', e)
  }
  const gotTheLock = app.requestSingleInstanceLock()
  if (!gotTheLock) {
    app.exit(0)
    return
  }

  app.on('second-instance', () => {
    if (app.hasSingleInstanceLock()) {
      app.releaseSingleInstanceLock()
    }

    if (proc) {
      proc.off('exit', restart)
      proc.kill()
    }

    app.exit(0)
  })

  app.focus({ steal: true })

  init()
})

function attachDebug(win: BrowserWindow, label: string) {
  if (!win) return
  win.webContents.on('did-fail-load', (_e, ec, desc, url) => {
    console.error(`[win:${label}] did-fail-load code=${ec} desc=${desc} url=${url}`)
  })
  win.webContents.on('crashed', () => {
    console.error(`[win:${label}] renderer crashed`)
  })
  win.webContents.on('render-process-gone', (_e, details) => {
    console.error(`[win:${label}] render-process-gone`, details)
  })
  win.webContents.on('dom-ready', () => {
    if (process.env.DEBUG_OLLAMA) {
      try { win.webContents.openDevTools({ mode: 'detach' }) } catch {}
    }
  })
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
  })

  require('@electron/remote/main').enable(welcomeWindow.webContents)

  console.log('[create] welcome window ->', MAIN_WINDOW_WEBPACK_ENTRY)
  welcomeWindow.loadURL(MAIN_WINDOW_WEBPACK_ENTRY)
  attachDebug(welcomeWindow, 'welcome')
  welcomeWindow.on('ready-to-show', () => welcomeWindow.show())
  welcomeWindow.on('closed', () => {
    if (process.platform === 'darwin') {
      app.dock.hide()
    }
  })
}

let tray: Tray | null = null
let mainWindow: BrowserWindow | null = null
let settingsWindow: BrowserWindow | null = null
let updateAvailable = false
const assetPath = app.isPackaged ? process.resourcesPath : path.join(__dirname, '..', '..', 'assets')

function trayIconPath() {
  return nativeTheme.shouldUseDarkColors
    ? updateAvailable
      ? path.join(assetPath, 'iconDarkUpdateTemplate.png')
      : path.join(assetPath, 'iconDarkTemplate.png')
    : updateAvailable
    ? path.join(assetPath, 'iconUpdateTemplate.png')
    : path.join(assetPath, 'iconTemplate.png')
}

function updateTrayIcon() {
  if (tray) {
    tray.setImage(trayIconPath())
  }
}

function openMainWindow() {
  if (mainWindow && !mainWindow.isDestroyed()) {
    if (mainWindow.isMinimized()) mainWindow.restore()
    mainWindow.focus()
    return
  }
  mainWindow = new BrowserWindow({
    width: 1100,
    height: 800,
    show: false,
    backgroundColor: '#0e0e0e',
    // For mac we go frameless to allow a seamless surface; keep framed on other platforms
    ...(process.platform === 'darwin'
      ? { frame: false as const, titleBarStyle: 'hiddenInset' as const, titleBarOverlay: { color: '#0e0e0e', symbolColor: '#ffffff', height: 32 } as any }
      : {}),
    webPreferences: { nodeIntegration: true, contextIsolation: false },
  })
  require('@electron/remote/main').enable(mainWindow.webContents)
  const url = `${MAIN_WINDOW_WEBPACK_ENTRY}?view=chat`
  console.log('[create] main window ->', url)
  mainWindow.loadURL(url)
  attachDebug(mainWindow, 'main')
  mainWindow.webContents.on('did-finish-load', () => {
    console.log('[main] did-finish-load')
  })
  mainWindow.webContents.on('dom-ready', () => {
    console.log('[main] dom-ready')
  })
  mainWindow.webContents.on('console-message', (_e, level, message, line, sourceId) => {
    console.log(`[renderer console l${level}] ${message} (${sourceId}:${line})`)
  })
  mainWindow.on('ready-to-show', () => mainWindow?.show())
  mainWindow.on('closed', () => { mainWindow = null })
}

function openSettingsWindow() {
  if (settingsWindow && !settingsWindow.isDestroyed()) {
    if (settingsWindow.isMinimized()) settingsWindow.restore()
    settingsWindow.focus()
    return
  }
  settingsWindow = new BrowserWindow({
    width: 900,
    height: 700,
    show: false,
    webPreferences: { nodeIntegration: true, contextIsolation: false },
  })
  require('@electron/remote/main').enable(settingsWindow.webContents)
  const url = `${MAIN_WINDOW_WEBPACK_ENTRY}?view=settings`
  console.log('[create] settings window ->', url)
  settingsWindow.loadURL(url)
  attachDebug(settingsWindow, 'settings')
  settingsWindow.on('ready-to-show', () => settingsWindow?.show())
  settingsWindow.on('closed', () => { settingsWindow = null })
}

function updateTray() {
  const updateItems: MenuItemConstructorOptions[] = [
    { label: 'An update is available', enabled: false },
    {
      label: 'Restart to update',
      click: () => autoUpdater.quitAndInstall(),
    },
    { type: 'separator' },
  ]

  const menu = Menu.buildFromTemplate([
    ...(updateAvailable ? updateItems : []),
    { label: 'Open Ollama', click: () => openMainWindow() },
    { label: 'Settings...', click: () => openSettingsWindow() },
    { type: 'separator' },
    { role: 'quit', label: 'Quit Ollama', accelerator: 'Command+Q' },
  ])

  if (!tray) {
    tray = new Tray(trayIconPath())
  }

  tray.setToolTip(updateAvailable ? 'An update is available' : 'Ollama')
  tray.setContextMenu(menu)
  tray.setImage(trayIconPath())

  nativeTheme.off('updated', updateTrayIcon)
  nativeTheme.on('updated', updateTrayIcon)
}

let proc: ChildProcess = null

function server() {
  // Allow skipping the embedded server for UI debugging
  if (process.env.OLLAMA_SKIP_SERVER === '1') {
    logger.info('[server] Skipping server start (OLLAMA_SKIP_SERVER=1)')
    return
  }

  const binary = app.isPackaged
    ? path.join(process.resourcesPath, 'ollama')
    : path.resolve(process.cwd(), '..', 'ollama')

  try {
    // Lazily require fs to reduce startup cost
    const fs = require('fs') as typeof import('fs')
    if (!fs.existsSync(binary)) {
      logger.error(`[server] binary missing at ${binary} – UI will still load. Set OLLAMA_SKIP_SERVER=1 to suppress this message.`)
      return // Do not crash; continue showing UI
    }
  } catch (e) {
    logger.error(`[server] existence check failed: ${(e as Error).message}`)
  }

  try {
    proc = spawn(binary, ['serve'])
  } catch (e) {
    logger.error(`[server] spawn failed: ${(e as Error).message}`)
    return
  }

  proc.on('error', err => {
    logger.error(`[server] process error: ${err.message}`)
  })

  proc.stdout.on('data', data => {
    logger.info(data.toString().trim())
  })

  proc.stderr.on('data', data => {
    logger.error(data.toString().trim())
  })

  proc.on('exit', code => {
    logger.error(`[server] exited with code ${code}`)
    restart()
  })
}

function restart() {
  setTimeout(server, 1000)
}

app.on('before-quit', () => {
  if (proc) {
    proc.off('exit', restart)
    proc.kill('SIGINT') // send SIGINT signal to the server, which also stops any loaded llms
  }
})

const updateURL = `https://ollama.com/api/update?os=${process.platform}&arch=${
  process.arch
}&version=${app.getVersion()}&id=${id()}`

let latest = ''
async function isNewReleaseAvailable() {
  try {
    const response = await fetch(updateURL)

    if (!response.ok) {
      return false
    }

    if (response.status === 204) {
      return false
    }

    const data = await response.json()

    const url = data?.url
    if (!url) {
      return false
    }

    if (latest === url) {
      return false
    }

    latest = url

    return true
  } catch (error) {
    logger.error(`update check failed - ${error}`)
    return false
  }
}

async function checkUpdate() {
  const available = await isNewReleaseAvailable()
  if (available) {
    logger.info('checking for update')
    autoUpdater.checkForUpdates()
  }
}

function init() {
  if (app.isPackaged) {
    checkUpdate()
    setInterval(() => {
      checkUpdate()
    }, 60 * 60 * 1000)
  }

  updateTray()

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

  server()

  const hasRun = !!store.get('first-time-run')
  const cliInstalled = installed()
  // Once user has completed first run, always show chat even if CLI temporarily missing
  const wantChat = hasRun
  console.log(`[init] hasRun=${hasRun} cliInstalled=${cliInstalled} -> ${wantChat ? 'open chat window' : 'open welcome window'}`)

  if (wantChat) {
    // Returning user: open main chat window only
    openMainWindow()
    if (process.platform === 'darwin') {
      // Keep dock visible for primary app usage
      app.dock.show()
    }
    app.setLoginItemSettings({ openAtLogin: app.getLoginItemSettings().openAtLogin })
  } else {
    // First run OR CLI missing: show welcome/onboarding window only
    firstRunWindow()
    if (process.platform === 'darwin') {
      try { app.dock.show() } catch {}
    }
    // Ensure we auto-start so users see onboarding again if they quit mid-way
    app.setLoginItemSettings({ openAtLogin: true })
  }
}

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('activate', () => {
  if (process.platform === 'darwin' && !mainWindow && store.get('first-time-run')) {
    openMainWindow()
  }
})

function id(): string {
  const id = store.get('id') as string

  if (id) {
    return id
  }

  const uuid = uuidv4()
  store.set('id', uuid)
  return uuid
}

autoUpdater.setFeedURL({ url: updateURL })

autoUpdater.on('error', e => {
  logger.error(`update check failed - ${e.message}`)
  console.error(`update check failed - ${e.message}`)
})

autoUpdater.on('update-downloaded', () => {
  updateAvailable = true
  updateTray()
})
