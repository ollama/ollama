import { ipcMain, webContents } from 'electron'
import Store from 'electron-store'
import { AppSettings, migrateAndValidate, defaultSettings, CURRENT_VERSION } from './schema'

const SETTINGS_NAMESPACE = 'app-settings'

const rawStore = new Store<{ settings?: AppSettings }>({ name: SETTINGS_NAMESPACE })
let settings: AppSettings = migrateAndValidate(rawStore.get('settings'))
rawStore.set('settings', settings)

const listeners = new Set<(s: AppSettings, patch?: { path: string; value: any }) => void>()

function notify(patch?: { path: string; value: any }) {
  for (const l of listeners) {
    try { l(settings, patch) } catch {/* listener error ignored */}
  }
  // Broadcast to all renderer processes
  for (const wc of webContents.getAllWebContents()) {
    try { wc.send('settings:changed', patch ? { path: patch.path, value: patch.value } : { full: settings }) } catch {/* broadcast error ignored */}
  }
}

export function getSettings(): AppSettings { return settings }
export function onSettingsChanged(fn: (s: AppSettings, patch?: { path: string; value: any }) => void) { listeners.add(fn); return () => listeners.delete(fn) }

function setByPath(path: string, value: any): boolean {
  const segs = path.split('.')
  let cursor: any = settings
  for (let i=0;i<segs.length-1;i++) {
    const k = segs[i]
    if (cursor[k] == null || typeof cursor[k] !== 'object') return false
    cursor = cursor[k]
  }
  const last = segs[segs.length-1]
  if (!(last in cursor)) return false
  cursor[last] = value
  return true
}

function getByPath(path: string): any {
  const segs = path.split('.')
  let cursor: any = settings
  for (const k of segs) {
    if (cursor == null) return undefined
    cursor = cursor[k]
  }
  return cursor
}

export function applyPatch(path: string, value: any) {
  if (!path) throw new Error('path required')
  if (!setByPath(path, value)) throw new Error('invalid path')
  // Re-validate entire structure to coerce types; then persist
  settings = migrateAndValidate(settings)
  rawStore.set('settings', settings)
  notify({ path, value: getByPath(path) })
}

export function updateBulk(partial: Partial<AppSettings>) {
  settings = migrateAndValidate({ ...settings, ...partial, _meta: { version: CURRENT_VERSION } })
  rawStore.set('settings', settings)
  notify()
}

export function resetSettings() {
  settings = defaultSettings()
  rawStore.set('settings', settings)
  notify()
}

// IPC wiring
ipcMain.handle('settings:get-all', () => getSettings())
ipcHandle('settings:get', (path: string) => getByPath(path))
ipcHandle('settings:set', (path: string, value: any) => { applyPatch(path, value); return { path, value: getByPath(path) } })
ipcHandle('settings:update', (partial: Partial<AppSettings>) => { updateBulk(partial); return getSettings() })
ipcHandle('settings:reset', () => { resetSettings(); return getSettings() })

function ipcHandle(channel: string, fn: (...args: any[]) => any) {
  ipcMain.handle(channel, (_e, ...args) => {
    try { return fn(...args) } catch (e: any) { return { error: e.message || String(e) } }
  })
}
