import { ipcRenderer } from 'electron'
import type { AppSettings } from './schema'

type ChangeListener = (patch: { path: string; value: any } | { full: AppSettings }) => void

const listeners = new Set<ChangeListener>()
let subscribed = false

function ensureSubscribed() {
  if (subscribed) return
  subscribed = true
  ipcRenderer.on('settings:changed', (_e, patch) => {
    for (const l of listeners) {
      try { l(patch) } catch { /* listener error ignored */ }
    }
  })
}

export async function fetchAllSettings(): Promise<AppSettings> {
  const res = await ipcRenderer.invoke('settings:get-all')
  if (res?.error) throw new Error(res.error)
  return res as AppSettings
}

export async function getSetting<T=any>(path: string): Promise<T> {
  const res = await ipcRenderer.invoke('settings:get', path)
  if (res?.error) throw new Error(res.error)
  return res as T
}

export async function setSetting(path: string, value: any): Promise<{ path: string; value: any }> {
  const res = await ipcRenderer.invoke('settings:set', path, value)
  if (res?.error) throw new Error(res.error)
  return res as { path: string; value: any }
}

export async function updateSettings(partial: Partial<AppSettings>): Promise<AppSettings> {
  const res = await ipcRenderer.invoke('settings:update', partial)
  if (res?.error) throw new Error(res.error)
  return res as AppSettings
}

export async function resetSettings(): Promise<AppSettings> {
  const res = await ipcRenderer.invoke('settings:reset')
  if (res?.error) throw new Error(res.error)
  return res as AppSettings
}

export function onSettingsChanged(fn: ChangeListener) { ensureSubscribed(); listeners.add(fn); return () => listeners.delete(fn) }
