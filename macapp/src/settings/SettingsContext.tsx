import React, { createContext, useContext, useEffect, useState, useCallback, useMemo } from 'react'
import type { AppSettings } from './schema'
import { fetchAllSettings, onSettingsChanged, setSetting, updateSettings } from './rendererClient'

interface SettingsContextValue {
  settings: AppSettings | null
  ready: boolean
  setValue: (path: string, value: any) => Promise<void>
  update: (partial: Partial<AppSettings>) => Promise<void>
}

const SettingsContext = createContext<SettingsContextValue | undefined>(undefined)

export const SettingsProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [settings, setSettings] = useState<AppSettings | null>(null)
  const [ready, setReady] = useState(false)

  useEffect(() => {
    let mounted = true
    fetchAllSettings().then(s => { if (mounted) { setSettings(s); setReady(true) } }).catch(err => { console.error('[settings] load failed', err); setReady(true) })
    const off = onSettingsChanged(patch => {
      setSettings(prev => {
        if (!prev) return prev
        if ('full' in patch) return patch.full
        const segs = patch.path.split('.')
        const clone: any = { ...prev }
        let cursor = clone
        for (let i=0;i<segs.length-1;i++) { cursor[segs[i]] = { ...cursor[segs[i]] }; cursor = cursor[segs[i]] }
        cursor[segs[segs.length-1]] = patch.value
        return clone
      })
    })
    return () => { mounted = false; off() }
  }, [])

  const setValue = useCallback(async (path: string, value: any) => {
    if (!settings) return
    // Optimistic update
    setSettings(prev => {
      if (!prev) return prev
      const segs = path.split('.')
      const clone: any = { ...prev }
      let cursor = clone
      for (let i=0;i<segs.length-1;i++) { cursor[segs[i]] = { ...cursor[segs[i]] }; cursor = cursor[segs[i]] }
      cursor[segs[segs.length-1]] = value
      return clone
    })
    try { await setSetting(path, value) } catch (e) { console.error('[settings] set failed', e); fetchAllSettings().then(setSettings) }
  }, [settings])

  const update = useCallback(async (partial: Partial<AppSettings>) => {
    try { await updateSettings(partial) } catch (e) { console.error('[settings] update failed', e); fetchAllSettings().then(setSettings) }
  }, [])

  const contextValue = useMemo(() => ({ settings, ready, setValue, update }), [settings, ready, setValue, update])
  return <SettingsContext.Provider value={contextValue}>{children}</SettingsContext.Provider>
}

export function useSettings(): SettingsContextValue {
  const ctx = useContext(SettingsContext)
  if (!ctx) throw new Error('useSettings must be used within SettingsProvider')
  return ctx
}
