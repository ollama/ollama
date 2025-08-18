import Store from 'electron-store'

export interface AppSettings {
  ui: {
    translucency: boolean
    theme: 'system' | 'dark' | 'light'
    sidebarCollapsed: boolean
    fontScale: number
  }
  chat: {
    defaultModel: string
    reasoningEnabledByDefault: boolean
    showThinking: boolean
    maxContextHint?: number
  }
  quickAsk: {
    pinnedDefault: boolean
    autoCloseOnCopy: boolean
    showThinking: boolean
  }
  behavior: {
    globalShortcut: string
    autoCheckUpdates: boolean
    launchAtLogin: boolean
  }
  privacy: {
    telemetryEnabled: boolean
  }
  _meta: { version: number }
}

export const CURRENT_VERSION = 1

export function defaultSettings(): AppSettings {
  return {
    ui: {
      translucency: false,
      theme: 'system',
      sidebarCollapsed: false,
      fontScale: 1.0,
    },
    chat: {
      defaultModel: 'gemma3n',
      reasoningEnabledByDefault: true,
      showThinking: true,
    },
    quickAsk: {
      pinnedDefault: false,
      autoCloseOnCopy: false,
      showThinking: true,
    },
    behavior: {
      globalShortcut: process.platform === 'darwin' ? 'Command+Shift+Space' : 'Control+Shift+Space',
      autoCheckUpdates: true,
      launchAtLogin: true,
    },
    privacy: {
      telemetryEnabled: true,
    },
    _meta: { version: CURRENT_VERSION },
  }
}

// Shallow type guards / coercion helpers
function coerceBoolean(v: any, d: boolean): boolean { return typeof v === 'boolean' ? v : d }
function coerceString<T extends string>(v: any, allowed: readonly T[], d: T): T { return allowed.includes(v) ? v as T : d }
function coerceNumber(v: any, d: number): number { return typeof v === 'number' && !Number.isNaN(v) ? v : d }

export function migrateAndValidate(raw: any): AppSettings {
  const defs = defaultSettings()
  let out: AppSettings = { ...defs, _meta: { version: CURRENT_VERSION } } as AppSettings
  if (!raw || typeof raw !== 'object') raw = {}

  // Start with defaults then overlay existing values coercively
  out.ui = {
    translucency: coerceBoolean(raw.ui?.translucency, defs.ui.translucency),
    theme: coerceString(raw.ui?.theme, ['system','dark','light'] as const, defs.ui.theme),
    sidebarCollapsed: coerceBoolean(raw.ui?.sidebarCollapsed, defs.ui.sidebarCollapsed),
    fontScale: coerceNumber(raw.ui?.fontScale, defs.ui.fontScale),
  }
  out.chat = {
    defaultModel: typeof raw.chat?.defaultModel === 'string' ? raw.chat.defaultModel : defs.chat.defaultModel,
    reasoningEnabledByDefault: coerceBoolean(raw.chat?.reasoningEnabledByDefault, defs.chat.reasoningEnabledByDefault),
    showThinking: coerceBoolean(raw.chat?.showThinking, defs.chat.showThinking),
    maxContextHint: typeof raw.chat?.maxContextHint === 'number' ? raw.chat.maxContextHint : undefined,
  }
  out.quickAsk = {
    pinnedDefault: coerceBoolean(raw.quickAsk?.pinnedDefault, defs.quickAsk.pinnedDefault),
    autoCloseOnCopy: coerceBoolean(raw.quickAsk?.autoCloseOnCopy, defs.quickAsk.autoCloseOnCopy),
    showThinking: coerceBoolean(raw.quickAsk?.showThinking, defs.quickAsk.showThinking),
  }
  out.behavior = {
    globalShortcut: typeof raw.behavior?.globalShortcut === 'string' ? raw.behavior.globalShortcut : defs.behavior.globalShortcut,
    autoCheckUpdates: coerceBoolean(raw.behavior?.autoCheckUpdates, defs.behavior.autoCheckUpdates),
    launchAtLogin: coerceBoolean(raw.behavior?.launchAtLogin, defs.behavior.launchAtLogin),
  }
  out.privacy = {
    telemetryEnabled: coerceBoolean(raw.privacy?.telemetryEnabled, defs.privacy.telemetryEnabled),
  }

  // Legacy migrations (only if version missing or < CURRENT_VERSION)
  const legacy = raw._meta?.version == null || raw._meta.version < CURRENT_VERSION
  if (legacy) {
    try {
      const rootStore = new Store()
      // mainWindowTranslucent legacy
      if (rootStore.get('mainWindowTranslucent') != null) {
        out.ui.translucency = !!rootStore.get('mainWindowTranslucent')
      }
    } catch {}
    try {
      const qaStore = new Store({ name: 'quick-ask' }) as any
      if (qaStore) {
        if (qaStore.get('pinned') != null) out.quickAsk.pinnedDefault = !!qaStore.get('pinned')
        if (qaStore.get('showThinking') != null) out.quickAsk.showThinking = !!qaStore.get('showThinking')
        if (qaStore.get('autoCloseOnCopy') != null) out.quickAsk.autoCloseOnCopy = !!qaStore.get('autoCloseOnCopy')
      }
    } catch {}
    try {
      const chatPrefs = new Store({ name: 'chat-preferences' }) as any
      if (chatPrefs.get('showThinking') != null) out.chat.showThinking = !!chatPrefs.get('showThinking')
    } catch {}
  }

  return out
}
