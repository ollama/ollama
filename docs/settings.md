# Settings System

This document describes the (new) unified application settings infrastructure used by the macOS desktop UI.

## Overview

Settings are now stored in a single `electron-store` namespace: `app-settings`. A validated, versioned schema
is defined in `macapp/src/settings/schema.ts` and accessed exclusively through a main‑process store and
IPC helpers exposed to renderer processes.

Key pieces:

| File | Purpose |
|------|---------|
| `schema.ts` | Types (`AppSettings`), defaults, migration + validation (`migrateAndValidate`). |
| `mainStore.ts` | Holds the in‑memory canonical settings object, persists via `electron-store`, exposes IPC handlers, broadcasts change events. |
| `rendererClient.ts` | Thin async IPC client (get / set / update / reset) + change event subscription for renderer. |
| `SettingsContext.tsx` | React provider + hook (`useSettings`) offering optimistic update (`setValue`) and bulk `update` APIs. |

## Shape

```txt
ui: { translucency, theme, sidebarCollapsed, fontScale }
chat: { defaultModel, reasoningEnabledByDefault, showThinking, maxContextHint? }
quickAsk: { pinnedDefault, autoCloseOnCopy, showThinking }
behavior: { globalShortcut, autoCheckUpdates, launchAtLogin }
privacy: { telemetryEnabled }
_meta: { version }
```

## Renderer Usage

Wrap your root tree once:

```tsx
<SettingsProvider>
  <App />
</SettingsProvider>
```

Consume settings anywhere:

```tsx
const { settings, ready, setValue } = useSettings()
if (!ready || !settings) return null
return <Toggle checked={settings.chat.showThinking} onChange={v => setValue('chat.showThinking', v)} />
```

`setValue(path, value)` performs an optimistic update then confirms with the main process (falling back to a full refresh if it fails). Paths must point to existing leaf keys.

Use `update(partial)` for multi‑field bulk updates.

## Migration

On first load (or when `_meta.version` is missing / lower than `CURRENT_VERSION`), legacy keys are imported:

* Root store `mainWindowTranslucent` -> `ui.translucency`
* Quick ask store (`quick-ask`) keys: `pinned`, `showThinking`, `autoCloseOnCopy` -> `quickAsk.*`
* Chat preferences store (`chat-preferences`) key `showThinking` -> `chat.showThinking`

Legacy stores are left in place (read‑only) for now; future cleanup can safely delete those keys once a stable release including this migration has been widely adopted.

## Main Process Reactions

`index.ts` subscribes to settings changes. Currently it reacts to `ui.translucency` to recreate the main window on macOS (Electron cannot reliably live‑toggle vibrancy). It also mirrors the value back to the legacy `mainWindowTranslucent` key for temporary compatibility.

Add additional reactions by importing `onSettingsChanged` and filtering on `patch.path`.

## Adding New Settings

1. Extend `AppSettings` and `defaultSettings()`.
2. Update `migrateAndValidate` to coerce new values and provide sensible defaults.
3. (Optional) Add legacy migration snippet before returning.
4. Access in renderer via `settings.<group>.<key>` and mutate with `setValue('group.key', newValue)`.

## Error Handling / Validation

All `set` operations re‑validate the entire structure through `migrateAndValidate` ensuring type safety and future forward compatibility. Unknown paths are rejected in the main process (renderer optimistic change will be overwritten on refresh).

## Testing Tips

* Toggle translucency and confirm the main window re‑creates with vibrancy changes.
* Toggle *Show Thinking* in chat; open Quick Ask and confirm the toggle state is shared (one global source of truth now).
* Change Quick Ask *Pin* default then close & re‑open the overlay; the initial state should reflect the updated default.

## Future Improvements

* Remove legacy store keys after one or two release cycles.
* Persist per‑model overrides (if needed) in a nested `chat.models` map rather than additional top‑level booleans.
* Add schema version bump & targeted migrations for breaking changes.

---
Questions or enhancements: open a PR or issue referencing this document.
