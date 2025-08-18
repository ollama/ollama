import React, { useState } from 'react'
import { useSettings } from './SettingsContext'
import { resetSettings } from './rendererClient'

interface NavItem { id: string; label: string }
const sections: NavItem[] = [
  { id: 'appearance', label: 'Appearance' },
  { id: 'chat', label: 'Chat' },
  { id: 'quickask', label: 'Quick Ask' },
  { id: 'behavior', label: 'Behavior' },
  { id: 'privacy', label: 'Privacy' },
  { id: 'advanced', label: 'Advanced' },
]

export const SettingsPanel: React.FC = () => {
  const { settings, setValue, ready } = useSettings()
  const [active, setActive] = useState('appearance')
  const [resetBusy, setResetBusy] = useState(false)
  const canRender = ready && !!settings

  const nav = (
    <nav className='w-48 shrink-0 px-3 py-4 space-y-1 overflow-y-auto border-r border-white/5'>
      <div className='text-[11px] uppercase tracking-wider opacity-60 mb-2 font-semibold'>Settings</div>
      {sections.map(s => (
        <button
          key={s.id}
          onClick={() => setActive(s.id)}
          className={`w-full text-left px-3 py-2 rounded-md text-sm transition border border-transparent focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500/60 ${active===s.id ? 'bg-emerald-600/15 text-emerald-300 border-emerald-600/30' : 'hover:bg-white/5 text-gray-300'}`}
          aria-current={active===s.id}
        >{s.label}</button>) )}
      <div className='pt-4 mt-4 border-t border-white/10'>
        <button
          onClick={async () => {
            if (!confirm('Reset all settings to defaults?')) {
              return
            }
            setResetBusy(true)
            try {
              await resetSettings()
            } finally {
              setResetBusy(false)
            }
          }}
          className='w-full px-3 py-2 rounded-md text-xs bg-red-600/80 hover:bg-red-600 text-white disabled:opacity-50'
          disabled={resetBusy}
        >{resetBusy ? 'Resetting…' : 'Reset to Defaults'}</button>
      </div>
    </nav>
  )

  const appearance = canRender && (
    <Section id='appearance' active={active} title='Appearance' description='Visual customization of the desktop app window.'>
      <Toggle label='Translucent main window (macOS)' description='Recreates the window to apply vibrancy.' value={settings.ui.translucency} onChange={v=>setValue('ui.translucency', v)} />
      <div className='mt-5'>
        <Label>Theme</Label>
        <div className='flex gap-3 mt-1'>
          {(['system','dark','light'] as const).map(t => (
            <button key={t} onClick={()=>setValue('ui.theme', t)} className={`px-3 py-2 rounded-md text-xs border ${settings.ui.theme===t ? 'bg-emerald-600/70 border-emerald-500 text-white' : 'bg-[#1d1d1d] border-[#2a2a2a] hover:bg-[#242424] text-gray-300'}`}>{t}</button>
          ))}
        </div>
        <p className='text-[10px] opacity-60 mt-1'>System uses the OS appearance. Light/Dark override manually.</p>
      </div>
      <div className='mt-5'>
        <Label>Font scale</Label>
        <input type='range' min={0.85} max={1.3} step={0.01} value={settings.ui.fontScale} onChange={e=>setValue('ui.fontScale', Number(e.target.value))} className='w-60' />
        <div className='text-[10px] mt-1 opacity-60'>Current: {settings.ui.fontScale.toFixed(2)}</div>
      </div>
    </Section>
  )

  const chat = canRender && (
    <Section id='chat' active={active} title='Chat' description='Defaults affecting new chat turns.'>
      <Toggle label='Enable reasoning by default' value={settings.chat.reasoningEnabledByDefault} onChange={v=>setValue('chat.reasoningEnabledByDefault', v)} />
      <Toggle label='Show thinking traces' value={settings.chat.showThinking} onChange={v=>setValue('chat.showThinking', v)} />
      <div className='mt-5'>
        <Label>Default model</Label>
        <input type='text' value={settings.chat.defaultModel} onChange={e=>setValue('chat.defaultModel', e.target.value.trim())} className='mt-1 bg-[#181818] border border-[#2a2a2a] rounded px-2 py-1 text-sm w-60 focus:border-emerald-600 outline-none' />
        <p className='text-[10px] opacity-60 mt-1'>Model used when starting a new session if none selected.</p>
      </div>
    </Section>
  )

  const quickAsk = canRender && (
    <Section id='quickask' active={active} title='Quick Ask' description='Ephemeral overlay panel behavior.'>
      <Toggle label='Pinned by default' value={settings.quickAsk.pinnedDefault} onChange={v=>setValue('quickAsk.pinnedDefault', v)} />
      <Toggle label='Auto-close after copy' value={settings.quickAsk.autoCloseOnCopy} onChange={v=>setValue('quickAsk.autoCloseOnCopy', v)} />
      <Toggle label='Show thinking traces' value={settings.quickAsk.showThinking} onChange={v=>setValue('quickAsk.showThinking', v)} />
    </Section>
  )

  const behavior = canRender && (
    <Section id='behavior' active={active} title='Behavior' description='Application-level behaviors.'>
      <Toggle label='Auto check for updates' value={settings.behavior.autoCheckUpdates} onChange={v=>setValue('behavior.autoCheckUpdates', v)} />
      <Toggle label='Launch at login' value={settings.behavior.launchAtLogin} onChange={v=>setValue('behavior.launchAtLogin', v)} />
      <div className='mt-5'>
        <Label>Global shortcut</Label>
        <code className='block bg-[#181818] border border-[#2a2a2a] rounded px-2 py-1 text-[11px] w-fit mt-1'>{settings.behavior.globalShortcut}</code>
        <p className='text-[10px] opacity-60 mt-1'>Change shortcut in a future update.</p>
      </div>
    </Section>
  )

  const privacy = canRender && (
    <Section id='privacy' active={active} title='Privacy' description='Telemetry & data collection preferences.'>
      <Toggle label='Enable telemetry' value={settings.privacy.telemetryEnabled} onChange={v=>setValue('privacy.telemetryEnabled', v)} />
      <p className='text-[10px] opacity-60 mt-2 max-w-sm'>Anonymous usage metrics help prioritize features. No chat content is sent.</p>
    </Section>
  )

  const advanced = canRender && (
    <Section id='advanced' active={active} title='Advanced' description='Low-level or experimental options.'>
      <p className='text-[12px] opacity-60'>More coming soon (context window hints, performance tuning, network modes).</p>
    </Section>
  )

  return (
    <div className='flex h-screen text-gray-200 bg-[#0f0f0f] font-[system-ui] text-sm'>
      {nav}
      <main className='flex-1 overflow-y-auto px-10 py-8 space-y-16'>
        {appearance}
        {chat}
        {quickAsk}
        {behavior}
        {privacy}
        {advanced}
        {!canRender && <div className='opacity-50 text-xs'>Loading settings…</div>}
      </main>
    </div>
  )
}

const Section: React.FC<{ id: string; active: string; title: string; description?: string; children: React.ReactNode }> = ({ id, active, title, description, children }) => {
  return (
    <section id={id} aria-hidden={active !== id} className='space-y-4'>
      <header>
        <h3 className='text-base font-semibold mb-1'>{title}</h3>
        {description && <p className='text-[11px] opacity-60 tracking-wide'>{description}</p>}
      </header>
      <div className='space-y-4'>
        {children}
      </div>
    </section>
  )
}

const Toggle: React.FC<{ label: string; value: boolean; onChange: (v:boolean)=>void; description?: string }> = ({ label, value, onChange, description }) => (
  <label className='flex items-start gap-3 cursor-pointer group'>
    <input type='checkbox' checked={value} onChange={e=>onChange(e.target.checked)} className='mt-[2px]' />
    <span className='flex flex-col'>
      <span className='text-[13px]'>{label}</span>
      {description && <span className='text-[10px] opacity-60 mt-[2px]'>{description}</span>}
    </span>
  </label>
)

const Label: React.FC<{ children: React.ReactNode }> = ({ children }) => <div className='text-[11px] font-medium tracking-wide uppercase opacity-60'>{children}</div>
