import { useState, useEffect } from 'react'
import { ipcRenderer } from 'electron'
import './chat/tokens.css'
import copy from 'copy-to-clipboard'
import { CheckIcon, DocumentDuplicateIcon } from '@heroicons/react/24/outline'
import Store from 'electron-store'
import { getCurrentWindow } from '@electron/remote'

import { install } from './install'
import OllamaIcon from './ollama.svg'
import { ChatView } from './chat/ChatView'
import { SettingsPanel } from './settings/SettingsPanel'
import { QuickAskPanel } from './quickask/QuickAskPanel'

const store = new Store()

enum Step {
  WELCOME = 0,
  CLI,
  FINISH,
}

export default function () {
  const standaloneQuick = false // legacy: quick ask now uses its own dedicated window/entry; embedded mode disabled
  const [view, setView] = useState<'onboarding' | 'chat' | 'settings'>('onboarding')
  const [quickAskVisible, setQuickAskVisible] = useState(false)
  const [step, setStep] = useState<Step>(Step.WELCOME)
  const [commandCopied, setCommandCopied] = useState<boolean>(false)
  const command = 'ollama run llama3.2'

  // Decide which view to show on load
  useEffect(() => {
    const sp = new URLSearchParams(window.location.search)
    const qv = sp.get('view')
    if (qv === 'quickask') {
      // Quick ask is no longer rendered inside main window; ignore.
    }
    if (qv === 'settings') {
      setView('settings')
      return
    }
    if (qv === 'chat') {
      setView('chat')
      return
    }
    // No explicit query param, decide based on first-time-run
    setView(store.get('first-time-run') ? 'chat' : 'onboarding')
  }, [])

  // Overlay listeners
  useEffect(() => {
    function onToggle(_e: any, force?: boolean) { setQuickAskVisible(v => (typeof force === 'boolean' ? force : !v)) }
    function onShow() { setQuickAskVisible(true) }
    function onHide() { setQuickAskVisible(false) }
    ipcRenderer.on('quick-ask:toggle', onToggle)
    ipcRenderer.on('quick-ask:show', onShow)
    ipcRenderer.on('quick-ask:hide', onHide)
    return () => {
      ipcRenderer.off('quick-ask:toggle', onToggle)
      ipcRenderer.off('quick-ask:show', onShow)
      ipcRenderer.off('quick-ask:hide', onHide)
    }
  }, [])

  // Escape to close overlay
  useEffect(() => {
    function esc(e: KeyboardEvent) { if (e.key === 'Escape' && quickAskVisible) setQuickAskVisible(false) }
    window.addEventListener('keydown', esc)
    return () => window.removeEventListener('keydown', esc)
  }, [quickAskVisible])

  const overlay = quickAskVisible && !standaloneQuick ? (
    <div className='fixed inset-0 z-50 flex items-start justify-center pt-24'>
      {/* Scrim */}
      <button
        className='absolute inset-0 backdrop-blur-sm bg-[rgba(0,0,0,0.32)] opacity-0 animate-[qa-scrim-fade_140ms_ease-out_forwards] cursor-default'
        aria-label='Close quick ask'
        onClick={()=>setQuickAskVisible(false)}
        onKeyDown={(e)=>{ if (e.key === 'Escape') { e.preventDefault(); setQuickAskVisible(false) } }}
        style={{ border: 'none', padding:0, background:'rgba(0,0,0,0.25)' }}
      />
      {/* Panel container */}
      <div
        className='relative pointer-events-auto w-[620px] max-w-[92%]'
        style={{ animation: 'qa-fade-in 120ms ease-out' }}
      >
        <QuickAskPanel onClose={() => setQuickAskVisible(false)} standalone={new URLSearchParams(window.location.search).get('standalone') === '1'} />
      </div>
      <style>{`@keyframes qa-fade-in { from { opacity:0; transform: translateY(6px) scale(.985); } to { opacity:1; transform: translateY(0) scale(1); } }
@keyframes qa-scrim-fade { to { opacity:1; } }`}</style>
    </div>
  ) : null

  let mainContent: JSX.Element
  if (view === 'chat') {
    mainContent = <ChatView />
  } else if (view === 'settings') {
    mainContent = <SettingsPanel />
  } else {
    // Onboarding flow
    mainContent = (
      <div className='drag'>
        <div className='mx-auto flex min-h-screen w-full flex-col justify-between bg-white px-4 pt-16 relative'>
          {step === Step.WELCOME && (
            <>
              <div className='mx-auto text-center'>
                <h1 className='mb-6 mt-4 text-2xl tracking-tight text-gray-900'>Welcome to Ollama</h1>
                <p className='mx-auto w-[65%] text-sm text-gray-400'>
                  Let's get you up and running with your own large language models.
                </p>
                <button
                  onClick={() => setStep(Step.CLI)}
                  className='no-drag rounded-dm mx-auto my-8 w-[40%] rounded-md bg-black px-4 py-2 text-sm text-white hover:brightness-110'
                >
                  Next
                </button>
              </div>
              <div className='mx-auto'>
                <OllamaIcon />
              </div>
            </>
          )}
          {step === Step.CLI && (
            <div className='mx-auto flex flex-col space-y-28 text-center'>
              <h1 className='mt-4 text-2xl tracking-tight text-gray-900'>Install the command line</h1>
              <pre className='mx-auto text-4xl text-gray-400'>&gt; ollama</pre>
              <div className='mx-auto'>
                <button
                  onClick={async () => {
                    try {
                      await install()
                      setStep(Step.FINISH)
                    } catch (e) {
                      console.error('could not install: ', e)
                    } finally {
                      getCurrentWindow().show()
                      getCurrentWindow().focus()
                    }
                  }}
                  className='no-drag rounded-dm mx-auto w-[60%] rounded-md bg-black px-4 py-2 text-sm text-white hover:brightness-110'
                >
                  Install
                </button>
                <p className='mx-auto my-4 w-[70%] text-xs text-gray-400'>
                  You will be prompted for administrator access
                </p>
              </div>
            </div>
          )}
          {step === Step.FINISH && (
            <div className='mx-auto flex flex-col space-y-20 text-center'>
              <h1 className='mt-4 text-2xl tracking-tight text-gray-900'>Run your first model</h1>
              <div className='flex flex-col'>
                <div className='group relative flex items-center'>
                  <pre className='language-none text-2xs w-full rounded-md bg-gray-100 px-4 py-3 text-start leading-normal'>
                    {command}
                  </pre>
                  <button
                    className={`no-drag absolute right-[5px] px-2 py-2 ${
                      commandCopied
                        ? 'text-gray-900 opacity-100 hover:cursor-auto'
                        : 'text-gray-200 opacity-50 hover:cursor-pointer'
                    } hover:font-bold hover:text-gray-900 group-hover:opacity-100`}
                    onClick={() => {
                      copy(command)
                      setCommandCopied(true)
                      setTimeout(() => setCommandCopied(false), 3000)
                    }}
                  >
                    {commandCopied ? (
                      <CheckIcon className='h-4 w-4 font-bold text-gray-500' />
                    ) : (
                      <DocumentDuplicateIcon className='h-4 w-4 text-gray-500' />
                    )}
                  </button>
                </div>
                <p className='mx-auto my-4 w-[70%] text-xs text-gray-400'>
                  Run this command in your favorite terminal.
                </p>
              </div>
              <button
                onClick={() => {
                  store.set('first-time-run', true)
                  setView('chat')
                }}
                className='no-drag rounded-dm mx-auto w-[60%] rounded-md bg-black px-4 py-2 text-sm text-white hover:brightness-110'
              >
                Finish
              </button>
            </div>
          )}
        </div>
      </div>
    )
  }

  // Standalone quick ask now handled by separate BrowserWindow (`quickask_window` entry). No early return path.
  return <>
      {mainContent}
      {overlay}
    </>
}

// SettingsPanel now lives in ./settings/SettingsPanel.tsx
