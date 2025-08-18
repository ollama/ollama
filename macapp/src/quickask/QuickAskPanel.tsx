import React, { useState, useEffect, useRef, useCallback } from 'react'
import Store from 'electron-store'
import { LocalProvider } from '../providers/LocalProvider'
import type { ChatMessage, ChatProvider } from '../providers/ChatProvider'
import { normalizeModelId } from '../providers/modelNormalize'
import { useSettings } from '../settings/SettingsContext'

interface QuickAskPanelProps { onClose?: () => void; standalone?: boolean }
const qaStore = new Store<{ qaModel?: string; qaReasoning?: boolean; history?: string[] }>({ name: 'quick-ask' })

export const QuickAskPanel: React.FC<QuickAskPanelProps> = ({ onClose, standalone }) => {
  const { settings, ready, setValue } = useSettings()
  const [input, setInput] = useState('')
  const [answer, setAnswer] = useState('')
  const [thinking, setThinking] = useState('')
  const [running, setRunning] = useState(false)
  const [model, setModel] = useState(() => qaStore.get('qaModel') || 'gemma3n')
  const [models, setModels] = useState<string[]>([])
  const [error, setError] = useState<string | null>(null)
  const [supportsThinking, setSupportsThinking] = useState<boolean | null>(null)
  const [reasoningEnabled, setReasoningEnabled] = useState<boolean>(() => {
    const saved = qaStore.get('qaReasoning')
    return typeof saved === 'boolean' ? saved : true
  })
  const [downgraded, setDowngraded] = useState<string | null>(null)
  const providerRef = useRef<ChatProvider>(new LocalProvider())
  const thinkingBufferRef = useRef('')
  const answerBufferRef = useRef('')
  const flushTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [history, setHistory] = useState<string[]>(() => qaStore.get('history') || [])
  const [historyIndex, setHistoryIndex] = useState<number | null>(null)
  const [pinned, setPinned] = useState<boolean>(() => settings?.quickAsk.pinnedDefault ?? false)
  const [showThinking, setShowThinking] = useState<boolean>(() => settings?.quickAsk.showThinking ?? true)
  const [autoCloseOnCopy, setAutoCloseOnCopy] = useState<boolean>(() => settings?.quickAsk.autoCloseOnCopy ?? false)
  const [thinkingTokens, setThinkingTokens] = useState(0)
  const [answerTokens, setAnswerTokens] = useState(0)
  const containerRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => { if (historyIndex !== null) containerRef.current?.querySelector('textarea')?.focus() }, [historyIndex])
  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const res = await fetch('http://localhost:11434/v1/models')
        if (!res.ok) throw new Error(String(res.status))
        const j = await res.json()
        const ids = (j.data || []).map((m: any) => m.id)
        if (!cancelled) setModels(ids)
      } catch { /* ignore */ }
    })()
    return () => { cancelled = true }
  }, [])
  useEffect(() => { containerRef.current?.querySelector('textarea')?.focus() }, [])

  async function fetchCapabilities(modelId: string) {
    setSupportsThinking(null)
    try {
      const norm = normalizeModelId(modelId) || modelId
      const res = await fetch('http://localhost:11434/api/show', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model: norm }) })
      if (!res.ok) { setSupportsThinking(false); return }
      const json = await res.json()
      const caps: string[] = json.capabilities || []
      const thinkingCap = caps.includes('thinking')
      setSupportsThinking(thinkingCap)
      if (!thinkingCap) setReasoningEnabled(false)
    } catch { setSupportsThinking(false); setReasoningEnabled(false) }
  }
  useEffect(() => { fetchCapabilities(model); qaStore.set('qaModel', model) }, [model])
  useEffect(() => { qaStore.set('qaReasoning', reasoningEnabled) }, [reasoningEnabled])
  useEffect(() => { qaStore.set('history', history.slice(0, 30)) }, [history])

  const initialSyncedRef = useRef(false)
  useEffect(() => {
    if (!ready || !settings || initialSyncedRef.current) return
    setPinned(settings.quickAsk.pinnedDefault)
    setShowThinking(settings.quickAsk.showThinking)
    setAutoCloseOnCopy(settings.quickAsk.autoCloseOnCopy)
    initialSyncedRef.current = true
  }, [ready, settings])

  const togglePinned = useCallback(() => { setPinned(p => { const nv = !p; if (ready) setValue('quickAsk.pinnedDefault', nv); return nv }) }, [ready, setValue])
  const toggleShowThinking = useCallback(() => { setShowThinking(s => { const nv = !s; if (ready) setValue('quickAsk.showThinking', nv); return nv }) }, [ready, setValue])
  const onChangeAutoClose = useCallback((checked: boolean) => { setAutoCloseOnCopy(() => { if (ready) setValue('quickAsk.autoCloseOnCopy', checked); return checked }) }, [ready, setValue])

  async function ask() {
    if (!input.trim() || running) return
    setRunning(true); setAnswer(''); setThinking(''); setError(null)
    thinkingBufferRef.current = ''
    answerBufferRef.current = ''
    const messages: ChatMessage[] = [{ role: 'user', content: input }]
    const effectiveModel = normalizeModelId(model) || model
    function doFlush() { if (thinkingBufferRef.current) { const d = thinkingBufferRef.current; thinkingBufferRef.current=''; setThinking(p=>p+d) } if (answerBufferRef.current) { const d = answerBufferRef.current; answerBufferRef.current=''; setAnswer(p=>p+d) } }
    function scheduleFlush(immediate?: boolean) { if (immediate) { if (flushTimerRef.current) { clearTimeout(flushTimerRef.current); flushTimerRef.current=null } doFlush(); return } if (!flushTimerRef.current) flushTimerRef.current=setTimeout(()=>{ flushTimerRef.current=null; doFlush() },32) }
    const callbacks = { onThinking:(t:string)=>{ thinkingBufferRef.current+=t; setThinkingTokens(c=>c+t.length); scheduleFlush() }, onText:(t:string)=>{ answerBufferRef.current+=t; setAnswerTokens(c=>c+t.length); scheduleFlush() }, onToolCall:(_:any)=>{}, onError:(e:any)=> setError(String(e)) }
    const run = (think:boolean)=> providerRef.current.startChatTurn({ messages, model: effectiveModel, think }, callbacks)
    const wantThinking = reasoningEnabled && supportsThinking !== false
    try { await run(wantThinking); scheduleFlush(true) } catch (e:any) { const msg=e?.message||String(e); if(/does not support thinking/i.test(msg) && wantThinking){ setDowngraded('Reasoning disabled: model lacks thinking capability.'); setSupportsThinking(false); setReasoningEnabled(false); setThinking(''); try { await run(false); scheduleFlush(true) } catch(e2:any){ setError(e2.message||String(e2)) } } else setError(msg) } finally { scheduleFlush(true); setRunning(false); setHistory(h=>[input.trim(), ...h.filter(q=>q!==input.trim())].slice(0,30)); setThinkingTokens(0); setAnswerTokens(0); if(!pinned) onClose?.() }
  }

  function onKey(e: React.KeyboardEvent) { if (e.key==='Enter' && (e.metaKey||(!e.shiftKey && !e.altKey))) { e.preventDefault(); ask() } if (e.key==='Escape') { onClose?onClose():window.close() } if (e.key==='ArrowUp' && history.length){ e.preventDefault(); setHistoryIndex(i=>{ const next = i===null?0:Math.min(i+1,history.length-1); setInput(history[next]); return next }) } if (e.key==='ArrowDown' && history.length){ e.preventDefault(); setHistoryIndex(i=>{ if(i===null) return null; const next=i-1; if(next<0){ setInput(''); return null } setInput(history[next]); return next }) } }

  const panel = (
    <div ref={containerRef} className='flex flex-col rounded-2xl w-full outline-none bg-[var(--ui-panel)] border border-[var(--ui-border-subtle)] shadow-[0_4px_24px_-6px_rgba(0,0,0,0.6),0_2px_8px_-2px_rgba(0,0,0,0.4)] focus-visible:ring-2 focus-visible:ring-emerald-500/50 text-[var(--ui-text)] text-sm' style={{ backdropFilter:'blur(14px) saturate(150%)', WebkitBackdropFilter:'blur(14px) saturate(150%)' }} tabIndex={-1}>
      <div className='flex items-center gap-2 px-4 pt-3 pb-2 select-none'>
        <div className='text-[10px] uppercase tracking-wider opacity-60 font-semibold'>Quick Ask</div>
        <select aria-label='Model' value={model} onChange={e=> setModel(e.target.value)} className='bg-[var(--ui-panel-alt)] border border-[var(--ui-border-subtle)] rounded-lg px-2 py-[5px] text-[11px] outline-none focus:border-[var(--ui-border-strong)] transition min-w-[140px]'>
          {models.length ? models.map(m => <option key={m} value={m}>{m}</option>) : <option value={model}>{model}</option>}
        </select>
        {(() => { let label: string; if(!supportsThinking) label='Reasoning unsupported'; else if(reasoningEnabled) label='Disable reasoning'; else label='Enable reasoning'; return (
          <button aria-label={label} onClick={()=> supportsThinking && setReasoningEnabled(r => !r)} disabled={!supportsThinking} className={`text-[10px] px-2 h-7 flex items-center justify-center rounded-lg border border-[var(--ui-border-subtle)] ${supportsThinking ? 'hover:bg-[var(--ui-panel-alt)] cursor-pointer' : 'opacity-35 cursor-not-allowed'} ${reasoningEnabled && supportsThinking ? 'bg-emerald-600/80 text-white' : 'bg-[var(--ui-panel-alt)] text-[var(--ui-text-dim)]'} transition select-none`}>{supportsThinking ? 'Think' : 'No‑think'}</button>) })()}
        <button aria-label={pinned ? 'Unpin panel (close after send)' : 'Pin panel (stay open after send)'} className={`ml-1 text-[10px] px-2 h-7 flex items-center justify-center rounded-lg border border-[var(--ui-border-subtle)] ${pinned ? 'bg-[var(--ui-pin)] text-white' : 'bg-[var(--ui-panel-alt)] text-[var(--ui-text-dim)] hover:bg-[#242424]'} transition select-none`} onClick={togglePinned}>{pinned ? 'Pinned' : 'Pin'}</button>
        <div className='ml-auto flex items-center gap-3 text-[10px] opacity-45 tracking-wide'>
          <span>Cmd+Enter=Send</span><span>Esc=Close</span>
        </div>
        <button aria-label='Close quick ask' onClick={()=> onClose?.()} className='ml-2 h-6 w-6 rounded-md flex items-center justify-center text-[var(--ui-text-dim)] hover:text-[var(--ui-text)] hover:bg-white/5 transition'>×</button>
      </div>
      <div className='px-4 pb-4 pt-1'>
        <textarea value={input} onChange={e=>setInput(e.target.value)} onKeyDown={onKey} placeholder='Ask anything...' rows={3} className='w-full resize-none bg-[var(--ui-panel-alt)] box-border border border-[var(--ui-border-faint)] focus:border-[var(--ui-border-subtle)] rounded-xl p-3 outline-none text-sm mb-3 placeholder:text-[var(--ui-text-faint)]' />
        <div className='flex gap-2 mb-2 items-center'>
          <button onClick={ask} disabled={running || !input.trim()} className='px-4 py-[7px] text-[12px] rounded-lg bg-[var(--ui-accent)] disabled:opacity-40 hover:bg-[var(--ui-accent-hover)] active:scale-[.97] transition font-medium shadow-[0_0_0_1px_rgba(0,0,0,0.3),0_4px_10px_-2px_rgba(0,0,0,0.45)]'>{running ? '…' : 'Send'}</button>
          {running && <button onClick={()=>providerRef.current.cancelCurrent?.()} className='px-3 py-[7px] text-[12px] rounded-lg bg-[var(--ui-danger)] hover:bg-[var(--ui-danger-hover)] transition font-medium'>Cancel</button>}
          <button onClick={()=>{ setInput(''); setAnswer(''); setThinking(''); setError(null); setHistoryIndex(null); setThinkingTokens(0); setAnswerTokens(0) }} disabled={running || (!input && !answer && !thinking && !error)} className='px-4 py-[7px] text-[12px] rounded-lg bg-[var(--ui-panel-alt)] hover:bg-[#2d2d2d] disabled:opacity-30 transition'>Clear</button>
          <div className='ml-auto flex gap-3 pr-1 text-[10px] opacity-60'>
            <span title='Thinking tokens'>T:{thinkingTokens}</span><span title='Answer tokens'>A:{answerTokens}</span>{supportsThinking === null && <span>Detecting…</span>}
          </div>
        </div>
        {history.length > 0 && <div className='text-[10px] opacity-35 mb-1'>History (↑/↓): {history.slice(0,3).join(' • ')}</div>}
        {(() => { const norm = normalizeModelId(model) || model; const alias = norm !== model; return <div className='text-[10px] opacity-50 mb-1'>Using model: <span className='font-mono'>{model}</span>{alias && <span className='ml-1 opacity-60'>(→ {norm})</span>}</div> })()}
        {thinking && <div className='mb-2'>
          <div className='flex items-center justify-between mb-1'>
            <span className='uppercase tracking-wide text-[10px] font-semibold text-amber-300'>Thinking</span>
            <div className='flex gap-2'>
              <button onClick={toggleShowThinking} className='text-[10px] px-2 py-[2px] rounded bg-[var(--ui-panel-alt)] hover:bg-[#333] border border-[var(--ui-border-faint)]'>{showThinking ? 'Hide' : 'Show'}</button>
              <button onClick={() => navigator.clipboard.writeText(thinking)} className='text-[10px] px-2 py-[2px] rounded bg-[var(--ui-panel-alt)] hover:bg-[#333] border border-[var(--ui-border-faint)]'>Copy</button>
            </div>
          </div>
          {showThinking && <div className='font-mono text-[11px] whitespace-pre-wrap max-h-28 overflow-auto bg-[rgba(255,255,255,0.04)] border border-amber-700/40 rounded-xl p-2 shadow-inner'>{thinking}</div>}
        </div>}
        {downgraded && <div className='text-[10px] text-amber-300 bg-amber-900/30 border border-amber-800/40 rounded-lg px-2 py-1 mb-2'>{downgraded}</div>}
        {error && <div className='text-[11px] text-red-300 bg-[rgba(120,20,20,0.45)] border border-red-800/40 rounded-lg p-2 mb-2 whitespace-pre-wrap'>{error}</div>}
        <div className='flex-1 overflow-auto bg-[var(--ui-panel-alt)] border border-[var(--ui-border-faint)] rounded-xl p-3 whitespace-pre-wrap text-sm min-h-[60px]' aria-live='polite'>{answer}</div>
        <div className='mt-3 flex items-center gap-3 text-[10px] opacity-55'>
          <label className='flex items-center gap-1 cursor-pointer'><input type='checkbox' checked={autoCloseOnCopy} onChange={e=>onChangeAutoClose(e.target.checked)} /><span>Auto-close after copy (when not pinned)</span></label>
          <button onClick={() => { if (answer) { navigator.clipboard.writeText(answer); if (autoCloseOnCopy && !pinned) onClose?.() } }} disabled={!answer} className='ml-auto text-[10px] px-2 py-[4px] rounded bg-[var(--ui-panel-alt)] hover:bg-[#2d2d2d] disabled:opacity-30 border border-[var(--ui-border-faint)]'>Copy Answer{autoCloseOnCopy && !pinned ? ' & Close' : ''}</button>
        </div>
      </div>
    </div>
  )

  if (standalone !== false) {
    return <div className='w-full h-full flex items-start justify-center p-5 box-border overflow-auto'><div style={{ width:'100%', maxWidth:760, minWidth:320, minHeight:280 }}>{panel}</div></div>
  }
  return panel
}
