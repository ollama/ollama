import React, { useState, useRef, useEffect } from 'react'
import './scrollbar.css'
import './titlebar.css'
import { v4 as uuidv4 } from 'uuid'
import { ChatMessage, ChatProvider, ToolCall, ToolResult, parseMockToolCalls, isToolCallToken } from '../providers/ChatProvider'
import { normalizeModelId, isAlias } from '../providers/modelNormalize'
import { LocalProvider } from '../providers/LocalProvider'
import { DummyTurboProvider } from '../providers/DummyTurboProvider'
import { TurboProvider } from '../providers/TurboProvider'
import Store from 'electron-store'
import { loadSessions, upsertSession, ChatSession, deleteSession as removeSession } from './persistence'
import { SessionManager } from './SessionManager'
import { Markdown } from './Markdown'

// Simple Turbo config panel for base URL & token persistence
const turboStore = new Store<{ turboToken?: string, turboBase?: string, turboEnabled?: boolean }>({ name: 'turbo-config' })

const TurboConfigPanel: React.FC = () => {
  const [token, setToken] = useState(turboStore.get('turboToken') || '')
  const [base, setBase] = useState(turboStore.get('turboBase') || '')
  function save() {
    turboStore.set('turboToken', token.trim() || undefined)
    turboStore.set('turboBase', base.trim() || undefined)
  }
  return (
    <div className='border rounded p-2 mb-2 bg-gray-50 space-y-1'>
      <div className='text-xs font-semibold'>Turbo Config</div>
      <input value={base} onChange={e=>setBase(e.target.value)} placeholder='Base URL (https://...)' className='w-full border rounded px-2 py-1 text-xs' />
      <input value={token} onChange={e=>setToken(e.target.value)} placeholder='API Token' className='w-full border rounded px-2 py-1 text-xs' />
      <button onClick={save} className='text-xs bg-black text-white px-2 py-1 rounded'>Save</button>
    </div>
  )
}

const MAX_LOOPS = 4

interface ToolCallState extends ToolCall { status: 'pending' | 'done'; output?: string }

// Preset model identifiers (friendly aliases) - can be mapped to backend IDs if needed
const PRESET_MODELS = [
  'gpt-oss',
  'gpt-oss:20b',
  'gpt-oss:120b',
  'gemma3n',
  'gemma3',
  'gemma3:1b',
  'gemma3:4b',
  'gemma3:12b',
  'gemma3:27b',
  'deepset-r1',
]

interface PresetMeta { ui: string; backend: string; alias: boolean; valid: boolean | null }

interface LocalModelInfo { id: string }

// Provider selection is purely global (Turbo toggle), not model-name based
function providerForModelGlobal(turboEnabled: boolean): 'local' | 'turbo' {
  return turboEnabled ? 'turbo' : 'local'
}

export const ChatView: React.FC = () => {
  const [selectedModel, setSelectedModel] = useState<string>(PRESET_MODELS[1])
  const [localModels, setLocalModels] = useState<LocalModelInfo[]>([])
  const [modelsLoaded, setModelsLoaded] = useState(false)
  const [modelError, setModelError] = useState<string | null>(null)
  const [turboEnabled, setTurboEnabled] = useState<boolean>(!!turboStore.get('turboEnabled'))
  const [turboNotice, setTurboNotice] = useState<string | null>(null)
  const [supportsThinking, setSupportsThinking] = useState<boolean | null>(null)
  const [capLoading, setCapLoading] = useState(false)
  const [capCache, setCapCache] = useState<Record<string,{thinking:boolean}>>({})
  const [reasoningEnabled, setReasoningEnabled] = useState(true)
  const [downgradeInfo, setDowngradeInfo] = useState<string | null>(null)
  const initialProvider = providerForModelGlobal(turboEnabled)
  const [providerName, setProviderName] = useState<'local' | 'turbo' | 'turbo-dummy'>(initialProvider)
  const providerRef = useRef<ChatProvider>(initialProvider === 'local' ? new LocalProvider() : new TurboProvider())
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [thinking, setThinking] = useState('')
  const [toolCalls, setToolCalls] = useState<ToolCallState[]>([])
  const [loopCount, setLoopCount] = useState(0)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [session, setSession] = useState<ChatSession | null>(null)
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [presetMeta, setPresetMeta] = useState<PresetMeta[]>([])
  const [missingModel, setMissingModel] = useState<string | null>(null)
  const [pulling, setPulling] = useState(false)
  const [pullProgress, setPullProgress] = useState<{status?:string; digest?:string; total?:number; completed?:number; percent?:number}>({})
  const pullAbortRef = useRef<AbortController | null>(null)
  const lastUserMessageRef = useRef<ChatMessage | null>(null)
  const userInitiatedPullRef = useRef(false)
  const lastPullTargetRef = useRef<string | null>(null)
  // Track per-layer (digest) progress to compute an overall percent correctly
  const pullStatsRef = useRef<{ layers: Record<string,{ total?: number; completed?: number }> }>({ layers: {} })
  const [pullStatsRefCurrent] = [pullStatsRef] // placeholder to keep lint quiet if unused temporarily
  const [sidebarCollapsed, setSidebarCollapsed] = useState<boolean>(() => {
    try { return !!(window as any).localStorage?.getItem('sidebar-collapsed') } catch { return false }
  })
  const scrollContainerRef = useRef<HTMLDivElement | null>(null)
  const [hasOverflowTop, setHasOverflowTop] = useState(false)
  const [hasOverflowBottom, setHasOverflowBottom] = useState(false)
  useEffect(() => {
    try {
      if (sidebarCollapsed) (window as any).localStorage?.setItem('sidebar-collapsed','1')
      else (window as any).localStorage?.removeItem('sidebar-collapsed')
    } catch {}
  }, [sidebarCollapsed])

  function handleNewSession() {
    const newSession: ChatSession = {
      id: uuidv4(),
      title: 'New Chat',
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      messages: [],
    }
    upsertSession(newSession)
    setSessions(prev => [newSession, ...prev])
    setSession(newSession)
    setMessages([])
    setError(null)
    setThinking('')
    setToolCalls([])
  }

  function handleSelectSession(session: ChatSession) {
    setSession(session)
    setMessages(session.messages)
    setError(null)
    setThinking('')
    setToolCalls([])
  }

  function handleDeleteSession(sessionToDelete: ChatSession) {
    removeSession(sessionToDelete.id)
    const remainingSessions = sessions.filter(s => s.id !== sessionToDelete.id)
    setSessions(remainingSessions)
    if (session?.id === sessionToDelete.id) {
      if (remainingSessions.length > 0) {
        handleSelectSession(remainingSessions[0])
      } else {
        handleNewSession()
      }
    }
  }

  function switchProvider(name: 'local' | 'turbo' | 'turbo-dummy') {
    if (providerName === name) return
    setProviderName(name)
    providerRef.current.cancelCurrent?.()
    providerRef.current = name === 'local' ? new LocalProvider() : (name === 'turbo-dummy' ? new DummyTurboProvider() : new TurboProvider())
  }

  // React when turboEnabled changes: persist & adjust provider if disabling
  useEffect(() => {
    if (turboEnabled) {
      turboStore.set('turboEnabled', true)
    } else {
      try { (turboStore as any).delete('turboEnabled') } catch { /* older versions require delete() */ }
    }
  const desired = providerForModelGlobal(turboEnabled)
  if (desired !== providerName) switchProvider(desired)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [turboEnabled])

  async function fetchCapabilities(model: string) {
    const norm = normalizeModelId(model) || model
    setCapLoading(true)
    try {
      let url = 'http://localhost:11434/api/show'
      const headers: Record<string,string> = { 'Content-Type': 'application/json' }
      if (providerName.startsWith('turbo')) {
        const base = (turboStore.get('turboBase') || '').replace(/\/$/, '')
        if (!base) throw new Error('Turbo base URL not configured')
        url = base + '/api/show'
        const tok = turboStore.get('turboToken')
        if (tok) headers['Authorization'] = `Bearer ${tok}`
      }
      const res = await fetch(url, { method: 'POST', headers, body: JSON.stringify({ model: norm }) })
      if (res.ok) {
        const json = await res.json()
        const caps: string[] = json.capabilities || []
        const thinking = caps.includes('thinking')
        setCapCache(prev => ({ ...prev, [norm]: { thinking } }))
        setSupportsThinking(thinking)
        setReasoningEnabled(thinking) // enable only if supported initially
      } else {
        if (res.status === 404 && providerName.startsWith('turbo')) {
          // mark preset invalid
          setPresetMeta(prev => prev.map(p => p.ui === model ? { ...p, valid: false } : p))
        }
        setSupportsThinking(false)
        setReasoningEnabled(false)
      }
    } catch (e) {
      if (providerName.startsWith('turbo')) {
        // likely network or model missing; mark invalid optimisticly if alias
        setPresetMeta(prev => prev.map(p => p.ui === model ? { ...p, valid: false } : p))
      }
      setSupportsThinking(false)
      setReasoningEnabled(false)
    } finally {
      setCapLoading(false)
    }
  }

  function onSelectModel(m: string) {
    setSelectedModel(m)
    const desired = providerForModelGlobal(turboEnabled)
    if (desired !== providerName) switchProvider(desired)
    // Capability handling
    const norm = m.replace(/:latest$/,'')
    if (capCache[norm]) {
      setSupportsThinking(capCache[norm].thinking)
      setReasoningEnabled(capCache[norm].thinking)
    } else {
      setSupportsThinking(null) // unknown; fetch proactively
      setReasoningEnabled(false) // disable until known to avoid unsupported errors
      fetchCapabilities(m)
    }
  }

  useEffect(() => {
    // Build preset meta (initial validity unknown)
    const meta: PresetMeta[] = PRESET_MODELS.map(ui => {
      const backend = normalizeModelId(ui) || ui
      const alias = backend !== ui
      const valid: boolean | null = null
      return { ui, backend, alias, valid }
    })
    setPresetMeta(meta)
  }, [])

  // Watch for error text changes to surface missing model prompt (in case throw path not caught in send try/catch)
  useEffect(() => {
    if (!error) return
    if (missingModel) return // already set
    const miss = error.match(/model\s+"?([\w:\-\.]+)"?\s+not\s+found/i)
    if (miss) setMissingModel(miss[1])
  }, [error, missingModel])

  // Load or create initial session
  useEffect(() => {
    const existing = loadSessions()
    setSessions(existing)
    if (existing.length) {
      setSession(existing[0])
      setMessages(existing[0].messages)
    } else {
      const newSession: ChatSession = { id: uuidv4(), title: 'New Chat', createdAt: new Date().toISOString(), updatedAt: new Date().toISOString(), messages: [] }
      setSession(newSession)
      upsertSession(newSession)
    }
  }, [])

  // Fetch locally installed models (Ollama / OpenAI-compatible list)
  useEffect(() => {
    let cancelled = false
    async function fetchModels() {
      try {
        setModelError(null)
        const res = await fetch('http://localhost:11434/v1/models')
        if (!res.ok) throw new Error(`models fetch failed: ${res.status}`)
        const json = await res.json()
        const data = Array.isArray(json.data) ? json.data : []
        const list: LocalModelInfo[] = data.map((m: any) => ({ id: m.id }))
        if (!cancelled) { setLocalModels(list); setModelsLoaded(true) }
      } catch (e:any) {
        if (!cancelled) { setModelError(e.message || String(e)); setModelsLoaded(true) }
      }
    }
    fetchModels()
    const interval = setInterval(fetchModels, 15000) // refresh every 15s
    return () => { cancelled = true; clearInterval(interval) }
  }, [])

  async function runTurn(allMessages: ChatMessage[], loopsRemaining: number) {
    setLoopCount(MAX_LOOPS - loopsRemaining)
    const provider = providerRef.current
  const normModel = normalizeModelId(selectedModel) || selectedModel
  const allowThinking = (supportsThinking === true) && reasoningEnabled
  await provider.startChatTurn({ messages: allMessages, maxLoopsRemaining: loopsRemaining, model: normModel, think: allowThinking }, {
      onThinking: d => setThinking(prev => prev + d),
      onText: d => {
        setMessages(prev => {
          const last = prev[prev.length - 1]
          if (last && last.role === 'assistant') {
            return [...prev.slice(0, -1), { ...last, content: last.content + d }]
          }
          return [...prev, { role: 'assistant', content: d }]
        })
      },
      onToolCall: call => setToolCalls(prev => [...prev, { ...call, status: 'pending' }]),
      onError: err => console.error(err),
    })
  }

  async function executePendingTools(): Promise<ToolResult[]> {
    const pending = toolCalls.filter(t => t.status === 'pending')
    if (!pending.length) return []
    const results = await providerRef.current.executeTools?.(pending) || []
    setToolCalls(prev => prev.map(tc => {
      const r = results.find(r => r.id === tc.id)
      return r ? { ...tc, status: 'done', output: r.output } : tc
    }))
    return results
  }

  async function agentLoop(baseMessages: ChatMessage[]) {
    let loops = MAX_LOOPS
    let history = [...baseMessages]
    while (loops > 0) {
      await runTurn(history, loops)
      const newToolCalls = toolCalls.filter(tc => tc.status === 'pending')
      if (!newToolCalls.length) break
      const results = await executePendingTools()
  history = [...history, ...newToolCalls.map(tc => ({ role: 'tool' as const, content: results.find(r => r.id === tc.id)?.output || '', tool_name: tc.name }))]
      loops--
    }
  }

  async function onSend() {
    if (!input.trim() || running) return
  // No per-model turbo gating
    setRunning(true)
    setThinking('')
    setError(null)
    setToolCalls([])
  const userMsg: ChatMessage = { role: 'user', content: input }
  lastUserMessageRef.current = userMsg
    setMessages(prev => [...prev, userMsg])
    if (session) {
      const updated: ChatSession = { ...session, messages: [...messages, userMsg], updatedAt: new Date().toISOString(), title: session.title === 'New Chat' && input.trim() ? input.slice(0, 40) : session.title }
      setSession(updated)
      upsertSession(updated)
      setSessions(prev => {
        const idx = prev.findIndex(s => s.id === updated.id)
        if (idx >= 0) { const copy = [...prev]; copy[idx] = updated; return copy } else { return [updated, ...prev] }
      })
    }

    // Detect mock tool call tokens before sending to provider (for dummy provider demonstration)
    if (isToolCallToken(input)) {
      const mocks = parseMockToolCalls(input)
      setToolCalls(mocks.map(m => ({ ...m, status: 'pending' })))
    }

    // If capabilities unknown and not already loading, we disable reasoning for this first send to avoid unsupported error
    if (supportsThinking === null && !capLoading) {
      // Fire off background fetch if somehow not started (e.g., first initial model on load)
      const norm = normalizeModelId(selectedModel) || selectedModel
      if (!capCache[norm]) fetchCapabilities(selectedModel)
    }

    try {
      await agentLoop([...messages, userMsg])
    } catch (e:any) {
      const errStr = String(e)
      setError(errStr)
      // Detect missing local model pattern (support ids containing ':' or '-')
      const miss = errStr.match(/model\s+"?([\w:\-\.]+)"?\s+not\s+found/i)
      if (miss) {
        setMissingModel(miss[1] || selectedModel)
      }
      const unsupported = /does not support thinking/i.test(errStr)
      if (unsupported) {
        setSupportsThinking(false)
        setReasoningEnabled(false)
        setThinking('')
        setError(null)
        setDowngradeInfo('Disabled reasoning: model lacks thinking capability.')
        setTimeout(() => setDowngradeInfo(null), 5000)
        try { await agentLoop([...messages, userMsg]) } catch (e2:any) { setError(e2.message || String(e2)) }
      }
    } finally {
      setRunning(false)
      setInput('')
      if (session) {
        const updated: ChatSession = { ...session, messages: [...messages, userMsg, ...([] as ChatMessage[])], updatedAt: new Date().toISOString() }
        setSession(updated)
        upsertSession(updated)
      }
    }
  }

  function resolveModelForPull(name: string): string {
    // Prefer normalized id; if normalization does nothing and name lacks ':' but selectedModel has colon, use selectedModel's normalized form
    const normSelected = normalizeModelId(selectedModel) || selectedModel
    const normName = normalizeModelId(name) || name
    if (normName === name && normName.indexOf(':') === -1 && normSelected.indexOf(':') !== -1) {
      return normSelected
    }
    return normName
  }

  async function pullModel(name: string) {
    if (pulling) return
    userInitiatedPullRef.current = true
    setPulling(true)
    setPullProgress({ status: 'starting' })
    setError(null)
    pullAbortRef.current = new AbortController()
    const finalName = resolveModelForPull(name)
    lastPullTargetRef.current = finalName
    // Insert a system message for inline progress display
  setMessages(prev => [...prev, { role: 'system', content: `Pulling model ${finalName}…` }])
    console.log('[pull] start', { requested: name, finalName })
    try {
      const res = await fetch('http://localhost:11434/api/pull', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: finalName }), signal: pullAbortRef.current.signal })
      if (!res.body) throw new Error('No pull stream')
      const reader = res.body.getReader()
      const decoder = new TextDecoder('utf-8')
      let buffered = ''
      let succeeded = false
      if (!userInitiatedPullRef.current) {
        throw new Error('Pull not user initiated; aborted')
      }
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffered += decoder.decode(value, { stream: true })
        let idx
        while ((idx = buffered.indexOf('\n')) >= 0) {
          const line = buffered.slice(0, idx).trim()
          buffered = buffered.slice(idx + 1)
          if (!line) continue
          try {
            const evt = JSON.parse(line)
            // Per-layer aggregation
            if (evt.digest) {
              const layer = pullStatsRef.current.layers[evt.digest] || (pullStatsRef.current.layers[evt.digest] = {})
              if (typeof evt.total === 'number') layer.total = evt.total
              if (typeof evt.completed === 'number') layer.completed = evt.completed
            }
            // Aggregate overall progress
            let overallTotal = 0
            let overallCompleted = 0
            for (const dg of Object.keys(pullStatsRef.current.layers)) {
              const l = pullStatsRef.current.layers[dg]
              const c = l.completed || 0
              const t = l.total ?? c // fallback to completed until total known
              overallCompleted += c
              overallTotal += t
            }
            let percent = overallTotal > 0 ? overallCompleted / overallTotal : 0
            if (percent > 1) percent = 1
            const statusNow = evt.status || pullProgress.status || '…'
            // Update global pull progress state
            setPullProgress(prev => ({ ...prev, status: statusNow, digest: evt.digest || prev.digest, total: overallTotal, completed: overallCompleted, percent }))
            if (evt.status === 'success') {
              percent = 1
              succeeded = true
              setPullProgress(prev => ({ ...prev, percent: 1, status: 'success' }))
            }
            // Update inline message using local variables to avoid stale closures
            setMessages(prev => {
              const copy = [...prev]
              for (let i = copy.length - 1; i >= 0; i--) {
                const m = copy[i]
                if (m.role === 'system' && m.content.startsWith('Pulling model ' + finalName)) {
                  const pct = Math.round(percent * 100)
                  const mb = (n:number)=> (n/1024/1024).toFixed(1)
                  const completedMB = mb(overallCompleted)
                  const totalMB = overallTotal > 0 ? mb(overallTotal) : undefined
                  let sizeStr = ''
                  if (overallTotal > 0) sizeStr = ` ${completedMB}MB/${totalMB}MB`
                  else if (overallCompleted > 0) sizeStr = ` ${completedMB}MB`
                  copy[i] = { ...m, content: succeeded ? `Model ${finalName} pulled successfully.` : `Pulling model ${finalName}… ${pct}%${sizeStr} (${statusNow})` }
                  break
                }
              }
              return copy
            })
          } catch {}
        }
      }
      // Completed; refresh model list on success
      if (succeeded) {
        console.log('[pull] success', finalName)
        setMissingModel(null)
        // Trigger refresh immediately
        try {
          const res2 = await fetch('http://localhost:11434/v1/models')
          if (res2.ok) {
            const json = await res2.json(); const data = Array.isArray(json.data) ? json.data : []
            setLocalModels(data.map((m: any) => ({ id: m.id })))
          }
        } catch {}
        // Retry last user message automatically
        if (lastUserMessageRef.current) {
          setTimeout(() => {
            setMessages(prev => [...prev, lastUserMessageRef.current!])
            agentLoop([...messages, lastUserMessageRef.current!]).catch(err => setError(String(err)))
          }, 300)
        }
      }
    } catch (e:any) {
      if (e.name === 'AbortError') {
        setError('Pull canceled')
      } else {
        const msg = e.message || String(e)
        // Network / server exit cases
        if (/network error/i.test(msg) || /failed to fetch/i.test(msg)) {
          setError('Connection lost during pull. Ensure the server is running and retry.')
        } else {
          setError(msg)
        }
        console.log('[pull] error', e)
      }
  // Mark inline message canceled/failed
  setMessages(prev => prev.map(m => (m.role==='system' && m.content.startsWith('Pulling model '+finalName)) ? { ...m, content: `Model pull for ${finalName} failed: ${error || 'canceled'}` } : m))
    } finally {
  pullAbortRef.current = null
  setPulling(false)
  userInitiatedPullRef.current = false
  pullStatsRef.current.layers = {}
    }
  }

  // Apply app-wide body styling & lock background scroll
  useEffect(() => {
    const prevBg = document.body.style.backgroundColor
    const prevOv = document.body.style.overflow
    document.body.style.backgroundColor = '#0e0e0e'
    document.documentElement.style.backgroundColor = '#0e0e0e'
    document.body.style.overflow = 'hidden'
    return () => {
      document.body.style.backgroundColor = prevBg
      document.documentElement.style.backgroundColor = ''
      document.body.style.overflow = prevOv
    }
  }, [])

  function handleRenameSession(id: string, title: string) {
    setSessions(prev => {
      return prev.map(s => s.id === id ? { ...s, title, updatedAt: new Date().toISOString() } : s)
    })
    if (session?.id === id) {
      setSession(s => s ? { ...s, title } : s)
    }
    const updated = loadSessions().map(s => s.id === id ? { ...s, title, updatedAt: new Date().toISOString() } : s)
    // persist
    try { (require('./persistence') as any).saveSessions(updated) } catch {}
  }

  // Auto-scroll when new messages, thinking updates, or pull progress append content
  useEffect(() => {
    const el = scrollContainerRef.current
    if (!el) return
    // If user is near bottom (within 120px), auto-scroll; else respect manual scroll position
    const threshold = 120
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < threshold
    if (atBottom) {
      el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
    }
    // Update overflow indicators after DOM changes
    requestAnimationFrame(() => {
      if (!el) return
      setHasOverflowTop(el.scrollTop > 4)
      setHasOverflowBottom(el.scrollHeight - el.scrollTop - el.clientHeight > 4)
    })
  }, [messages, thinking, pulling, toolCalls])

  // Listen to manual scroll to update gradient visibility
  useEffect(() => {
    const el = scrollContainerRef.current
    if (!el) return
    function onScroll() {
      setHasOverflowTop(el.scrollTop > 4)
      setHasOverflowBottom(el.scrollHeight - el.scrollTop - el.clientHeight > 4)
    }
    onScroll()
    el.addEventListener('scroll', onScroll, { passive: true })
    const ro = new ResizeObserver(() => onScroll())
    ro.observe(el)
    return () => { el.removeEventListener('scroll', onScroll); ro.disconnect() }
  }, [])

  const isMac = /Mac|Darwin/.test(navigator.platform)
  return (
  <div className={`relative flex flex-row h-screen w-full overflow-hidden bg-[#0e0e0e] text-gray-200 ${isMac ? 'pt-[32px]' : ''}`}> {/* seamless canvas */}
      {isMac && <div className='drag-region-fixed' />}
      <SessionManager
        sessions={sessions}
        currentSession={session}
        onNewSession={handleNewSession}
        onSelectSession={handleSelectSession}
        onDeleteSession={handleDeleteSession}
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(c => !c)}
        onRename={handleRenameSession}
      />
      <div className='flex-1 flex flex-col h-full min-h-0'>
  {/* drag region now handled by fixed element */}
        {/* Scrollable conversation area */}
  <div ref={scrollContainerRef} className='relative flex-1 min-h-0 overflow-y-auto no-scrollbar px-6 py-8 flex flex-col items-center'>
          {/* Top gradient overlay */}
          <div className={`pointer-events-none absolute left-0 top-0 w-full h-6 bg-gradient-to-b from-[#0e0e0e] to-transparent transition-opacity duration-200 ${hasOverflowTop ? 'opacity-100' : 'opacity-0'}`}/>
          {/* Bottom gradient overlay */}
          <div className={`pointer-events-none absolute left-0 bottom-0 w-full h-8 bg-gradient-to-t from-[#0e0e0e] to-transparent transition-opacity duration-200 ${hasOverflowBottom ? 'opacity-100' : 'opacity-0'}`}/>
          <div className='w-full max-w-2xl space-y-5 pb-6'>
              {(missingModel || pulling) && (
                <div className='sticky top-2 z-10 text-xs bg-blue-900/60 backdrop-blur border border-blue-600 rounded p-2 flex flex-col gap-2 shadow'>
                  {!pulling && missingModel && (
                  <>
                    <div>Model <span className='font-mono'>{missingModel}</span> not found locally. Pull now?</div>
                    <div className='flex gap-2'>
                      <button onClick={()=>pullModel(missingModel)} className='px-2 py-1 text-xs bg-blue-600 hover:bg-blue-500 rounded text-white'>Pull</button>
                      <button onClick={()=>setMissingModel(null)} className='px-2 py-1 text-xs bg-gray-600 hover:bg-gray-500 rounded text-white'>Dismiss</button>
                    </div>
                  </>
                )}
                {pulling && (
                  <>
                    <div className='flex justify-between items-center'>
                      <span>Pulling <span className='font-mono'>{missingModel}</span></span>
                      <span>{Math.round((pullProgress.percent||0)*100)}%</span>
                    </div>
                    <div className='w-full h-2 bg-blue-950 rounded overflow-hidden'>
                      <div className='h-full bg-blue-500 transition-all duration-200' style={{ width: `${(pullProgress.percent||0)*100}%` }} />
                    </div>
                    <div className='mt-1 flex justify-between items-center'>
                      <span className='opacity-70 font-mono truncate'>{pullProgress.status || '...'}</span>
                      <button onClick={()=>{ pullAbortRef.current?.abort(); }} className='px-2 py-[2px] text-[10px] bg-red-600 hover:bg-red-500 rounded text-white'>Cancel</button>
                    </div>
                  </>
                )}
              </div>
            )}
          {modelError && localModels.length===0 && (
            <div className='text-xs bg-amber-900/20 border border-amber-800/40 text-amber-300 rounded p-2 backdrop-blur-sm'>
              Backend unreachable: {modelError}. The local server may not be running (binary missing or skipped). You can still explore the UI; local models will appear once the server is available.
            </div>
          )}
          {messages.map((m,i) => {
            const roleLabel = (
              <span className={`block text-[11px] uppercase tracking-wide mb-1 font-medium ${m.role==='user' ? 'text-gray-400' : (m.role==='assistant' ? 'text-emerald-400' : 'text-gray-500')}`}>{m.role}</span>
            )
            return (
              <div key={i} className='text-sm leading-relaxed rounded-lg px-4 py-3 bg-[#121212] border border-[#1b1b1b]'>
                {roleLabel}
                {m.role === 'assistant' || m.role === 'system' ? (
                  <Markdown content={m.content} />
                ) : (
                  <div className='whitespace-pre-wrap text-gray-100'>{m.content}</div>
                )}
              </div>
            )
          })}
          {thinking && (
            <div className='text-xs'>
              <div className='mt-2 bg-[#1d1d1d] rounded p-3 font-mono text-[11px] max-h-56 overflow-auto whitespace-pre-wrap border border-gray-700'>
                {thinking}
              </div>
            </div>
          )}
          {error && (
            <div className='text-xs text-red-400 bg-red-900/25 border border-red-800/50 rounded p-2 whitespace-pre-wrap flex flex-col gap-1'>
              <div className='flex justify-between items-start'>
                <span className='flex-1 pr-2'>{error}</span>
                <button
                  onClick={() => navigator.clipboard.writeText(error)}
                  className='text-[10px] bg-gray-700 hover:bg-gray-600 rounded px-2 py-[2px] font-mono'
                  title='Copy Error'
                >
                  Copy
                </button>
              </div>
              {missingModel && !pulling && (
                <div className='mt-2'>
                  <button onClick={()=>pullModel(missingModel)} className='px-2 py-[2px] text-[10px] bg-blue-600 hover:bg-blue-500 rounded text-white mr-2'>Pull {missingModel}</button>
                  <button onClick={()=>setMissingModel(null)} className='px-2 py-[2px] text-[10px] bg-gray-700 hover:bg-gray-600 rounded text-white'>Dismiss</button>
                </div>
              )}
              {!pulling && /Connection lost during pull/.test(error) && lastPullTargetRef.current && (
                <div className='mt-2'>
                  <button onClick={()=>pullModel(lastPullTargetRef.current!)} className='mt-1 px-2 py-[2px] text-[10px] bg-amber-600 hover:bg-amber-500 rounded text-white'>Retry Pull {lastPullTargetRef.current}</button>
                </div>
              )}
            </div>
          )}
          {modelError && (
            <div className='text-xs text-amber-400 bg-amber-900/15 border border-amber-800/40 rounded p-2 whitespace-pre-wrap'>
              Model list: {modelError}
            </div>
          )}
          {toolCalls.length > 0 && (
            <div className='bg-[#161616] border border-[#262626] rounded p-3 text-xs shadow-sm'>
              <div className='font-semibold mb-1'>Tool Calls</div>
              {toolCalls.map(tc => (
                <div key={tc.id} className='flex flex-wrap items-center gap-1'>
                  <span className='font-mono'>{tc.name}</span>
                  <span className={`px-1 rounded ${tc.status==='pending' ? 'bg-gray-600' : 'bg-emerald-600'}`}>{tc.status}</span>
                  {tc.output && <span className='opacity-70 truncate'>→ {tc.output}</span>}
                </div>
              ))}
            </div>
          )}
          </div>
        </div>
        {/* Fixed input / controls area */}
        <div className='px-6 pb-8 pt-3 flex flex-col items-center bg-[#0e0e0e]'>
          <div className='w-full max-w-2xl space-y-3'>
          <div className='flex items-stretch bg-[#141414] border border-[#262626] rounded-2xl shadow-inner focus-within:border-[#3a3a3a] overflow-hidden transition-colors'>
            <div className='flex items-center'>
              <select
                value={selectedModel}
                onChange={e => onSelectModel(e.target.value)}
                className='bg-transparent text-xs px-3 py-2 outline-none appearance-none cursor-pointer'
              >
                <optgroup label='Downloaded'>
                  {localModels.map(m => (
                    <option key={m.id} value={m.id} className='bg-[#1c1c1c]'>✔ {m.id}</option>
                  ))}
                  {localModels.length === 0 && <option disabled value='__none' className='bg-[#1c1c1c]'>No local models</option>}
                </optgroup>
                <optgroup label='Presets'>
          {presetMeta.filter(pm => !localModels.find(l => l.id === pm.ui)).map(pm => (
                    <option
                      key={pm.ui}
                      value={pm.ui}
                      className='bg-[#1c1c1c]'
                      disabled={pm.valid === false}
                      title={pm.valid === false ? `Model not available with ${providerName} provider` : undefined}
                    >
                      {pm.ui}{pm.alias && ' *'}
                    </option>
                  ))}
                </optgroup>
              </select>
            </div>
            <button
              onClick={() => supportsThinking && setReasoningEnabled(r => !r)}
              disabled={!supportsThinking}
              title={!supportsThinking ? 'Model does not support reasoning' : (reasoningEnabled ? 'Disable reasoning' : 'Enable reasoning')}
              className={`text-xs px-2 border-r border-[#262626] ${supportsThinking ? 'opacity-100 hover:bg-[#1f1f1f]' : 'opacity-40 cursor-not-allowed'}`}
            >R</button>
            <input
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key==='Enter' && !e.shiftKey && (e.preventDefault(), onSend())}
              placeholder='Send a message'
              className='flex-1 bg-transparent px-2 py-2 text-sm outline-none'
            />
            <button
              onClick={onSend}
              disabled={running}
        className='px-4 py-2 text-sm font-semibold disabled:opacity-40 bg-emerald-600 hover:bg-emerald-500 transition rounded-none'
            >
              {running ? '...' : 'Send'}
            </button>
          </div>
      <div className='flex gap-4 mt-1 pl-2 text-[10px] opacity-50 tracking-wide'>
            <div><span className='font-mono'>✔</span> downloaded</div>
            <div>list refreshes every 15s</div>
            <div>* alias mapped to backend</div>
            <div>R toggles reasoning</div>
          </div>
      <div className='mt-2 flex items-center gap-3 text-[10px] opacity-80'>
            <label className='flex items-center gap-1 cursor-pointer select-none'>
              <input
                type='checkbox'
                checked={turboEnabled}
                onChange={e => {
                  const next = e.target.checked
                  setTurboEnabled(next)
                  if (next) {
                    setTurboNotice('Turbo provider enabled (feature coming soon).')
                    setTimeout(() => setTurboNotice(null), 5000)
                  } else {
                    setTurboNotice('Turbo disabled; remote models locked.')
                    setTimeout(() => setTurboNotice(null), 4000)
                  }
                }}
              />
              <span className='uppercase tracking-wide'>Turbo (coming soon)</span>
            </label>
            {/* Removed model-based turbo restriction message */}
          </div>
          {turboNotice && (
            <div className='mt-1 text-[10px] text-purple-300 bg-purple-900/30 border border-purple-700/50 rounded px-2 py-1'>{turboNotice}</div>
          )}
          {downgradeInfo && (
            <div className='mt-1 text-[10px] text-amber-300 bg-amber-900/30 border border-amber-700/50 rounded px-2 py-1'>{downgradeInfo}</div>
          )}
          {turboEnabled && (
            <div className='mt-2'>
              <details className='text-xs opacity-70'>
                <summary className='cursor-pointer select-none'>Turbo config</summary>
                <div className='mt-1'>
                  <TurboConfigPanel />
                </div>
              </details>
            </div>
          )}
          </div>
        </div>
      </div>
    </div>
  )
}
