import React, { useState, useRef, useEffect } from 'react'
import './scrollbar.css'
import './titlebar.css'
import { v4 as uuidv4 } from 'uuid'
import { ChatMessage, ChatProvider, ToolCall, parseMockToolCalls, isToolCallToken } from '../providers/ChatProvider'
import { useChatStreaming } from './useChatStreaming'
import { normalizeModelId } from '../providers/modelNormalize'
import { LocalProvider } from '../providers/LocalProvider'
import { DummyTurboProvider } from '../providers/DummyTurboProvider'
import { TurboProvider } from '../providers/TurboProvider'
import Store from 'electron-store'
import { useSettings } from '../settings/SettingsContext'
import { loadSessions, upsertSession, ChatSession, deleteSession as removeSession, saveSessions } from './persistence'
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
  // const [modelsLoaded, setModelsLoaded] = useState(false) // unused
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
  // const [loopCount, setLoopCount] = useState(0) // unused after simplification
  const [running, setRunning] = useState(false)
  // Fast send lock to avoid rapid Enter presses before React state flush sets `running`
  const runningRef = useRef(false)
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
  // pullStatsRefCurrent placeholder removed (not needed)
  const [sidebarCollapsed, setSidebarCollapsed] = useState<boolean>(() => {
    try { return !!(window as any).localStorage?.getItem('sidebar-collapsed') } catch { return false }
  })
  const scrollContainerRef = useRef<HTMLDivElement | null>(null)
  const [hasOverflowTop, setHasOverflowTop] = useState(false)
  const [hasOverflowBottom, setHasOverflowBottom] = useState(false)
  // Streaming micro-batching buffers to avoid per-token React re-renders causing UI lockups
  const thinkingBufferRef = useRef('')
  const textBufferRef = useRef('')
  const flushTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pendingThinkingRef = useRef(false) // track whether we have any thinking tokens for labeling
  // Global showThinking setting
  const { settings, setValue } = useSettings()
  const showThinking = settings?.chat.showThinking ?? true
  function toggleShowThinking() { setValue('chat.showThinking', !showThinking) }

  function flushStreamingBuffers(immediate?: boolean) {
    // Optionally run immediately (e.g. after turn completes)
    if (flushTimerRef.current && immediate) {
      clearTimeout(flushTimerRef.current)
      flushTimerRef.current = null
    }
    if (flushTimerRef.current && !immediate) return
    if (!immediate) {
      flushTimerRef.current = setTimeout(() => {
        flushTimerRef.current = null // timer elapsed
        doFlush()
      }, 32) // ~30fps max update cadence
      return
    }
    doFlush()
  }

  function doFlush() {
    const think = thinkingBufferRef.current
    const text = textBufferRef.current
    if (think) {
      thinkingBufferRef.current = ''
      setThinking(prev => prev + think)
    }
    if (text) {
      textBufferRef.current = ''
      setMessages(prev => {
        const last = prev[prev.length - 1]
        if (last && last.role === 'assistant') {
          return [...prev.slice(0, -1), { ...last, content: last.content + text }]
        }
        return [...prev, { role: 'assistant', content: text }]
      })
    }
  }

  // Cleanup timers on unmount
  useEffect(() => () => { if (flushTimerRef.current) { clearTimeout(flushTimerRef.current) /* cleanup pending flush */ } }, [])
    useEffect(() => {
      try {
        if (sidebarCollapsed) (window as any).localStorage?.setItem('sidebar-collapsed','1')
        else (window as any).localStorage?.removeItem('sidebar-collapsed')
      } catch { /* sidebar collapsed state persistence failed */ }
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
    // Backfill ids for legacy messages without one
    const withIds = session.messages.map(m => m.id ? m : { ...m, id: uuidv4() })
    setMessages(withIds)
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
    if (name === 'local') providerRef.current = new LocalProvider()
    else if (name === 'turbo-dummy') providerRef.current = new DummyTurboProvider()
    else providerRef.current = new TurboProvider()
  }

  // React when turboEnabled changes: persist & adjust provider if disabling
  useEffect(() => {
    if (turboEnabled) {
      turboStore.set('turboEnabled', true)
    } else {
  try { (turboStore as any).delete('turboEnabled') } catch { /* legacy electron-store delete failure ignored */ }
    }
    const desired = providerForModelGlobal(turboEnabled)
    if (desired !== providerName) switchProvider(desired)
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
    } catch {
      if (providerName.startsWith('turbo')) {
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
  // Detect missing model (allow :, -, .) – simplified regex (no unnecessary escapes)
  const miss = /model\s+"?([\w:.-]+)"?\s+not\s+found/i.exec(error)
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
        if (!cancelled) setLocalModels(list)
      } catch (e:any) {
        if (!cancelled) setModelError(e.message || String(e))
      }
    }
    fetchModels()
    const interval = setInterval(fetchModels, 15000) // refresh every 15s
    return () => { cancelled = true; clearInterval(interval) }
  }, [])

  // Hook encapsulating streaming/loop logic
  const { agentLoop } = useChatStreaming({
    providerRef,
    maxLoops: MAX_LOOPS,
    selectedModel,
    normalizeModel: (m: string) => normalizeModelId(m) || m,
    allowThinking: () => (supportsThinking === true) && reasoningEnabled,
    getToolCalls: () => toolCalls as any,
    setToolCalls: updater => setToolCalls(updater as any),
  })

  function buildUserMessage(content: string): ChatMessage { return { id: uuidv4(), role: 'user', content } }

  function enqueueUserMessage(userMsg: ChatMessage) {
    setMessages(prev => [...prev, userMsg])
    if (session) {
      const updated: ChatSession = { ...session, messages: [...messages, userMsg], updatedAt: new Date().toISOString(), title: session.title === 'New Chat' && userMsg.content.trim() ? userMsg.content.slice(0, 40) : session.title }
      setSession(updated)
      upsertSession(updated)
      setSessions(prev => {
        const idx = prev.findIndex(s => s.id === updated.id)
        if (idx >= 0) { const copy = [...prev]; copy[idx] = updated; return copy } else { return [updated, ...prev] }
      })
    }
  }

  async function sendMessageFlow(text: string) {
    const userMsg = buildUserMessage(text)
    lastUserMessageRef.current = userMsg
    enqueueUserMessage(userMsg)

    if (isToolCallToken(text)) {
      const mocks = parseMockToolCalls(text)
      setToolCalls(mocks.map(m => ({ ...m, status: 'pending' })))
    } // no else branch
    if (supportsThinking === null && !capLoading) {
      const norm = normalizeModelId(selectedModel) || selectedModel
      if (!capCache[norm]) fetchCapabilities(selectedModel)
    } // else capabilities known
    try {
      await agentLoop([...messages, userMsg], {
        onThinking: (d: string) => { pendingThinkingRef.current = true; thinkingBufferRef.current += d; flushStreamingBuffers() },
        onText: (d: string) => { textBufferRef.current += d; flushStreamingBuffers() },
  onToolCalls: () => { /* tool calls suppressed in main chat for now */ },
        flushBuffers: (immediate?: boolean) => flushStreamingBuffers(!!immediate),
      } as any)
  } catch (e: any) {
      const errStr = String(e)
      setError(errStr)
      const miss = /model\s+"?([\w:.-]+)"?\s+not\s+found/i.exec(errStr)
      if (miss) setMissingModel(miss[1] || selectedModel)
      if (/does not support thinking/i.test(errStr)) {
        setSupportsThinking(false)
        setReasoningEnabled(false)
        setThinking('')
        setError(null)
        setDowngradeInfo('Disabled reasoning: model lacks thinking capability.')
        setTimeout(() => setDowngradeInfo(null), 5000)
        try {
          await agentLoop([...messages, userMsg], {
            onThinking: (d: string) => { pendingThinkingRef.current = true; thinkingBufferRef.current += d; flushStreamingBuffers() },
            onText: (d: string) => { textBufferRef.current += d; flushStreamingBuffers() },
            onToolCalls: () => { /* tool calls suppressed on downgrade retry */ },
            flushBuffers: (imm?: boolean) => flushStreamingBuffers(!!imm),
          } as any)
  } catch (e2: any) { setError(e2.message || String(e2)) }
      }
    }
  }

  async function onSend() {
    if (!input.trim() || runningRef.current) return
    runningRef.current = true
    setRunning(true)
    setThinking(''); setError(null); setToolCalls([]); pendingThinkingRef.current = false
    const currentInput = input
    try { await sendMessageFlow(currentInput) } finally {
      setRunning(false); runningRef.current = false; setInput('')
      if (session) {
        const updated: ChatSession = { ...session, messages: [...messages], updatedAt: new Date().toISOString() }
        setSession(updated); upsertSession(updated)
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

  // Helper for formatting MB
  function formatMB(n: number) { return (n/1024/1024).toFixed(1) }
  function updateInlinePullMessage(finalName: string, succeeded: boolean, percent: number, overallCompleted: number, overallTotal: number, statusNow: string) {
    setMessages(prev => {
      const copy = [...prev]
      for (let i = copy.length - 1; i >= 0; i--) {
        const m = copy[i]
        if (m.role === 'system' && m.content.startsWith('Pulling model ' + finalName)) {
          const pct = Math.round(percent * 100)
          const completedMB = formatMB(overallCompleted)
          const totalMB = overallTotal > 0 ? formatMB(overallTotal) : undefined
          let sizeStr = ''
          if (overallTotal > 0) sizeStr = ` ${completedMB}MB/${totalMB}MB`
          else if (overallCompleted > 0) sizeStr = ` ${completedMB}MB`
          copy[i] = { ...m, content: succeeded ? `Model ${finalName} pulled successfully.` : `Pulling model ${finalName}… ${pct}%${sizeStr} (${statusNow})` }
          break
        }
      }
      return copy
    })
  }
  function aggregateProgress(evt: any) {
    if (evt.digest) {
      const layer = pullStatsRef.current.layers[evt.digest] || (pullStatsRef.current.layers[evt.digest] = {})
      if (typeof evt.total === 'number') layer.total = evt.total
      if (typeof evt.completed === 'number') layer.completed = evt.completed
    }
    let overallTotal = 0
    let overallCompleted = 0
    for (const dg of Object.keys(pullStatsRef.current.layers)) {
      const l = pullStatsRef.current.layers[dg]
      const c = l.completed || 0
      const t = l.total ?? c
      overallCompleted += c
      overallTotal += t
    }
    let percent = overallTotal > 0 ? overallCompleted / overallTotal : 0
    if (percent > 1) percent = 1
    const statusNow = evt.status || pullProgress.status || '…'
    let succeeded = false
    setPullProgress(prev => ({ ...prev, status: statusNow, digest: evt.digest || prev.digest, total: overallTotal, completed: overallCompleted, percent }))
    if (evt.status === 'success') {
      percent = 1
      succeeded = true
      setPullProgress(prev => ({ ...prev, percent: 1, status: 'success' }))
    }
    updateInlinePullMessage(lastPullTargetRef.current || '', succeeded, percent, overallCompleted, overallTotal, statusNow)
    return succeeded
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
    setMessages(prev => [...prev, { id: uuidv4(), role: 'system', content: `Pulling model ${finalName}…` }])
    console.log('[pull] start', { requested: name, finalName })

    try {
      const res = await fetch('http://localhost:11434/api/pull', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: finalName }), signal: pullAbortRef.current.signal })
      if (!res.body) throw new Error('No pull stream')
      const reader = res.body.getReader()
      const decoder = new TextDecoder('utf-8')
      let buffered = ''
      let succeeded = false
      if (!userInitiatedPullRef.current) throw new Error('Pull not user initiated; aborted')
    for (;;) { // deliberate infinite loop reading stream; exits on done break
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
    if (aggregateProgress(evt)) { succeeded = true }
      } catch { /* ignore malformed JSON line */ }
        }
      }
      if (succeeded) {
        console.log('[pull] success', finalName)
        setMissingModel(null)
        try {
          const res2 = await fetch('http://localhost:11434/v1/models')
          if (res2.ok) {
            const json = await res2.json(); const data = Array.isArray(json.data) ? json.data : []
            setLocalModels(data.map((m: any) => ({ id: m.id })))
          }
        } catch { /* ignore refresh error */ }
        if (lastUserMessageRef.current) {
          setTimeout(() => {
            if (lastUserMessageRef.current) {
              setMessages(prev => [...prev, lastUserMessageRef.current])
              agentLoop([...messages, lastUserMessageRef.current], {
                onThinking: (d: string) => { pendingThinkingRef.current = true; thinkingBufferRef.current += d; flushStreamingBuffers() },
                onText: (d: string) => { textBufferRef.current += d; flushStreamingBuffers() },
                onToolCalls: () => { /* tool calls suppressed on retry */ },
                flushBuffers: (imm?: boolean) => flushStreamingBuffers(!!imm),
              } as any).catch(err => setError(String(err)))
            }
          }, 300)
        }
      }
    } catch (e:any) {
      if (e.name === 'AbortError') {
        setError('Pull canceled')
      } else {
        const msg = e.message || String(e)
        if (/network error/i.test(msg) || /failed to fetch/i.test(msg)) setError('Connection lost during pull. Ensure the server is running and retry.')
        else setError(msg)
        console.log('[pull] error', e)
      }
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
  try { saveSessions(updated) } catch { /* persist rename failed */ }
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

  // Prefer userAgentData when available; fallback to platform
  // navigator.platform is deprecated; keep as fallback for older Chromium versions
  const isMac = (navigator as any).userAgentData ? (navigator as any).userAgentData.platform === 'macOS' : /Mac|Darwin/.test(navigator.platform)
  // Drag & drop file support
  const [dragging, setDragging] = useState(false)
  useEffect(() => {
    function onDragOver(e: DragEvent) { e.preventDefault(); if (!dragging) setDragging(true) }
    function onDragLeave(e: DragEvent) { if ((e as any).relatedTarget == null) setDragging(false) }
    async function onDrop(e: DragEvent) {
      e.preventDefault(); setDragging(false)
      const files = Array.from(e.dataTransfer?.files || [])
          for (const file of files.slice(0,3)) { // limit to first 3
        if (file.size > 512 * 1024) continue // skip >512KB
        try {
          const text = await file.text()
          const snippet = text.length > 4000 ? text.slice(0,4000) + '\n...[truncated]' : text
              const userMsg: ChatMessage = { id: uuidv4(), role: 'user', content: `File: ${file.name}\n\n${snippet}` }
          setMessages(prev => [...prev, userMsg])
          if (session) {
            const updated: ChatSession = { ...session, messages: [...messages, userMsg], updatedAt: new Date().toISOString() }
            setSession(updated); upsertSession(updated)
          }
            } catch { /* ignore file read failure */ }
      }
    }
    window.addEventListener('dragover', onDragOver as any)
    window.addEventListener('dragleave', onDragLeave as any)
    window.addEventListener('drop', onDrop as any)
    return () => { window.removeEventListener('dragover', onDragOver as any); window.removeEventListener('dragleave', onDragLeave as any); window.removeEventListener('drop', onDrop as any) }
  }, [dragging, session, messages])
  return (
  <div className={`relative flex flex-row h-screen w-full overflow-hidden bg-[#0e0e0e] text-gray-200 ${isMac ? 'pt-[32px]' : ''}`}> {/* seamless canvas */}
      {dragging && (
        <div className='pointer-events-none absolute inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm text-gray-200 text-sm font-semibold border-2 border-dashed border-emerald-500'>
          Drop files to insert contents
        </div>
      )}
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
            let roleColor = 'text-gray-500'
            if (m.role === 'user') roleColor = 'text-gray-400'
            else if (m.role === 'assistant') roleColor = 'text-emerald-400'
            const roleLabel = (
              <span className={`block text-[11px] uppercase tracking-wide mb-1 font-medium ${roleColor}`}>{m.role}</span>
            )
            return (
        <div key={m.id || i} className='text-sm leading-relaxed rounded-lg px-4 py-3 bg-[#121212] border border-[#1b1b1b]'>
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
            <div className='text-xs mt-2'>
              <div className='flex items-center justify-between mb-1 select-none'>
                <span className='uppercase tracking-wide text-[10px] font-semibold text-amber-300'>Thinking</span>
                <div className='flex items-center gap-2'>
                  <button
                    onClick={() => toggleShowThinking()}
                    className='text-[10px] px-2 py-[2px] rounded bg-[#2a2a2a] hover:bg-[#333] border border-[#3a3a3a]'
                    title={showThinking ? 'Hide internal reasoning' : 'Show internal reasoning'}
                  >{showThinking ? 'Hide' : 'Show'}</button>
                  <button
                    onClick={() => { navigator.clipboard.writeText(thinking) }}
                    className='text-[10px] px-2 py-[2px] rounded bg-[#2a2a2a] hover:bg-[#333] border border-[#3a3a3a]'
                    title='Copy reasoning'
                  >Copy</button>
                </div>
              </div>
              {showThinking && (
                <div className='bg-[#1d1d1d] rounded p-3 font-mono text-[11px] max-h-56 overflow-auto whitespace-pre-wrap border border-amber-700/40 shadow-inner'>
                  {thinking}
                </div>
              )}
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
                  <button onClick={()=> lastPullTargetRef.current && pullModel(lastPullTargetRef.current)} className='mt-1 px-2 py-[2px] text-[10px] bg-amber-600 hover:bg-amber-500 rounded text-white'>Retry Pull {lastPullTargetRef.current}</button>
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
            {(() => {
              let toggleTitle = 'Enable reasoning'
              if (!supportsThinking) toggleTitle = 'Model does not support reasoning'
              else if (reasoningEnabled) toggleTitle = 'Disable reasoning'
              return (
                <button
                  onClick={() => supportsThinking && setReasoningEnabled(r => !r)}
                  disabled={!supportsThinking}
                  title={toggleTitle}
                  className={`text-xs px-2 border-r border-[#262626] ${supportsThinking ? 'opacity-100 hover:bg-[#1f1f1f]' : 'opacity-40 cursor-not-allowed'}`}
                >R</button>
              )
            })()}
            <input
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key==='Enter' && !e.shiftKey && (e.preventDefault(), onSend())}
              placeholder={running ? 'Generating…' : 'Send a message'}
              disabled={running}
              className='flex-1 bg-transparent px-2 py-2 text-sm outline-none disabled:opacity-50'
            />
            {running ? (
              <button
                onClick={() => providerRef.current.cancelCurrent?.()}
                className='px-4 py-2 text-sm font-semibold bg-red-600 hover:bg-red-500 transition rounded-none'
              >Cancel</button>
            ) : (
              <button
                onClick={onSend}
                disabled={running}
                className='px-4 py-2 text-sm font-semibold disabled:opacity-40 bg-emerald-600 hover:bg-emerald-500 transition rounded-none'
              >Send</button>
            )}
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
