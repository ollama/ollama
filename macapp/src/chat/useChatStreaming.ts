import { useRef } from 'react'
import { ChatMessage, ChatProvider, ToolCall, ToolResult } from '../providers/ChatProvider'

export interface StreamingCallbacks {
  onThinking(delta: string): void
  onText(delta: string): void
  onToolCalls(calls: ToolCall[]): void
  flushBuffers(immediate?: boolean): void
}

interface UseChatStreamingOpts {
  providerRef: React.MutableRefObject<ChatProvider>
  maxLoops: number
  selectedModel: string
  normalizeModel: (id: string) => string
  allowThinking: () => boolean
  getToolCalls: () => ToolCall[]
  setToolCalls: (updater: (prev: ToolCall[]) => ToolCall[]) => void
}

export function useChatStreaming(opts: UseChatStreamingOpts) {
  const { providerRef, maxLoops, selectedModel, normalizeModel, allowThinking, getToolCalls, setToolCalls } = opts
  const lastHistoryRef = useRef<ChatMessage[]>([])

  async function runTurn(history: ChatMessage[], loopsRemaining: number, cb: StreamingCallbacks) {
    const provider = providerRef.current
    const normModel = normalizeModel(selectedModel) || selectedModel
    await provider.startChatTurn({ messages: history, maxLoopsRemaining: loopsRemaining, model: normModel, think: allowThinking() }, {
      onThinking: d => { cb.onThinking(d) },
      onText: d => { cb.onText(d) },
      onToolCall: call => {
        setToolCalls(prev => [...prev, { ...call, status: 'pending' } as any])
      },
      onError: err => console.error(err),
      onDone: () => cb.flushBuffers(true),
    })
    cb.flushBuffers(true)
  }

  async function executePendingTools(): Promise<ToolResult[]> {
    const pending = getToolCalls().filter((t: any) => t.status === 'pending') as (ToolCall & { status: string })[]
    if (!pending.length) return []
    const results = await providerRef.current.executeTools?.(pending) || []
    setToolCalls(prev => prev.map(tc => {
      const r = results.find(r => r.id === (tc as any).id)
      return r ? { ...(tc as any), status: 'done', output: (r as any).output } : tc
    }) as any)
    return results
  }

  async function agentLoop(baseMessages: ChatMessage[], cb: StreamingCallbacks) {
    let loops = maxLoops
    let history = [...baseMessages]
    lastHistoryRef.current = history
    while (loops > 0) {
      await runTurn(history, loops, cb)
      const newToolCalls = getToolCalls().filter((tc: any) => tc.status === 'pending') as any[]
      if (!newToolCalls.length) break
      const results = await executePendingTools()
      history = [...history, ...newToolCalls.map(tc => ({ role: 'tool' as const, content: results.find(r => r.id === tc.id)?.output || '', tool_name: tc.name }))]
      loops--
    }
  }

  return { runTurn, agentLoop, executePendingTools }
}
