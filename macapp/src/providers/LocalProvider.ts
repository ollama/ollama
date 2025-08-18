import { ChatProvider, ChatTurnRequest, ChatTurnResult, ProviderCapabilities, StreamHandlers, ToolCall, ToolResult } from './ChatProvider'
import { normalizeModelId } from './modelNormalize'
import { v4 as uuidv4 } from 'uuid'

const caps: ProviderCapabilities = {
  tools: true,
  thinking: true, // backend can stream thinking for supported models
  maxContext: 8192,
}

export class LocalProvider implements ChatProvider {
  private abortController: AbortController | null = null

  private processEventLine(line: string, state: {
    final: ChatTurnResult
    assistantContent: string
    thinkingContent: string
    doneEmitted: boolean
    handlers: StreamHandlers
  }) {
    if (!line.trim() || state.doneEmitted) return
    let evt: any
  try { evt = JSON.parse(line) } catch { return }
    if (evt.error) { state.handlers.onError?.(new Error(evt.error)); return }
    const message = evt.message || {}
    if (message.thinking) {
      state.thinkingContent += message.thinking
      state.handlers.onThinking?.(message.thinking)
    }
    if (message.content) {
      state.assistantContent += message.content
      state.handlers.onText?.(message.content)
    }
    if (message.tool_calls) {
      for (const tc of message.tool_calls as any[]) {
        const call: ToolCall = { id: tc.id || tc.function?.name || uuidv4(), name: tc.function?.name, arguments: tc.function?.arguments || {} }
        state.handlers.onToolCall?.(call)
        state.final.toolCalls = [...(state.final.toolCalls || []), call]
      }
    }
    if (evt.done && !state.doneEmitted) {
      state.doneEmitted = true
      state.final.finishReason = evt.done_reason
      state.final.message = { role: 'assistant', content: state.assistantContent }
      state.handlers.onDone?.(state.final)
    }
  }

  capabilities(): ProviderCapabilities { return caps }

  async startChatTurn(req: ChatTurnRequest, handlers: StreamHandlers): Promise<ChatTurnResult> {
    this.abortController = new AbortController()
  const normalize = (m: string) => normalizeModelId(m) || m
    const body = {
      model: normalize(req.model || 'llama3.2'),
      messages: req.messages.map(m => ({ role: m.role, content: m.content, tool_name: m.tool_name })).filter(m => m.content || m.tool_name),
      tools: (req.tools || []).map(t => ({
        type: 'function',
        function: { name: t.name, description: t.description, parameters: { type: 'object', properties: {}, required: [] as string[] } },
      })),
      stream: true,
      think: req.think !== false,
    }

    try {
      const res = await fetch('http://localhost:11434/api/chat', {
        method: 'POST',
        body: JSON.stringify(body),
        headers: { 'Content-Type': 'application/json' },
        signal: this.abortController.signal,
      })
      if (!res.ok || !res.body) {
        let detail = ''
        try {
          const text = await res.text()
          if (text) {
            try {
              const j = JSON.parse(text)
              detail = j.error || j.message || text
            } catch {
              detail = text
            }
          }
  } catch { /* ignore parse/IO errors while extracting error detail */ }
        throw new Error(`chat failed: ${res.status}${detail ? ' - '+detail : ''}`)
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
  const final: ChatTurnResult = {}
      let buffer = ''
      const state = {
        final,
        assistantContent: '',
        thinkingContent: '',
        doneEmitted: false,
        handlers,
      }
      const STALL_MS = 25000
      let lastActivity = Date.now()

      const checkStall = () => {
        if (state.doneEmitted) return
    const elapsed = Date.now() - lastActivity
    const stalled = elapsed > STALL_MS
    if (stalled) {
          this.abortController?.abort()
          handlers.onError?.(new Error('stream stalled'))
        } else {
          setTimeout(checkStall, 5000)
        }
      }
      setTimeout(checkStall, 5000)

      let reading = true
      while (reading) {
        const { value, done } = await reader.read()
  if (done) { break }
        lastActivity = Date.now()
        buffer += decoder.decode(value, { stream: true })
        let idx: number
        while ((idx = buffer.indexOf('\n')) !== -1) {
          const line = buffer.slice(0, idx)
          buffer = buffer.slice(idx + 1)
          this.processEventLine(line, state)
          if (state.doneEmitted) { try { await reader.cancel() } catch { /* reader cancel failed */ } break }
        }
        if (state.doneEmitted) break
      }
      if (!state.doneEmitted) {
        // Stream ended without explicit done
        final.message = final.message || { role: 'assistant', content: state.assistantContent }
        final.finishReason = final.finishReason || 'incomplete'
        handlers.onDone?.(final)
      }
      return final
    } catch (err:any) {
      if (err.name === 'AbortError') {
        handlers.onError?.(new Error('aborted'))
      } else {
        handlers.onError?.(err)
      }
      throw err
    }
  }

  async executeTools(toolCalls: ToolCall[]): Promise<ToolResult[]> {
    if (toolCalls.length === 0) return []
    try {
      const body = {
        calls: toolCalls.map(tc => ({ id: tc.id, name: tc.name, arguments: tc.arguments || {} })),
      }
      const res = await fetch('http://localhost:11434/api/tools/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!res.ok) {
        const text = await res.text()
        throw new Error(`tool execute failed: ${res.status} ${text}`)
      }
      const json = await res.json()
      const resultsRaw = (json.results || []) as any[]
      return resultsRaw.map(r => ({ id: r.id, name: r.name, output: r.output }))
    } catch (e:any) {
      // Fallback: if endpoint not available yet, return mock outputs to avoid breaking UI
      console.warn('executeTools fallback', e)
      return toolCalls.map(tc => ({ id: tc.id, name: tc.name, output: `error: ${e.message || e}` }))
    }
  }

  cancelCurrent() { this.abortController?.abort() }
}
