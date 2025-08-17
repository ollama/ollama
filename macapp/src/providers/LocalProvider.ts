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
        } catch {}
        throw new Error(`chat failed: ${res.status}${detail ? ' - '+detail : ''}`)
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let final: ChatTurnResult = {}
      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        const text = decoder.decode(value)
        for (const line of text.split('\n')) {
          if (!line.trim()) continue
          try {
            const evt = JSON.parse(line)
            if (evt.error) { handlers.onError?.(new Error(evt.error)); continue }
            const message = evt.message || {}
            if (message.thinking) handlers.onThinking?.(message.thinking)
            if (message.content) handlers.onText?.(message.content)
            if (message.tool_calls) {
              for (const tc of message.tool_calls as any[]) {
                const call: ToolCall = { id: tc.id || tc.function?.name || uuidv4(), name: tc.function?.name, arguments: tc.function?.arguments || {} }
                handlers.onToolCall?.(call)
                final.toolCalls = [...(final.toolCalls || []), call]
              }
            }
            if (evt.done) {
              final.finishReason = evt.done_reason
              final.message = { role: 'assistant', content: '' }
              handlers.onDone?.(final)
            }
          } catch (e) {
            console.warn('parse line err', e)
          }
        }
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
