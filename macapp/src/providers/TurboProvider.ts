import { ChatProvider, ChatTurnRequest, ChatTurnResult, ProviderCapabilities, StreamHandlers, ToolCall, ToolResult } from './ChatProvider'
import { normalizeModelId } from './modelNormalize'
import { v4 as uuidv4 } from 'uuid'
import Store from 'electron-store'

// Minimal real Turbo provider scaffold: expects TURBO_BASE_URL + token (optional)
const caps: ProviderCapabilities = { tools: true, thinking: true, maxContext: 32768 }

const store = new Store<{ turboToken?: string, turboBase?: string }>({ name: 'turbo-config' })

export class TurboProvider implements ChatProvider {
  private abortController: AbortController | null = null
  capabilities(): ProviderCapabilities { return caps }

  private base(): string { return (store.get('turboBase') || process.env.TURBO_BASE_URL || '').replace(/\/$/, '') }
  private token(): string | undefined { return store.get('turboToken') || process.env.TURBO_API_KEY }

  async startChatTurn(req: ChatTurnRequest, handlers: StreamHandlers): Promise<ChatTurnResult> {
    this.abortController = new AbortController()
    const base = this.base()
    if (!base) {
      const err = new Error('Turbo base URL not configured. Open "Turbo config" and set Base URL.')
      handlers.onError?.(err)
      throw err
    }
    const body = {
      model: normalizeModelId(req.model) || 'turbo-default', // placeholder or selected
      messages: req.messages.map(m => ({ role: m.role, content: m.content, tool_name: m.tool_name })).filter(m => m.content || m.tool_name),
      stream: true,
      think: req.think !== false,
    }
    const url = base + '/api/chat'
    const headers: Record<string,string> = { 'Content-Type': 'application/json' }
    const tok = this.token(); if (tok) headers['Authorization'] = `Bearer ${tok}`

    let final: ChatTurnResult = {}
    let res: Response
    try {
      res = await fetch(url, { method: 'POST', body: JSON.stringify(body), headers, signal: this.abortController.signal })
    } catch (e:any) {
      const msg = e?.message || String(e)
      throw new Error(`turbo network error: ${msg}`)
    }
    if (!res.ok || !res.body) {
      let detail = ''
      try { detail = await res.text() } catch {}
      let reason = `turbo chat failed: ${res.status}`
      if (res.status === 404) reason += ' (model not found or unavailable)'
      if (res.status === 401 || res.status === 403) reason += ' (auth required/invalid token)'
      throw new Error(`${reason}${detail ? ' - ' + detail : ''}`)
    }
    const reader = res.body.getReader(); const decoder = new TextDecoder()
    while (true) {
      const { value, done } = await reader.read(); if (done) break
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
          if (evt.done) { final.finishReason = evt.done_reason; final.message = { role: 'assistant', content: '' }; handlers.onDone?.(final) }
        } catch {}
      }
    }
    return final
  }

  async executeTools(toolCalls: ToolCall[]): Promise<ToolResult[]> {
    if (toolCalls.length === 0) return []
    const url = this.base() + '/api/tools/execute'
    const headers: Record<string,string> = { 'Content-Type': 'application/json' }
    const tok = this.token(); if (tok) headers['Authorization'] = `Bearer ${tok}`
    const body = { calls: toolCalls.map(tc => ({ id: tc.id, name: tc.name, arguments: tc.arguments || {} })) }
    const res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body) })
    if (!res.ok) throw new Error('turbo tool exec failed')
    const json = await res.json(); return (json.results || []).map((r: any) => ({ id: r.id, name: r.name, output: r.output }))
  }

  cancelCurrent() { this.abortController?.abort() }
}
