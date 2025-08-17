import { ChatProvider, ChatTurnRequest, ChatTurnResult, ProviderCapabilities, StreamHandlers, ToolCall, ToolResult } from './ChatProvider'

const caps: ProviderCapabilities = { tools: true, thinking: true, maxContext: 32768, cloud: true }

export class DummyTurboProvider implements ChatProvider {
  private aborted = false
  capabilities(): ProviderCapabilities { return caps }
  cancelCurrent() { this.aborted = true }

  async startChatTurn(req: ChatTurnRequest, handlers: StreamHandlers): Promise<ChatTurnResult> {
    this.aborted = false
    const thinking = 'Analyzing request...'
    for (let i = 0; i < thinking.length; i++) {
      if (this.aborted) throw new Error('aborted')
      handlers.onThinking?.(thinking[i])
      await new Promise(r => setTimeout(r, 5))
    }
    await new Promise(r => setTimeout(r, 150))
    const userContent = req.messages.filter(m => m.role === 'user').slice(-1)[0]?.content || ''
    const base = `Turbo reply: ${userContent}`
    for (let i = 0; i < base.length; i++) {
      if (this.aborted) throw new Error('aborted')
      handlers.onText?.(base[i])
      await new Promise(r => setTimeout(r, 4))
    }
    // Simulate a mock tool call if trigger token present
    if (/\{\{call:/.test(userContent)) {
      const call: ToolCall = { id: 'turbo-mock-1', name: 'echo', arguments: { text: 'hello' } }
      handlers.onToolCall?.(call)
    }
    const final: ChatTurnResult = { message: { role: 'assistant', content: '' }, finishReason: 'stop' }
    handlers.onDone?.(final)
    return final
  }

  async executeTools(toolCalls: ToolCall[]): Promise<ToolResult[]> {
    return toolCalls.map(tc => ({ id: tc.id, name: tc.name, output: `turbo-mock(${tc.name})` }))
  }
}
