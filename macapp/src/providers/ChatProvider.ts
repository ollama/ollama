// Provider interfaces and shared chat types
// Phase 2 (T2)

export interface ToolCallArgumentSchema {
  name: string
  description?: string
  type: string // simple subset for now (string, number, object, array)
  required?: boolean
}

export interface ToolDefinition {
  name: string
  description: string
  // For MVP keep args as free-form; later can adopt JSON Schema
  parameters?: ToolCallArgumentSchema[]
}

export interface ToolCall {
  id: string
  name: string
  arguments: Record<string, unknown>
}

export interface ToolResult {
  id: string // matches ToolCall.id
  name: string
  output: string
  error?: string
  latencyMs?: number
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system' | 'tool'
  content: string
  tool_name?: string
  // Optional stable identifier for rendering/persistence (added later; legacy sessions may not have it)
  id?: string
}

export interface ChatTurnRequest {
  messages: ChatMessage[]
  tools?: ToolDefinition[]
  maxLoopsRemaining?: number
  signal?: AbortSignal
  model?: string // optional selected model identifier
  think?: boolean // request thinking tokens if supported (default true)
}

export interface StreamHandlers {
  onText?(delta: string): void
  onThinking?(delta: string): void
  onToolCall?(call: ToolCall): void
  onToolResult?(result: ToolResult): void
  onDone?(final: ChatTurnResult): void
  onError?(err: Error): void
}

export interface ChatTurnResult {
  message?: ChatMessage // final assistant message (without tool calls injected)
  toolCalls?: ToolCall[]
  finishReason?: string
}

export interface ProviderCapabilities {
  tools: boolean
  thinking: boolean
  maxContext: number
  cloud?: boolean
}

export interface ChatProvider {
  startChatTurn(req: ChatTurnRequest, handlers: StreamHandlers): Promise<ChatTurnResult>
  executeTools?(toolCalls: ToolCall[]): Promise<ToolResult[]>
  cancelCurrent?(): void
  capabilities(): ProviderCapabilities
}

export function isToolCallToken(text: string): boolean {
  // Simple heuristic for mock tool triggering e.g. {{call:echo}}
  return /\{\{call:[a-zA-Z0-9_.-]+\}\}/.test(text)
}

export function parseMockToolCalls(text: string): ToolCall[] {
  const matches = text.match(/\{\{call:([a-zA-Z0-9_.-]+)\}\}/g)
  if (!matches) return []
  return matches.map((m, idx) => {
    const name = m.slice(7, -2) // strip '{{call:' and '}}'
    return { id: `mock-${Date.now()}-${idx}`, name, arguments: {} }
  })
}
