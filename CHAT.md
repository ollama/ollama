# Agent Chat & Turbo Mode Specification

This document tracks the design and implementation progress for adding an **Agent Chat** feature and a future **Turbo (cloud) mode** to the macOS desktop app.

---

## 1. Goals

- Add multi-turn agent loop with tool calling.
- Support local (on-device) and future cloud (Turbo) execution paths via a pluggable provider interface.
- Surface reasoning ("thinking") and tool execution progress in UI.
- Prepare for cloud-hosted Turbo mode (auth, higher context, server-side tool loop).
- Maintain conversation persistence and session management.

---

## 2. Current State (Snapshot)

- `chat.tsx` streams `/api/chat` and accumulates content.
- Tool selection UI stub (`Tools.tsx`) with fetch to `/api/tools` (not yet server-backed).
- Backend supports tool call schema in streamed responses (`ToolCalls`).
- No loop to execute tool calls and re-query the model.
- Turbo toggle in UI is a placeholder.
done          – completion (finish_reason)
---

## 3. Architecture Overview

### Providers

Common interface abstracts chat + tool execution across local and Turbo providers.

```ts
interface ChatProvider {
  startChatTurn(req: ChatTurnRequest, handlers: StreamHandlers): Promise<ChatTurnResult>
  executeTools?(toolCalls: ToolCall[]): Promise<ToolResult[]>
  cancelCurrent?(): void
  capabilities(): ProviderCapabilities
}

interface ProviderCapabilities {
  tools: boolean
  thinking: boolean
  maxContext: number
  cloud?: boolean
}
```

### Event Model (Unified)

Runtime streaming events normalized:

```text
text          – incremental assistant text
thinking      – incremental reasoning tokens
tool_call     – model proposing a tool call
tool_result   – result of executed tool
done          – completion (finish_reason)
error         – error event
```

### Agent Loop (Client-Orchestrated MVP)

1. Send user + prior messages to provider with tool definitions.
2. Stream assistant output; collect tool calls.
3. If tool calls present:
   - Execute (client -> provider.executeTools -> backend).
   - Append each tool result as `tool` messages.
   - Re-enter chat turn with updated history.
4. Repeat until no new tool calls or iteration cap reached.
\n### Future (Server-Orchestrated Turbo)

Turbo provider’s backend executes tool loop internally, streaming `tool_call` and `tool_result` events directly.

---

## 4. Backend Extensions (Planned)

| Endpoint | Purpose | MVP Phase |
|----------|---------|-----------|
| `GET /api/tools` | List available tools (schema + description) | Later (local) |
| `POST /api/tools/execute` | Execute tool calls and return outputs | Later (local) |
| Cloud `POST /v1/agent/chat` | Combined streaming w/ server loop | Turbo phase |
| Cloud `GET /v1/agent/tools` | Cloud tool listing | Turbo |

Initial implementation proceeds with **dummy Turbo** (no backend).

---

## 5. Tool Model

Backend JSON schema style aligns with OpenAI function-calling. Tool execution results are injected as messages:

```json
{ "role": "tool", "tool_name": "time.now", "content": "2025-08-16T00:00:00Z" }
```

---

## 6. Iteration Limits & Safety

- `MAX_TOOL_LOOPS` default 4.
- Early stop if no tool calls.
- Abort support via `AbortController` / provider `cancelCurrent`.

---

## 7. UI Enhancements (Later Phases)

- Tool execution panel with per-call status.
- Reasoning collapse/expand.
- Sidebar persisted sessions (`electron-store`).
- Settings: context slider -> provider max; Turbo default toggle; reasoning visibility.
- Fallback banners (quota, network, auth).

---

## 8. Authentication (Turbo Future)

- OAuth / device code to get tokens.
- `AuthManager` caches & refreshes tokens.
- Turbo requests add `Authorization` header.

---

## 9. Metrics (Future)

Metrics to capture per turn:

- Provider
- Duration
- Token counts (prompt, completion, total)
- Tool count
- Errors (categorize: network, model, tool, timeout)

---

## 10. Phased Delivery Plan

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Create spec & checklist | ✅ Done |
| 2 | Provider interfaces + dummy Turbo provider skeleton | ✅ Done (ChatProvider, DummyTurboProvider) |
| 3 | Refactor `chat.tsx` to use `LocalProvider` | ✅ Done (ChatView + LocalProvider abstraction) |
| 4 | Implement agent loop (client) | ✅ Done (loop w/ MAX_LOOPS=4) |
| 5 | Basic Turbo toggle UI & state | ✅ Done (provider select incl. dummy + real Turbo scaffold) |
| 6 | Tool execution via local endpoint | ✅ Done (`/api/tools`, `/api/tools/execute`, executeTools wired) |
| 7 | Reasoning panel & tool call UI | ✅ Done (collapsible reasoning + status list) |
| 8 | Session persistence | ✅ Done (electron-store sessions) |
| 9 | Real Turbo backend integration | ✅ Initial scaffold (env/base + token) |
| 10 | Auth & quotas | ✅ Basic token store + quota placeholder |
| 11 | Security hardening & caps | ✅ Iteration cap, abort, allowlist + sanitation |

---

## 11. Detailed Task Checklist

### Completed Core Tasks

- [x] (T2) Add `ChatProvider` interface & shared types.
- [x] (T3) Implement `LocalProvider` using existing streaming.
- [x] (T4) Add dummy `TurboProvider` (simulated thinking + response).
- [x] (T5) Refactor chat to route through selected provider (`ChatView`).
- [x] (T6) UI provider toggle (Local / Turbo / Turbo Dummy).
- [x] (T7) Agent loop with tool call detection & iteration cap.
- [x] (T8) Checklist updates.
- [x] Real tool execution endpoints + client wiring.
- [x] Reasoning panel (collapsible) & tool call progress UI.
- [x] Session persistence via `electron-store`.
- [x] Turbo provider scaffold (env base + auth header).
- [x] Auth token + base URL config panel.
- [x] Quota & metrics placeholder (context chars, loops used).
- [x] Security: allowlist tools, sanitize outputs, abort handling, loop cap enforcement.

### Remaining / Future Enhancements

- [ ] Full Turbo backend integration (server-side loop, token metrics).
- [ ] Expanded tool schema (parameter validation & JSON schema generation).
- [ ] Advanced tool suite (web.fetch, vector search, filesystem restricted ops).
- [ ] Rich metrics (token counts, latency histograms, tool timing breakdown).
- [ ] CI SBOM & security instrumentation.
- [ ] UI polish (separate sidebar, drag-resize reasoning panel, dark mode refinements).
- [ ] Error categorization & retry UX.

---

## 12. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Infinite agent loops | Iteration cap + UI stop button |
| Tool schema drift | Shared TypeScript + Go generation later |
| Complex refactor regressions | Keep provider refactor additive; feature flag toggle |
| Network errors degrade UX | Auto fallback to local provider with notice |

---

## 13. Success Criteria

- Local mode unchanged performance post-refactor.
- Dummy Turbo produces distinct thinking segment within 300ms of first token.
- Tool call loop works with mock tool returning deterministic output.
- Clear visual indication of active provider.

---

## 14. Implementation Notes

- Keep changes to `chat.tsx` localized: wrap existing logic into provider adapter first, then prune legacy path.
- Introduce new files under `macapp/src/providers/`.
- Dummy Turbo: generate pseudo thinking with setInterval, then final content.
- For tool call mock: If assistant text contains token `{{call:echo}}` parse as tool call.

---

## 15. Open Questions (To Address Later)

- Schema versioning for tools? (Add `version` field?)
- Max reasoning token limit for display?
- Offline caching of cloud failures (retry queue)?

---

## 16. Change Log

| Date | Change |
|------|--------|
| 2025-08-16 | Initial spec created (Phase 1) |
| 2025-08-16 | Formatting fixes for markdown lint compliance |
| 2025-08-16 | Implemented provider interfaces, Local & Dummy Turbo providers, agent loop, tool endpoints, reasoning & tool UI, session persistence, Turbo scaffold, auth config, security hardening |

---

_End of spec._
