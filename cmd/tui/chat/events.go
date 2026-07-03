package chat

import (
	"context"
	"slices"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	coreagent "github.com/ollama/ollama/agent"
	"github.com/ollama/ollama/api"
)

type chatAgentMsg struct {
	event coreagent.Event
}

type chatApprovalPromptMsg struct {
	request coreagent.ApprovalRequest
	reply   chan<- coreagent.Approval
}

type chatRunDoneMsg struct {
	result               *coreagent.RunResult
	err                  error
	newMessagesPersisted bool
	persistedMessages    []api.Message
}

type chatCompactDoneMsg struct {
	result coreagent.CompactionResult
	err    error
}

type chatCompactProgressMsg struct {
	tokens int
}

// resetStreamingState clears the transient streaming flags that every
// non-streaming event resets before applying its own state.
func (m *chatModel) resetStreamingState() {
	m.finishThinkingEntry()
	m.awaitingModel = false
	m.thinking = false
	m.thinkingTokens = 0
}

// resetRunState clears all run-progress flags (streaming plus compaction
// progress) for terminal events that fully reset the run view.
func (m *chatModel) resetRunState() {
	m.finishThinkingEntry()
	m.awaitingModel = false
	m.compacting = false
	m.compactingTokens = 0
	m.detectedToolCalls = nil
	m.thinking = false
	m.thinkingTokens = 0
}

type chatModelPreloadDoneMsg struct {
	model               string
	contextWindowTokens int
	err                 error
}

type chatEventsClosedMsg struct{}

type chatTickMsg struct{}

func (m *chatModel) applyAgentEvent(event coreagent.Event) {
	contextChanged := false

	switch event.Type {
	case coreagent.EventThinkingDelta:
		m.awaitingModel = false
		if event.Thinking != "" {
			m.thinking = true
			if event.Tokens > 0 {
				m.thinkingTokens = max(m.thinkingTokens, event.Tokens)
			} else {
				m.thinkingTokens += approximateTokenCount(event.Thinking)
			}
			idx := m.ensureLiveAssistantMessage()
			m.liveMessages[idx].Thinking += event.Thinking
			m.syncThinkingEntry()
			contextChanged = true
		}
	case coreagent.EventMessageDelta:
		m.resetStreamingState()
		m.groupCompletedToolHistory()
		m.detectedToolCalls = nil
		idx := m.ensureAssistantEntry()
		m.entries[idx].content += event.Content
		m.markEntryDirty(idx)
		msgIdx := m.ensureLiveAssistantMessage()
		m.liveMessages[msgIdx].Content += event.Content
		contextChanged = true
	case coreagent.EventToolCallDetected:
		m.finishThinkingEntry()
		m.awaitingModel = m.running
		m.thinking = false
		m.thinkingTokens = 0
		m.addDetectedToolCalls(event.ToolCalls)
		idx := m.ensureLiveAssistantMessage()
		m.liveMessages[idx].ToolCalls = append(m.liveMessages[idx].ToolCalls, event.ToolCalls...)
		contextChanged = true
	case coreagent.EventToolStarted:
		m.resetStreamingState()
		m.refreshContextWindowTokens(m.opts.Model)
		startedAt := time.Now()
		idx := m.findActiveToolEntry(event.ToolCallID)
		if idx < 0 {
			m.groupCompletedToolHistory()
			m.entries = append(m.entries, newChatEntry(chatEntry{role: "tool"}))
			idx = len(m.entries) - 1
		}
		m.entries[idx].detail = event.ToolName
		m.entries[idx].label = toolInvocationLabel(event.ToolName, event.Args)
		m.entries[idx].status = "running"
		m.entries[idx].toolID = event.ToolCallID
		m.entries[idx].args = event.Args
		m.entries[idx].startedAt = startedAt
		m.applyToolOutputModeTo(idx)
		m.markEntryDirty(idx)
		m.groupCompletedToolHistory()
	case coreagent.EventToolFinished:
		m.resetStreamingState()
		m.refreshContextWindowTokens(m.opts.Model)
		if event.WorkingDir != "" {
			m.workingDir = event.WorkingDir
		}
		startedAt := m.toolStartedAt(event.ToolCallID)
		status := toolFinishedStatus(event)
		idx := m.findToolEntry(event.ToolCallID)
		if idx < 0 {
			m.entries = append(m.entries, newChatEntry(chatEntry{role: "tool"}))
			idx = len(m.entries) - 1
		}
		m.entries[idx].content = event.Content
		m.entries[idx].label = toolInvocationLabel(event.ToolName, event.Args)
		m.entries[idx].detail = event.ToolName
		m.entries[idx].status = status
		if status != "denied" {
			m.entries[idx].err = event.Error
		}
		m.entries[idx].toolID = event.ToolCallID
		m.entries[idx].args = event.Args
		m.entries[idx].startedAt = startedAt
		m.entries[idx].finishedAt = time.Now()
		m.applyToolOutputModeTo(idx)
		m.markEntryDirty(idx)
		m.liveMessages = append(m.liveMessages, api.Message{
			Role:       "tool",
			Content:    event.Content,
			ToolName:   event.ToolName,
			ToolCallID: event.ToolCallID,
		})
		m.groupCompletedToolHistory()
		contextChanged = true
	case coreagent.EventCompacted:
		m.resetRunState()
		if len(event.Messages) > 0 {
			m.liveMessages = slices.Clone(event.Messages)
			m.messages = slices.Clone(event.Messages)
			contextChanged = true
		}
		m.status = "compacted"
	case coreagent.EventCompactionStarted:
		m.awaitingModel = false
		m.compacting = true
		m.compactingTokens = 0
		m.thinking = false
		m.thinkingTokens = 0
		m.status = "compacting"
	case coreagent.EventCompactionProgress:
		m.awaitingModel = false
		m.compacting = true
		m.thinking = false
		m.thinkingTokens = 0
		if event.Tokens > m.compactingTokens {
			m.compactingTokens = event.Tokens
		}
	case coreagent.EventCompactionSkipped:
		m.resetRunState()
		message := event.Content
		if strings.TrimSpace(message) == "" {
			message = coreagent.CompactionSkippedMessage(event.Error)
		}
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "system", content: message}))
		m.status = "compact skipped"
	case coreagent.EventError:
		m.resetRunState()
		m.eventErrorRendered = true
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: event.Error, err: event.Error}))
	}

	if contextChanged {
		m.refreshLiveContextEstimate()
	}
}

func (m *chatModel) addDetectedToolCalls(calls []api.ToolCall) {
	if len(calls) == 0 {
		return
	}
	seen := make(map[string]struct{}, len(m.detectedToolCalls)+len(calls))
	for _, entry := range m.detectedToolCalls {
		if entry.toolID != "" {
			seen[entry.toolID] = struct{}{}
		}
	}
	for _, call := range calls {
		if call.ID != "" {
			if _, ok := seen[call.ID]; ok {
				continue
			}
			seen[call.ID] = struct{}{}
		}
		args := call.Function.Arguments.ToMap()
		m.detectedToolCalls = append(m.detectedToolCalls, newChatEntry(chatEntry{
			role:   "tool",
			label:  toolInvocationLabel(call.Function.Name, args),
			detail: call.Function.Name,
			status: "queued",
			toolID: call.ID,
			args:   args,
		}))
	}
}

func toolFinishedStatus(event coreagent.Event) string {
	switch strings.TrimSpace(event.Status) {
	case "denied":
		return "denied"
	case "done":
		return "done"
	case "error":
		return "error"
	}
	if isDeniedToolResult(event.Content) || isDeniedToolResult(event.Error) {
		return "denied"
	}
	if event.Error != "" {
		return "error"
	}
	return "done"
}

func messagesEndWithCompactionResult(messages []api.Message) bool {
	if len(messages) == 0 {
		return false
	}
	return coreagent.IsCompactionToolResult(messages[len(messages)-1])
}

func (m chatModel) awaitingToolStart() bool {
	if len(m.liveMessages) == 0 {
		return false
	}
	msg := m.liveMessages[len(m.liveMessages)-1]
	if msg.Role != "assistant" || len(msg.ToolCalls) == 0 {
		return false
	}
	for _, call := range msg.ToolCalls {
		if call.ID == "" || m.findToolEntry(call.ID) < 0 {
			return true
		}
	}
	return false
}

func (m *chatModel) ensureLiveAssistantMessage() int {
	if len(m.liveMessages) > 0 && m.liveMessages[len(m.liveMessages)-1].Role == "assistant" {
		return len(m.liveMessages) - 1
	}
	m.liveMessages = append(m.liveMessages, api.Message{Role: "assistant"})
	return len(m.liveMessages) - 1
}

func (m *chatModel) refreshLiveContextEstimate() {
	messages := m.liveMessages
	if len(messages) == 0 {
		messages = m.messages
	}
	m.contextTokens = m.estimatePromptTokens(messages, "")
	m.contextEstimate = true
}

//nolint:containedctx // event sinks need the session context to unblock sends on cancellation.
type chatEventSink struct {
	ctx                  context.Context
	ch                   chan<- tea.Msg
	newMessagesPersisted *bool
}

func (s chatEventSink) Emit(event coreagent.Event) error {
	if s.newMessagesPersisted != nil {
		*s.newMessagesPersisted = true
	}
	select {
	case s.ch <- chatAgentMsg{event: event}:
		return nil
	case <-s.ctx.Done():
		return s.ctx.Err()
	}
}

func waitForChatMsg(ch <-chan tea.Msg) tea.Cmd {
	if ch == nil {
		return nil
	}
	return func() tea.Msg {
		msg, ok := <-ch
		if !ok {
			return chatEventsClosedMsg{}
		}
		return msg
	}
}

func (m *chatModel) scheduleTick() tea.Cmd {
	if m.tickActive {
		return nil
	}
	m.tickActive = true
	return chatTickCmd()
}

func chatTickCmd() tea.Cmd {
	return tea.Tick(350*time.Millisecond, func(time.Time) tea.Msg {
		return chatTickMsg{}
	})
}

func preloadModelCmd(ctx context.Context, preload func(context.Context, string, *api.ThinkValue) (int, error), model string, think *api.ThinkValue) tea.Cmd {
	if preload == nil || strings.TrimSpace(model) == "" {
		return nil
	}
	if think != nil {
		copied := *think
		think = &copied
	}
	return func() tea.Msg {
		if ctx == nil {
			ctx = context.Background()
		}
		tokens, err := preload(ctx, model, think)
		return chatModelPreloadDoneMsg{model: model, contextWindowTokens: tokens, err: err}
	}
}

func isUnsupportedThinkingError(err error) bool {
	if err == nil {
		return false
	}
	text := strings.ToLower(err.Error())
	return strings.Contains(text, "does not support thinking")
}

func thinkRequestsThinking(think *api.ThinkValue) bool {
	if think == nil {
		return false
	}
	return think.Bool()
}
