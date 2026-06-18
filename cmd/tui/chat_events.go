package tui

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
	reply   chan<- coreagent.ApprovalResult
}

type chatRunDoneMsg struct {
	result *coreagent.RunResult
	err    error
}

type chatCompactDoneMsg struct {
	result coreagent.CompactionResult
	err    error
}

type chatCompactProgressMsg struct {
	tokens int
}

type chatTickMsg struct{}

func (m *chatModel) applyAgentEvent(event coreagent.Event) {
	contextChanged := false

	switch event.Type {
	case coreagent.EventMessageStarted:
		m.lastEventError = ""
		m.compacting = false
		m.compactingTokens = 0
		m.thinking = false
		m.thinkingTokens = 0
		m.refreshContextWindowTokens(m.opts.Model)
	case coreagent.EventThinkingDelta:
		if event.Thinking != "" {
			m.thinking = true
			m.thinkingTokens = max(m.thinkingTokens, eventEvalCount(event))
			if eventEvalCount(event) <= 0 {
				m.thinkingTokens += estimateTokenCount(event.Thinking)
			}
			idx := m.ensureLiveAssistantMessage()
			m.liveMessages[idx].Thinking += event.Thinking
			contextChanged = true
		}
	case coreagent.EventMessageDelta:
		m.thinking = false
		m.thinkingTokens = 0
		m.groupCompletedToolHistory()
		idx := m.ensureAssistantEntry()
		m.entries[idx].content += event.Content
		m.markEntryDirty(idx)
		msgIdx := m.ensureLiveAssistantMessage()
		m.liveMessages[msgIdx].Content += event.Content
		contextChanged = true
	case coreagent.EventToolCallDetected:
		m.thinking = false
		m.thinkingTokens = 0
		idx := m.ensureLiveAssistantMessage()
		m.liveMessages[idx].ToolCalls = append(m.liveMessages[idx].ToolCalls, event.ToolCalls...)
		contextChanged = true
	case coreagent.EventToolStarted:
		m.thinking = false
		m.thinkingTokens = 0
		m.refreshContextWindowTokens(m.opts.Model)
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
		m.entries[idx].startedAt = event.StartedAt
		m.applyToolOutputModeTo(idx)
		m.markEntryDirty(idx)
	case coreagent.EventToolFinished:
		m.thinking = false
		m.thinkingTokens = 0
		m.refreshContextWindowTokens(m.opts.Model)
		if event.WorkingDir != "" {
			m.workingDir = event.WorkingDir
		}
		startedAt := m.toolStartedAt(event.ToolCallID)
		status := "done"
		if event.Error != "" {
			status = "error"
		}
		idx := m.findToolEntry(event.ToolCallID)
		if idx < 0 {
			m.entries = append(m.entries, newChatEntry(chatEntry{role: "tool"}))
			idx = len(m.entries) - 1
		}
		m.entries[idx].content = event.Content
		m.entries[idx].label = toolInvocationLabel(event.ToolName, event.Args)
		m.entries[idx].detail = event.ToolName
		m.entries[idx].status = status
		m.entries[idx].err = event.Error
		m.entries[idx].toolID = event.ToolCallID
		m.entries[idx].args = event.Args
		m.entries[idx].startedAt = startedAt
		m.entries[idx].finishedAt = event.FinishedAt
		m.applyToolOutputModeTo(idx)
		m.markEntryDirty(idx)
		m.liveMessages = append(m.liveMessages, api.Message{
			Role:       "tool",
			Content:    event.Content,
			ToolName:   event.ToolName,
			ToolCallID: event.ToolCallID,
		})
		contextChanged = true
	case coreagent.EventToolsUnavailable:
		m.thinking = false
		m.thinkingTokens = 0
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "system", content: "Tools are unavailable for this model."}))
	case coreagent.EventCompacted:
		m.compacting = false
		m.compactingTokens = 0
		m.thinking = false
		m.thinkingTokens = 0
		if len(event.Messages) > 0 {
			m.liveMessages = slices.Clone(event.Messages)
			contextChanged = true
		}
		m.status = "compacted"
	case coreagent.EventCompactionStarted:
		m.compacting = true
		m.compactingTokens = 0
		m.thinking = false
		m.thinkingTokens = 0
		m.status = "compacting"
	case coreagent.EventCompactionProgress:
		m.compacting = true
		m.thinking = false
		m.thinkingTokens = 0
		if event.Tokens > m.compactingTokens {
			m.compactingTokens = event.Tokens
		}
	case coreagent.EventCompactionSkipped:
		m.compacting = false
		m.compactingTokens = 0
		m.thinking = false
		m.thinkingTokens = 0
		message := event.Content
		if strings.TrimSpace(message) == "" {
			message = coreagent.CompactionSkippedMessage(event.Error)
		}
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "system", content: message}))
		m.status = "compact skipped"
	case coreagent.EventError:
		m.compacting = false
		m.compactingTokens = 0
		m.thinking = false
		m.thinkingTokens = 0
		m.lastEventError = event.Error
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: event.Error, err: event.Error}))
	}

	if contextChanged {
		m.refreshLiveContextEstimate()
	}
	m.applyResponseMetrics(event.Response)
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
	ctx context.Context
	ch  chan<- tea.Msg
}

func (s chatEventSink) Emit(event coreagent.Event) error {
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
			return nil
		}
		return msg
	}
}

func chatTickCmd() tea.Cmd {
	return tea.Tick(120*time.Millisecond, func(time.Time) tea.Msg {
		return chatTickMsg{}
	})
}
