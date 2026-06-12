package tui

import (
	"context"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	coreagent "github.com/ollama/ollama/agent"
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
	m.applyResponseMetrics(event.Response)

	switch event.Type {
	case coreagent.EventMessageStarted:
		m.thinking = false
		m.thinkingTokens = 0
	case coreagent.EventThinkingDelta:
		if event.Thinking != "" {
			m.thinking = true
			m.thinkingTokens = max(m.thinkingTokens, eventEvalCount(event))
			if eventEvalCount(event) <= 0 {
				m.thinkingTokens += estimateTokenCount(event.Thinking)
			}
		}
	case coreagent.EventMessageDelta:
		m.thinking = false
		m.thinkingTokens = 0
		m.groupCompletedToolHistory()
		idx := m.ensureAssistantEntry()
		m.entries[idx].content += event.Content
		m.markEntryDirty(idx)
	case coreagent.EventToolCallDetected:
		m.thinking = false
		m.thinkingTokens = 0
	case coreagent.EventToolStarted:
		m.thinking = false
		m.thinkingTokens = 0
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
	case coreagent.EventToolsUnavailable:
		m.thinking = false
		m.thinkingTokens = 0
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "system", content: "Tools are unavailable for this model."}))
	case coreagent.EventCompacted:
		m.thinking = false
		m.thinkingTokens = 0
		m.status = "compacted"
	case coreagent.EventCompactionSkipped:
		m.thinking = false
		m.thinkingTokens = 0
		message := event.Content
		if strings.TrimSpace(message) == "" {
			message = coreagent.CompactionSkippedMessage(event.Error)
		}
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "system", content: message}))
		m.status = "compact skipped"
	case coreagent.EventError:
		m.thinking = false
		m.thinkingTokens = 0
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: event.Error, err: event.Error}))
	}
}

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
