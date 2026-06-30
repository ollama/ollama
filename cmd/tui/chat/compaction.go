package chat

import (
	"context"
	"slices"

	tea "github.com/charmbracelet/bubbletea"

	coreagent "github.com/ollama/ollama/agent"
	"github.com/ollama/ollama/api"
)

func (m *chatModel) startManualCompaction() (tea.Model, tea.Cmd) {
	if m.running || m.compacting {
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "system", content: "Wait for the current response to finish before compacting."}))
		return *m, nil
	}
	m.refreshContextWindowTokens(m.opts.Model)
	if m.opts.Compactor == nil {
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "system", content: coreagent.CompactionSkippedMessage("compaction is unavailable")}))
		m.status = "compact skipped"
		return *m, nil
	}

	ctx := m.ctx
	if ctx == nil {
		ctx = context.Background()
	}
	runCtx, cancel := context.WithCancel(ctx)
	compactor := m.opts.Compactor
	events := make(chan tea.Msg, 128)
	m.compacting = true
	m.compactingTokens = 0
	m.cancel = cancel
	m.compactEvents = events
	m.status = "compacting"
	messages := slices.Clone(m.messages)
	var tools api.Tools
	if m.opts.Tools != nil {
		tools = m.opts.Tools.Tools()
	}
	req := coreagent.CompactionRequest{
		ChatID:       m.chatID,
		Model:        m.opts.Model,
		SystemPrompt: m.systemPrompt(""),
		Messages:     messages,
		Tools:        tools,
		Format:       m.opts.Format,
		Options:      m.opts.Options,
		KeepAlive:    m.opts.KeepAlive,
		Force:        true,
		Progress: func(progress coreagent.CompactionProgress) {
			select {
			case events <- chatCompactProgressMsg{tokens: progress.Tokens}:
			case <-runCtx.Done():
			}
		},
	}
	go func() {
		defer close(events)
		result, err := compactor.MaybeCompact(runCtx, req)
		select {
		case events <- chatCompactDoneMsg{result: result, err: err}:
		case <-runCtx.Done():
		}
	}()
	tickCmd := m.scheduleTick()
	return *m, tea.Batch(waitForChatMsg(events), tickCmd)
}

func (m chatModel) finishManualCompaction(msg chatCompactDoneMsg) (tea.Model, tea.Cmd) {
	wasCanceling := m.status == "canceling"
	m.compacting = false
	m.compactEvents = nil
	m.cancel = nil
	m.compactingTokens = 0
	if wasCanceling || isChatContextCanceledError(msg.err) {
		m.status = "compact canceled"
		return m, m.startNextQueued()
	}
	if msg.err != nil {
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "system", content: coreagent.CompactionSkippedMessage(msg.err.Error())}))
		m.status = "compact skipped"
		return m, m.startNextQueued()
	}
	if !msg.result.Compacted {
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "system", content: coreagent.CompactionSkippedMessage(msg.result.Reason)}))
		m.status = "compact skipped"
		return m, m.startNextQueued()
	}

	m.messages = msg.result.Messages
	m.liveMessages = nil
	m.entries = entriesFromMessages(m.messages)
	m.contextTokens = m.estimatePromptTokens(m.messages, "")
	m.contextEstimate = true
	m.scroll = 0
	m.status = "compacted"
	return m, m.startNextQueued()
}
