package chat

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	coreagent "github.com/ollama/ollama/agent"
	"github.com/ollama/ollama/api"
)

func TestChatEnterQueuesWhileRunning(t *testing.T) {
	m := chatModel{
		input:   []rune("next prompt"),
		running: true,
	}

	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = updated.(chatModel)

	if cmd != nil {
		t.Fatal("queueing during a run should not start another command immediately")
	}
	if !m.running {
		t.Fatal("current run should stay active")
	}
	if got := string(m.input); got != "" {
		t.Fatalf("input = %q, want cleared after queue", got)
	}
	if len(m.queued) != 1 || m.queued[0] != "next prompt" {
		t.Fatalf("queued = %#v, want next prompt", m.queued)
	}
	if !strings.Contains(stripANSI(m.View()), "queued 1") {
		t.Fatalf("view should show queued count: %q", stripANSI(m.View()))
	}
}

func compactionToolCallMessage() api.Message {
	return api.Message{Role: "assistant", ToolCalls: []api.ToolCall{{
		ID: coreagent.CompactionToolCallID,
		Function: api.ToolCallFunction{
			Name: coreagent.CompactionToolName,
		},
	}}}
}

func TestChatRunDoneStartsQueuedMessage(t *testing.T) {
	m := chatModel{
		ctx:     context.Background(),
		running: true,
		queued:  []string{"next prompt"},
		opts: Options{
			Model:  "test",
			Client: chatTestClient{},
		},
	}

	updated, cmd := m.Update(chatRunDoneMsg{result: &coreagent.RunResult{Messages: []api.Message{{Role: "assistant", Content: "done"}}}})
	m = updated.(chatModel)

	if cmd == nil {
		t.Fatal("queued message should start a new run")
	}
	if !m.running {
		t.Fatal("queued message should be running")
	}
	if len(m.queued) != 0 {
		t.Fatalf("queued = %#v, want empty after start", m.queued)
	}
	if len(m.entries) == 0 || m.entries[len(m.entries)-1].role != "user" || m.entries[len(m.entries)-1].content != "next prompt" {
		t.Fatalf("last entry should be queued user prompt: %#v", m.entries)
	}
}

func TestChatResizeEnablesBoundedFrame(t *testing.T) {
	m := chatModel{}

	updated, cmd := m.Update(tea.WindowSizeMsg{Width: 80, Height: 24})
	m = updated.(chatModel)
	if cmd != nil {
		t.Fatal("initial window size should not clear the screen")
	}
	if m.boundedFrame {
		t.Fatal("initial window size should keep terminal-flow rendering")
	}

	updated, cmd = m.Update(tea.WindowSizeMsg{Width: 100, Height: 30})
	m = updated.(chatModel)
	if cmd == nil {
		t.Fatal("resize should clear stale terminal-flow rendering")
	}
	if !m.boundedFrame {
		t.Fatal("resize should enable bounded frame rendering")
	}
}

func TestChatTickAdvancesSpinnerWhileRunning(t *testing.T) {
	m := chatModel{running: true}

	updated, cmd := m.Update(chatTickMsg{})
	fm := updated.(chatModel)

	if fm.spinner != 1 {
		t.Fatalf("spinner = %d, want 1", fm.spinner)
	}
	if cmd == nil {
		t.Fatal("running tick should schedule another tick")
	}
}

func TestChatTickStopsWhenIdle(t *testing.T) {
	m := chatModel{}

	updated, cmd := m.Update(chatTickMsg{})
	fm := updated.(chatModel)

	if fm.spinner != 0 {
		t.Fatalf("spinner = %d, want 0", fm.spinner)
	}
	if cmd != nil {
		t.Fatal("idle tick should not schedule another tick")
	}
}

func TestChatActivityLabelOmitsResponding(t *testing.T) {
	m := chatModel{
		running: true,
		entries: []chatEntry{
			{role: "assistant", content: "hello"},
		},
	}

	if got := m.activityLabel(); got != "" {
		t.Fatalf("activityLabel = %q, want empty", got)
	}
}

func TestChatActivityLineDelaysWaitingForModelSpinner(t *testing.T) {
	m := chatModel{running: true}

	if got := m.activityLabel(); got != "" {
		t.Fatalf("activityLabel = %q, want empty", got)
	}
	if got := stripANSI(m.activityLine()); got != "" {
		t.Fatalf("activityLine = %q, want delayed empty line", got)
	}
	m.spinner = waitingSpinnerTicks
	line := strings.TrimSpace(stripANSI(m.activityLine()))
	if !strings.Contains(line, "Working...") {
		t.Fatalf("activityLine = %q, want Working label", line)
	}
	if strings.Contains(line, "waiting for model") {
		t.Fatalf("activityLine = %q, want Working label without old waiting label", line)
	}
}

func TestChatActivityLineShowsDelayedWaitingSpinnerOnFollowUp(t *testing.T) {
	m := chatModel{
		running: true,
		spinner: waitingSpinnerTicks,
		entries: []chatEntry{
			{role: "user", content: "first"},
			{role: "assistant", content: "done"},
			{role: "user", content: "follow up"},
		},
	}

	if got := m.activityLabel(); got != "" {
		t.Fatalf("activityLabel = %q, want empty", got)
	}
	if got := strings.TrimSpace(stripANSI(m.activityLine())); !strings.Contains(got, "Working...") || strings.Contains(got, "waiting for model") {
		t.Fatalf("activityLine = %q, want delayed Working label without old waiting label", got)
	}
}

func TestChatActivityLineShowsWaitingAfterToolResultBeforeFollowUpStream(t *testing.T) {
	m := chatModel{
		running: true,
		spinner: waitingSpinnerTicks,
		entries: []chatEntry{
			{role: "user", content: "look it up"},
			{role: "assistant", content: "I'll search."},
			{role: "tool", label: "Web Search(\"query\")", status: "done", content: "result"},
		},
	}

	if got := strings.TrimSpace(stripANSI(m.activityLine())); strings.Contains(got, "Working") {
		t.Fatalf("activityLine = %q, want no waiting label before next request", got)
	}

	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventRequestBuilt})
	if got := strings.TrimSpace(stripANSI(m.activityLine())); !strings.Contains(got, "Working...") {
		t.Fatalf("activityLine = %q, want Working label while follow-up request waits", got)
	}

	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventMessageStarted})
	if got := strings.TrimSpace(stripANSI(m.activityLine())); !strings.Contains(got, "Working...") {
		t.Fatalf("activityLine = %q, want Working label until visible stream output", got)
	}

	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventMessageDelta, Content: "done"})
	if got := strings.TrimSpace(stripANSI(m.activityLine())); strings.Contains(got, "Working") {
		t.Fatalf("activityLine = %q, want waiting label cleared when visible output arrives", got)
	}
}

func TestChatActivityLineShowsWorkingWhileToolCallWaitsForToolStart(t *testing.T) {
	args := api.NewToolCallFunctionArguments()
	args.Set("command", "pwd")
	m := chatModel{
		running: true,
		spinner: waitingSpinnerTicks,
	}

	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventRequestBuilt})
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventMessageStarted})
	m.applyAgentEvent(coreagent.Event{
		Type: coreagent.EventToolCallDetected,
		ToolCalls: []api.ToolCall{{
			ID: "call-1",
			Function: api.ToolCallFunction{
				Name:      "bash",
				Arguments: args,
			},
		}},
	})
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventModelStreamDone})

	if len(m.entries) != 0 {
		t.Fatalf("detected tool call should not create history entries: %#v", m.entries)
	}
	if got := strings.TrimSpace(stripANSI(m.activityLine())); !strings.Contains(got, "Working...") {
		t.Fatalf("activityLine = %q, want Working label while waiting for tool start", got)
	}

	m.applyAgentEvent(coreagent.Event{
		Type:       coreagent.EventToolStarted,
		ToolCallID: "call-1",
		ToolName:   "bash",
		Args:       args.ToMap(),
	})
	if got := strings.TrimSpace(stripANSI(m.activityLine())); got != "" {
		t.Fatalf("activityLine = %q, want active tool row to replace spinner", got)
	}
}

func TestChatModelPreloadIsSilentAndRefreshesContext(t *testing.T) {
	compactor := coreagent.NewSimpleCompactor(nil, nil, coreagent.CompactionOptions{
		ContextWindowTokens: 262144,
	})
	m := chatModel{
		ctx:             context.Background(),
		preloadingModel: "llama3.2",
		spinner:         waitingSpinnerTicks,
		opts: Options{
			Model:               "llama3.2",
			Compactor:           compactor,
			ContextWindowTokens: 262144,
			ContextWindowTokensForModel: func(_ context.Context, model string, fallback int) int {
				if model != "llama3.2" || fallback != 262144 {
					t.Fatalf("resolver called with model=%q fallback=%d", model, fallback)
				}
				return 8192
			},
		},
	}

	if line := stripANSI(m.activityLine()); strings.TrimSpace(line) != "" {
		t.Fatalf("activityLine = %q, want silent preload", line)
	}

	updated, cmd := m.Update(chatModelPreloadDoneMsg{model: "llama3.2"})
	if cmd != nil {
		t.Fatal("preload completion should not schedule a command")
	}
	fm := updated.(chatModel)
	if fm.preloadingModel != "" {
		t.Fatalf("preloadingModel = %q, want empty", fm.preloadingModel)
	}
	if fm.opts.ContextWindowTokens != 8192 {
		t.Fatalf("ContextWindowTokens = %d, want effective runner window 8192", fm.opts.ContextWindowTokens)
	}
	if compactor.Options.ContextWindowTokens != 8192 {
		t.Fatalf("compactor ContextWindowTokens = %d, want 8192", compactor.Options.ContextWindowTokens)
	}
}

func TestChatModelPreloadUnsupportedThinkingDisablesThinkingAndRetries(t *testing.T) {
	think := &api.ThinkValue{Value: "high"}
	var retryThink *api.ThinkValue
	m := chatModel{
		ctx:             context.Background(),
		preloadingModel: "llama3.2",
		opts: Options{
			Think: think,
			PreloadModel: func(ctx context.Context, model string, think *api.ThinkValue) error {
				if think != nil {
					copied := *think
					retryThink = &copied
				}
				return nil
			},
		},
	}

	updated, cmd := m.Update(chatModelPreloadDoneMsg{
		model: "llama3.2",
		err:   errors.New(`400 Bad Request: "llama3.2" does not support thinking`),
	})
	fm := updated.(chatModel)

	if cmd == nil {
		t.Fatal("unsupported thinking should retry preload with thinking disabled")
	}
	if fm.opts.Think == nil || fm.opts.Think.Bool() {
		t.Fatalf("think = %#v, want disabled", fm.opts.Think)
	}
	if fm.preloadingModel != "llama3.2" {
		t.Fatalf("preloadingModel = %q, want llama3.2", fm.preloadingModel)
	}
	if len(fm.entries) != 0 {
		t.Fatalf("unsupported thinking should not render a preload error: %#v", fm.entries)
	}

	msg := cmd()
	batch, ok := msg.(tea.BatchMsg)
	if !ok || len(batch) == 0 {
		t.Fatalf("retry command = %#v, want tea.BatchMsg", msg)
	}
	if got := batch[0](); got == nil {
		t.Fatal("preload retry command returned nil message")
	}
	if retryThink == nil || retryThink.Bool() {
		t.Fatalf("retry think = %#v, want disabled", retryThink)
	}
}

func TestChatModelPreloadIgnoresStaleCompletion(t *testing.T) {
	m := chatModel{preloadingModel: "qwen3"}

	updated, _ := m.Update(chatModelPreloadDoneMsg{model: "llama3.2"})
	fm := updated.(chatModel)
	if fm.preloadingModel != "qwen3" {
		t.Fatalf("preloadingModel = %q, want qwen3", fm.preloadingModel)
	}
}

func TestChatThinkingShowsTokenCount(t *testing.T) {
	m := chatModel{running: true}

	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventThinkingDelta, Thinking: "abcdefgh"})

	if got := m.activityLabel(); got != "Thinking 2 tokens" {
		t.Fatalf("activityLabel = %q, want Thinking 2 tokens", got)
	}
	if strings.Contains(stripANSI(m.renderTranscript(80)), "abcdefgh") {
		t.Fatalf("thinking text should not render in transcript: %q", stripANSI(m.renderTranscript(80)))
	}

	response := api.ChatResponse{Metrics: api.Metrics{EvalCount: 12}}
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventThinkingDelta, Thinking: "more", Response: &response})
	if got := m.activityLabel(); got != "Thinking 12 tokens" {
		t.Fatalf("activityLabel = %q, want Thinking 12 tokens", got)
	}

	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventMessageDelta, Content: "done"})
	if m.thinking || m.thinkingTokens != 0 {
		t.Fatalf("thinking state was not cleared: thinking=%v tokens=%d", m.thinking, m.thinkingTokens)
	}
}

func TestChatModelLineShowsContextOnlyWhenUseful(t *testing.T) {
	m := chatModel{
		width:           120,
		height:          24,
		contextTokens:   50,
		contextEstimate: true,
		opts: Options{
			Model:               "llama3.2",
			Options:             map[string]any{"num_ctx": 100},
			CompactionThreshold: 0.75,
		},
	}

	view := stripANSI(m.View())
	if strings.Contains(view, "ctx ") {
		t.Fatalf("view should hide distant context pressure: %q", view)
	}
	if strings.Contains(view, "compact at") || strings.Contains(view, "compact due") {
		t.Fatalf("view should not use old compaction copy: %q", view)
	}

	m.contextTokens = 61
	view = stripANSI(m.View())
	if !strings.Contains(view, "llama3.2  ctx ~61/100 (61%)") || strings.Contains(view, "compaction soon") {
		t.Fatalf("view should show context count beside model after 60%% without compaction copy: %q", view)
	}
	if count := strings.Count(view, "ctx "); count != 1 {
		t.Fatalf("context should render only in the footer/model line, got %d occurrences: %q", count, view)
	}

	m.contextTokens = 65
	view = stripANSI(m.View())
	if !strings.Contains(view, "llama3.2  ctx ~65/100 (65%)") || strings.Contains(view, "compaction soon") {
		t.Fatalf("view should show context count near threshold without compaction copy: %q", view)
	}
	if count := strings.Count(view, "ctx "); count != 1 {
		t.Fatalf("context should render only in the footer/model line, got %d occurrences: %q", count, view)
	}

	m.contextTokens = 75
	view = stripANSI(m.View())
	if !strings.Contains(view, "llama3.2  ctx ~75/100 (75%)") || strings.Contains(view, "compaction soon") {
		t.Fatalf("view should show context count at threshold without compaction copy: %q", view)
	}

	m.contextTokens = 20
	m.status = "compacted"
	view = stripANSI(m.View())
	if strings.Contains(view, "after compaction") || strings.Contains(view, "ctx ") {
		t.Fatalf("view should not show after-compaction copy or distant context pressure: %q", view)
	}

	m.status = ""
	m.contextTokens = 125
	view = stripANSI(m.View())
	if !strings.Contains(view, "ctx ~125/100 (125%)") {
		t.Fatalf("view should show over-window usage instead of clamping: %q", view)
	}
}

func TestChatStartRunEstimatesFullPrompt(t *testing.T) {
	registry := coreagent.NewRegistry()
	registry.Register(chatTestTool{})

	systemPrompt := strings.Repeat("system prompt ", 20)
	userMsg := api.Message{Role: "user", Content: "hello"}
	m := chatModel{
		ctx:   context.Background(),
		input: []rune(userMsg.Content),
		opts: Options{
			Model:        "test",
			Client:       chatTestClient{},
			Tools:        registry,
			SystemPrompt: systemPrompt,
		},
	}

	updated, _ := m.handleSubmit()
	fm := updated.(chatModel)
	if fm.cancel != nil {
		fm.cancel()
	}

	want := estimatePromptTokenCount(systemPrompt, []api.Message{userMsg}, registry.Tools(), "")
	if fm.contextTokens != want {
		t.Fatalf("contextTokens = %d, want full prompt estimate %d", fm.contextTokens, want)
	}

	messageOnly := estimatePromptTokenCount("", []api.Message{userMsg}, nil, "")
	if fm.contextTokens <= messageOnly {
		t.Fatalf("context estimate should include system prompt and tools: got %d, message-only %d", fm.contextTokens, messageOnly)
	}
	if !fm.contextEstimate {
		t.Fatal("initial context count should be marked estimated")
	}
}

func TestChatStartRunRefreshesEffectiveContextWindow(t *testing.T) {
	compactor := coreagent.NewSimpleCompactor(nil, nil, coreagent.CompactionOptions{
		ContextWindowTokens: 262144,
	})
	m := chatModel{
		ctx:   context.Background(),
		input: []rune("hello"),
		opts: Options{
			Model:               "llama3.2",
			Client:              chatTestClient{},
			Compactor:           compactor,
			ContextWindowTokens: 262144,
			ContextWindowTokensForModel: func(_ context.Context, model string, fallback int) int {
				if model != "llama3.2" || fallback != 262144 {
					t.Fatalf("resolver called with model=%q fallback=%d", model, fallback)
				}
				return 8192
			},
		},
	}

	updated, _ := m.handleSubmit()
	fm := updated.(chatModel)
	if fm.cancel != nil {
		fm.cancel()
	}
	if fm.opts.ContextWindowTokens != 8192 {
		t.Fatalf("ContextWindowTokens = %d, want effective runner window 8192", fm.opts.ContextWindowTokens)
	}
	if compactor.Options.ContextWindowTokens != 8192 {
		t.Fatalf("compactor ContextWindowTokens = %d, want 8192", compactor.Options.ContextWindowTokens)
	}
}

func TestChatRunDoneKeepsAPIPromptEvalCount(t *testing.T) {
	registry := coreagent.NewRegistry()
	registry.Register(chatTestTool{})
	messages := []api.Message{{Role: "user", Content: "hello"}}
	m := chatModel{
		opts: Options{
			Tools:        registry,
			SystemPrompt: strings.Repeat("system prompt ", 20),
		},
	}

	updated, _ := m.Update(chatRunDoneMsg{
		result: &coreagent.RunResult{
			Messages: messages,
			Latest: api.ChatResponse{
				Metrics: api.Metrics{PromptEvalCount: 123},
			},
		},
	})

	fm := updated.(chatModel)
	if fm.contextTokens != 123 {
		t.Fatalf("contextTokens = %d, want API prompt eval count", fm.contextTokens)
	}
	if fm.contextEstimate {
		t.Fatal("API prompt eval count should not be marked estimated")
	}
}

func TestChatRunDoneIncludesGeneratedTokenCount(t *testing.T) {
	m := chatModel{}

	updated, _ := m.Update(chatRunDoneMsg{
		result: &coreagent.RunResult{
			Messages: []api.Message{{Role: "assistant", Content: "done"}},
			Latest: api.ChatResponse{
				Metrics: api.Metrics{PromptEvalCount: 123, EvalCount: 7},
			},
		},
	})

	fm := updated.(chatModel)
	if fm.contextTokens != 130 {
		t.Fatalf("contextTokens = %d, want prompt + generated token count", fm.contextTokens)
	}
	if fm.contextEstimate {
		t.Fatal("API token counts should not be marked estimated after run completion")
	}
}

func TestChatRunDoneSuppressesDuplicateEventError(t *testing.T) {
	eventErr := errors.New("tool round limit reached; send another message to continue")
	err := errors.New("agent stopped: tool round limit reached; send another message to continue")
	m := chatModel{running: true}
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventError, Error: eventErr.Error()})

	updated, _ := m.Update(chatRunDoneMsg{err: err, newMessagesPersisted: true})
	fm := updated.(chatModel)

	var errorEntries int
	for _, entry := range fm.entries {
		if entry.role == "error" {
			errorEntries++
		}
	}
	if errorEntries != 1 {
		t.Fatalf("error entries = %d, want one streamed error", errorEntries)
	}
	if fm.status != "error" {
		t.Fatalf("status = %q, want error", fm.status)
	}
}

func TestChatRunDoneKeepsOnlyPersistedMessagesAfterRunError(t *testing.T) {
	err := errors.New("model connection failed")
	persisted := []api.Message{
		{Role: "user", Content: "old prompt"},
		{Role: "assistant", Content: "old answer"},
		{Role: "user", Content: "new prompt"},
	}
	m := chatModel{
		running: true,
		messages: []api.Message{
			{Role: "user", Content: "old prompt"},
			{Role: "assistant", Content: "old answer"},
		},
		liveMessages: []api.Message{
			{Role: "user", Content: "old prompt"},
			{Role: "assistant", Content: "old answer"},
			{Role: "user", Content: "new prompt"},
			{Role: "assistant", Content: "partial assistant not persisted"},
		},
	}

	updated, _ := m.Update(chatRunDoneMsg{err: err, newMessagesPersisted: true, persistedMessages: persisted})
	fm := updated.(chatModel)

	if len(fm.messages) != 3 || fm.messages[2].Content != "new prompt" {
		t.Fatalf("messages = %#v, want persisted submitted messages after error", fm.messages)
	}
	for _, msg := range fm.messages {
		if strings.Contains(msg.Content, "partial assistant") {
			t.Fatalf("messages = %#v, should not include unpersisted assistant text", fm.messages)
		}
	}
	if fm.liveMessages != nil {
		t.Fatalf("liveMessages = %#v, want cleared after error", fm.liveMessages)
	}
	if fm.status != "error" {
		t.Fatalf("status = %q, want error", fm.status)
	}
}

func TestChatRunDoneDoesNotPromoteLiveMessagesBeforePersistence(t *testing.T) {
	err := errors.New("prompt is too large for the current context")
	m := chatModel{
		running: true,
		messages: []api.Message{
			{Role: "user", Content: "old prompt"},
			{Role: "assistant", Content: "old answer"},
		},
		liveMessages: []api.Message{
			{Role: "user", Content: "old prompt"},
			{Role: "assistant", Content: "old answer"},
			{Role: "user", Content: "oversized prompt"},
		},
	}

	updated, _ := m.Update(chatRunDoneMsg{err: err})
	fm := updated.(chatModel)

	if len(fm.messages) != 2 || fm.messages[1].Content != "old answer" {
		t.Fatalf("messages = %#v, want previous persisted messages only", fm.messages)
	}
	if fm.liveMessages != nil {
		t.Fatalf("liveMessages = %#v, want cleared after failed preflight", fm.liveMessages)
	}
	if fm.status != "error" {
		t.Fatalf("status = %q, want error", fm.status)
	}
}

func TestChatStreamingMetricsDoNotDropLiveContextEstimate(t *testing.T) {
	m := chatModel{
		running: true,
		liveMessages: []api.Message{
			{Role: "user", Content: strings.Repeat("large prompt ", 400)},
		},
		contextTokens:   900,
		contextEstimate: true,
	}

	m.applyAgentEvent(coreagent.Event{
		Type:    coreagent.EventMessageDelta,
		Content: "partial response",
		Response: &api.ChatResponse{
			Metrics: api.Metrics{PromptEvalCount: 12},
		},
	})

	if m.contextTokens == 12 {
		t.Fatal("streaming prompt metrics should not replace the live context estimate")
	}
	if m.contextTokens < 900 {
		t.Fatalf("contextTokens dropped during streaming: got %d, want at least 900", m.contextTokens)
	}
	if !m.contextEstimate {
		t.Fatal("live context should remain marked estimated while the model is running")
	}
}

func TestChatCompactedEventUsesCompactedPromptEstimate(t *testing.T) {
	messages := []api.Message{
		compactionToolCallMessage(),
		{Role: "tool", ToolName: coreagent.CompactionToolName, ToolCallID: coreagent.CompactionToolCallID, Content: coreagent.CompactionSummaryMessagePrefix + "summary"},
	}
	m := chatModel{
		running:       true,
		contextTokens: 3900,
	}

	m.applyAgentEvent(coreagent.Event{
		Type:         coreagent.EventCompacted,
		Messages:     messages,
		PromptTokens: 420,
		Response: &api.ChatResponse{
			Metrics: api.Metrics{PromptEvalCount: 3900, EvalCount: 100},
		},
	})

	if m.contextTokens != 420 {
		t.Fatalf("contextTokens = %d, want compacted prompt estimate", m.contextTokens)
	}
	if !m.contextEstimate {
		t.Fatal("compacted prompt count should remain marked estimated")
	}
	if len(m.messages) != len(messages) || len(m.liveMessages) != len(messages) {
		t.Fatalf("compacted messages were not promoted: messages=%#v live=%#v", m.messages, m.liveMessages)
	}

	m.applyAgentEvent(coreagent.Event{
		Type: coreagent.EventRunFinished,
		Response: &api.ChatResponse{
			Metrics: api.Metrics{PromptEvalCount: 3900, EvalCount: 100},
		},
	})
	if m.contextTokens != 420 {
		t.Fatalf("run-finished metrics reset contextTokens to %d, want compacted prompt estimate", m.contextTokens)
	}
}

func TestChatRunDoneKeepsCompactedEstimateWhenCompactionEndsTurn(t *testing.T) {
	messages := []api.Message{
		compactionToolCallMessage(),
		{Role: "tool", ToolName: coreagent.CompactionToolName, ToolCallID: coreagent.CompactionToolCallID, Content: coreagent.CompactionSummaryMessagePrefix + "summary"},
	}
	m := chatModel{}
	want := m.estimatePromptTokens(messages, "")

	updated, _ := m.Update(chatRunDoneMsg{
		result: &coreagent.RunResult{
			Messages: messages,
			Latest: api.ChatResponse{
				Metrics: api.Metrics{PromptEvalCount: 3900, EvalCount: 100},
			},
		},
	})

	fm := updated.(chatModel)
	if fm.contextTokens != want {
		t.Fatalf("contextTokens = %d, want compacted estimate %d", fm.contextTokens, want)
	}
	if !fm.contextEstimate {
		t.Fatal("compacted end-of-turn count should remain marked estimated")
	}
}

func TestChatAgentEventsRefreshLiveContextEstimateForToolCalls(t *testing.T) {
	args := api.NewToolCallFunctionArguments()
	args.Set("query", "Parth Sareen")
	toolCall := api.ToolCall{
		ID: "call-1",
		Function: api.ToolCallFunction{
			Name:      "web_search",
			Arguments: args,
		},
	}

	m := chatModel{
		ctx:   context.Background(),
		input: []rune("who is parth"),
		opts: Options{
			Model:  "test",
			Client: chatTestClient{},
		},
	}

	updated, _ := m.handleSubmit()
	m = updated.(chatModel)
	if m.cancel != nil {
		m.cancel()
	}
	initial := m.contextTokens

	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventMessageDelta, Content: "I'll search."})
	afterAssistant := m.contextTokens
	if afterAssistant <= initial {
		t.Fatalf("assistant delta should increase context estimate: initial=%d after=%d", initial, afterAssistant)
	}

	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventToolCallDetected, ToolCalls: []api.ToolCall{toolCall}})
	afterToolCall := m.contextTokens
	if afterToolCall <= afterAssistant {
		t.Fatalf("tool call should increase context estimate: assistant=%d after=%d", afterAssistant, afterToolCall)
	}

	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventToolFinished, ToolCallID: "call-1", ToolName: "web_search", Content: strings.Repeat("result ", 20)})
	if m.contextTokens <= afterToolCall {
		t.Fatalf("tool output should increase context estimate: tool call=%d after=%d", afterToolCall, m.contextTokens)
	}
	if !m.contextEstimate {
		t.Fatal("live context should remain marked estimated")
	}
}

func TestChatVerboseRunDoneRendersModelMetrics(t *testing.T) {
	m := chatModel{
		opts:    Options{Verbose: true},
		entries: []chatEntry{newChatEntry(chatEntry{role: "assistant", content: "done"})},
	}

	updated, _ := m.Update(chatRunDoneMsg{
		result: &coreagent.RunResult{
			Messages: []api.Message{{Role: "assistant", Content: "done"}},
			Latest: api.ChatResponse{
				Metrics: api.Metrics{
					TotalDuration:      617565500 * time.Nanosecond,
					LoadDuration:       222099916 * time.Nanosecond,
					PromptEvalCount:    55,
					PromptEvalDuration: 103384 * time.Microsecond,
					EvalCount:          28,
					EvalDuration:       290393 * time.Microsecond,
				},
			},
		},
	})
	m = updated.(chatModel)

	transcript := stripANSI(m.renderTranscript(160))
	for _, want := range []string{
		"done",
		"total duration:       617.5655ms",
		"load duration:        222.099916ms",
		"prompt eval count:    55 token(s)",
		"prompt eval duration: 103.384ms",
		"prompt eval rate:     532.00 tokens/s",
		"eval count:           28 token(s)",
		"eval duration:        290.393ms",
		"eval rate:            96.42 tokens/s",
	} {
		if !strings.Contains(transcript, want) {
			t.Fatalf("transcript missing %q:\n%s", want, transcript)
		}
	}
}

func TestChatRunDoneOmitsModelMetricsWithoutVerbose(t *testing.T) {
	m := chatModel{
		entries: []chatEntry{newChatEntry(chatEntry{role: "assistant", content: "done"})},
	}

	updated, _ := m.Update(chatRunDoneMsg{
		result: &coreagent.RunResult{
			Messages: []api.Message{{Role: "assistant", Content: "done"}},
			Latest: api.ChatResponse{
				Metrics: api.Metrics{TotalDuration: time.Second, EvalCount: 1},
			},
		},
	})
	m = updated.(chatModel)

	if transcript := stripANSI(m.renderTranscript(120)); strings.Contains(transcript, "total duration:") {
		t.Fatalf("non-verbose transcript should not include metrics:\n%s", transcript)
	}
}

func TestChatViewShowsRunningActivityOnce(t *testing.T) {
	m := chatModel{
		running:        true,
		thinking:       true,
		thinkingTokens: 7,
		width:          80,
		height:         24,
	}

	view := stripANSI(m.View())
	if got := strings.Count(view, "Thinking 7 tokens"); got != 1 {
		t.Fatalf("thinking activity rendered %d times, want 1:\n%s", got, view)
	}
	if strings.Contains(view, "sent ") || strings.Contains(view, "received ") {
		t.Fatalf("view should not show sent/received metrics: %q", view)
	}
}

func TestChatViewShowsCompactingActivity(t *testing.T) {
	m := chatModel{
		compacting:       true,
		compactingTokens: 42,
		width:            80,
		height:           24,
	}

	view := stripANSI(m.View())
	if got := strings.Count(view, "Compacting 42 tokens"); got != 1 {
		t.Fatalf("compacting activity rendered %d times, want 1:\n%s", got, view)
	}
}

func TestChatAutoCompactionEventsShowActivity(t *testing.T) {
	m := chatModel{running: true}

	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventCompactionStarted})
	if !m.compacting {
		t.Fatal("compaction start should mark the chat as compacting")
	}
	if got := m.activityLabel(); got != "Compacting" {
		t.Fatalf("activityLabel = %q, want Compacting", got)
	}

	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventCompactionProgress, Tokens: 12})
	if got := m.activityLabel(); got != "Compacting 12 tokens" {
		t.Fatalf("activityLabel = %q, want Compacting 12 tokens", got)
	}

	m.applyAgentEvent(coreagent.Event{
		Type: coreagent.EventCompacted,
		Messages: []api.Message{
			compactionToolCallMessage(),
			{Role: "tool", ToolName: coreagent.CompactionToolName, ToolCallID: coreagent.CompactionToolCallID, Content: coreagent.CompactionSummaryMessagePrefix + "summary"},
		},
	})
	if m.compacting {
		t.Fatal("compacted event should clear compacting state")
	}
}

func TestChatCtrlCCancelsQuietly(t *testing.T) {
	canceled := false
	m := chatModel{
		running: true,
		cancel: func() {
			canceled = true
		},
	}

	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyCtrlC})
	m = updated.(chatModel)
	if !canceled {
		t.Fatal("ctrl+c should cancel an active run")
	}
	if cmd != nil {
		t.Fatal("ctrl+c during a run should not quit")
	}
	if m.status != "canceling" {
		t.Fatalf("status = %q, want canceling", m.status)
	}

	updated, _ = m.Update(chatRunDoneMsg{err: context.Canceled})
	m = updated.(chatModel)
	if m.running {
		t.Fatal("run should no longer be active after cancellation completes")
	}
	if m.status != "Tell the model what to do instead." {
		t.Fatalf("status = %q, want friendly cancellation hint", m.status)
	}
	for _, entry := range m.entries {
		if entry.role == "error" {
			t.Fatalf("cancellation should not append an error entry: %#v", entry)
		}
	}
}

func TestChatClosedEventChannelCompletesCanceledRun(t *testing.T) {
	m := chatModel{
		running:        true,
		status:         "canceling",
		thinking:       true,
		thinkingTokens: 42,
		cancel:         func() {},
		events:         make(chan tea.Msg),
		liveMessages: []api.Message{
			{Role: "user", Content: "write a summary"},
			{Role: "assistant", Content: "partial answer"},
		},
		entries: []chatEntry{
			{role: "user", content: "write a summary"},
			{role: "assistant", content: "partial answer"},
		},
	}

	updated, cmd := m.Update(chatEventsClosedMsg{})
	m = updated.(chatModel)
	if cmd == nil {
		t.Fatal("closed event channel should flush the preserved partial transcript")
	}
	if m.running {
		t.Fatal("run should be cleared when event channel closes during cancel")
	}
	if m.thinking || m.thinkingTokens != 0 {
		t.Fatalf("thinking state should be cleared: thinking=%v tokens=%d", m.thinking, m.thinkingTokens)
	}
	if m.cancel != nil || m.events != nil {
		t.Fatalf("cancel/events should be cleared: cancel=%v events=%v", m.cancel, m.events)
	}
	if len(m.messages) != 2 || m.messages[1].Content != "partial answer" {
		t.Fatalf("messages = %#v, want promoted partial live messages", m.messages)
	}
	if m.liveMessages != nil {
		t.Fatalf("liveMessages = %#v, want cleared after promotion", m.liveMessages)
	}
	if transcript := stripANSI(m.renderTranscript(80)); !strings.Contains(transcript, "partial answer") {
		t.Fatalf("transcript lost partial response: %q", transcript)
	}
	if m.status != "Tell the model what to do instead." {
		t.Fatalf("status = %q, want friendly cancellation hint", m.status)
	}
}

func TestWaitForChatMsgReportsClosedChannel(t *testing.T) {
	ch := make(chan tea.Msg)
	close(ch)

	msg := waitForChatMsg(ch)()
	if _, ok := msg.(chatEventsClosedMsg); !ok {
		t.Fatalf("waitForChatMsg closed channel = %#v, want chatEventsClosedMsg", msg)
	}
}

func TestChatRunDoneTreatsHTTPContextCanceledStringAsCancellation(t *testing.T) {
	m := chatModel{running: true}

	updated, _ := m.Update(chatRunDoneMsg{err: errors.New(`Post "http://127.0.0.1:11434/api/chat": context canceled`)})
	m = updated.(chatModel)
	if m.running {
		t.Fatal("run should no longer be active after cancellation completes")
	}
	if m.status != "Tell the model what to do instead." {
		t.Fatalf("status = %q, want friendly cancellation hint", m.status)
	}
	for _, entry := range m.entries {
		if entry.role == "error" {
			t.Fatalf("cancellation should not append an error entry: %#v", entry)
		}
	}
}

func TestChatCtrlCClearsDraftInsteadOfQuitting(t *testing.T) {
	m := chatModel{
		input:     []rune("draft"),
		complete:  2,
		quitArmed: true,
		status:    "press ctrl+c again to quit",
	}

	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyCtrlC})
	m = updated.(chatModel)

	if cmd != nil {
		t.Fatal("ctrl+c with draft input should not quit")
	}
	if got := string(m.input); got != "" {
		t.Fatalf("input = %q, want cleared", got)
	}
	if m.complete != 0 {
		t.Fatalf("complete = %d, want reset", m.complete)
	}
	if m.quitArmed || m.quitArmedKey != "" || m.quitting {
		t.Fatalf("quit state = armed %v key %q quitting %v, want false", m.quitArmed, m.quitArmedKey, m.quitting)
	}
	if m.status != "ready" {
		t.Fatalf("status = %q, want ready", m.status)
	}
}

func TestChatCtrlCClearsDraftWhileRunningBeforeCanceling(t *testing.T) {
	canceled := false
	m := chatModel{
		input:   []rune("queued prompt"),
		running: true,
		cancel: func() {
			canceled = true
		},
	}

	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyCtrlC})
	m = updated.(chatModel)

	if cmd != nil {
		t.Fatal("ctrl+c with draft input should not quit")
	}
	if canceled {
		t.Fatal("ctrl+c should clear draft input before canceling a run")
	}
	if got := string(m.input); got != "" {
		t.Fatalf("input = %q, want cleared", got)
	}
	if !m.running {
		t.Fatal("run should remain active after clearing draft input")
	}
	if m.status != "ready" {
		t.Fatalf("status = %q, want ready", m.status)
	}
}

func TestChatDoubleEscClearsDraft(t *testing.T) {
	m := chatModel{
		input:    []rune("draft"),
		complete: 2,
	}

	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEsc})
	m = updated.(chatModel)
	if cmd != nil {
		t.Fatal("first esc should not quit")
	}
	if got := string(m.input); got != "draft" {
		t.Fatalf("first esc input = %q, want unchanged", got)
	}
	if !m.escArmed {
		t.Fatal("first esc should arm clear")
	}
	if !strings.Contains(m.status, "press esc again") {
		t.Fatalf("status = %q, want esc hint", m.status)
	}

	updated, cmd = m.Update(tea.KeyMsg{Type: tea.KeyEsc})
	m = updated.(chatModel)
	if cmd != nil {
		t.Fatal("second esc should not quit")
	}
	if got := string(m.input); got != "" {
		t.Fatalf("input = %q, want cleared", got)
	}
	if m.complete != 0 {
		t.Fatalf("complete = %d, want reset", m.complete)
	}
	if m.escArmed {
		t.Fatal("second esc should disarm clear")
	}
	if m.status != "ready" {
		t.Fatalf("status = %q, want ready", m.status)
	}
}

func TestChatDoubleEscClearsDraftAndCancelsRun(t *testing.T) {
	canceled := false
	m := chatModel{
		input:   []rune("queued prompt"),
		running: true,
		cancel: func() {
			canceled = true
		},
	}

	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEsc})
	m = updated.(chatModel)
	if cmd != nil {
		t.Fatal("first esc should not quit")
	}
	if canceled {
		t.Fatal("first esc should not cancel")
	}
	if got := string(m.input); got != "queued prompt" {
		t.Fatalf("first esc input = %q, want unchanged", got)
	}
	if !m.escArmed {
		t.Fatal("first esc should arm clear/cancel")
	}

	updated, cmd = m.Update(tea.KeyMsg{Type: tea.KeyEsc})
	m = updated.(chatModel)
	if cmd != nil {
		t.Fatal("second esc should not quit")
	}
	if !canceled {
		t.Fatal("second esc should cancel active run")
	}
	if got := string(m.input); got != "" {
		t.Fatalf("input = %q, want cleared", got)
	}
	if m.escArmed {
		t.Fatal("second esc should disarm")
	}
	if m.status != "canceling" {
		t.Fatalf("status = %q, want canceling", m.status)
	}
}

func TestChatEscClearHintDisarmsOnOtherKey(t *testing.T) {
	m := chatModel{input: []rune("draft")}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyEsc})
	m = updated.(chatModel)
	if !m.escArmed {
		t.Fatal("first esc should arm clear")
	}

	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("x")})
	m = updated.(chatModel)
	if m.escArmed {
		t.Fatal("typing should disarm esc confirmation")
	}
	if m.status != "ready" {
		t.Fatalf("status = %q, want ready", m.status)
	}
	if got := string(m.input); got != "draftx" {
		t.Fatalf("input = %q, want draftx", got)
	}
}

func TestChatCtrlCRequiresSecondPressToQuit(t *testing.T) {
	m := chatModel{}

	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyCtrlC})
	m = updated.(chatModel)
	if cmd != nil {
		t.Fatal("first idle ctrl+c should only arm quit")
	}
	if !m.quitArmed || m.quitArmedKey != "ctrl+c" || m.quitting {
		t.Fatalf("quit state = armed %v key %q quitting %v, want armed ctrl+c only", m.quitArmed, m.quitArmedKey, m.quitting)
	}
	if m.status != "press ctrl+c again to quit" {
		t.Fatalf("status = %q, want quit confirmation", m.status)
	}

	updated, cmd = m.Update(tea.KeyMsg{Type: tea.KeyCtrlC})
	m = updated.(chatModel)
	if cmd == nil {
		t.Fatal("second idle ctrl+c should quit")
	}
	if !m.quitting {
		t.Fatal("model should be marked quitting")
	}
}

func TestChatCtrlCQuitWarningDisarmsOnOtherKey(t *testing.T) {
	m := chatModel{}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyCtrlC})
	m = updated.(chatModel)
	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("x")})
	m = updated.(chatModel)

	if m.quitArmed {
		t.Fatal("typing should disarm quit confirmation")
	}
	if m.quitArmedKey != "" {
		t.Fatalf("quitArmedKey = %q, want empty", m.quitArmedKey)
	}
	if m.status != "ready" {
		t.Fatalf("status = %q, want ready", m.status)
	}
	if got := string(m.input); got != "x" {
		t.Fatalf("input = %q, want x", got)
	}
}

func TestChatQuitConfirmationRequiresMatchingKey(t *testing.T) {
	m := chatModel{}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyCtrlC})
	m = updated.(chatModel)
	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyCtrlD})
	m = updated.(chatModel)
	if cmd != nil || m.quitting {
		t.Fatal("ctrl+d should not quit after a ctrl+c confirmation")
	}
	if !m.quitArmed || m.quitArmedKey != "ctrl+d" {
		t.Fatalf("quit state = armed %v key %q, want ctrl+d armed", m.quitArmed, m.quitArmedKey)
	}
	if m.status != "press ctrl+d again to quit" {
		t.Fatalf("status = %q, want ctrl+d confirmation", m.status)
	}
}

func TestChatCtrlDExitsWithEmptyInput(t *testing.T) {
	m := chatModel{}

	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyCtrlD})
	m = updated.(chatModel)
	if cmd != nil {
		t.Fatal("first ctrl+d with empty input should only arm quit")
	}
	if !m.quitArmed || m.quitArmedKey != "ctrl+d" || m.quitting {
		t.Fatalf("quit state = armed %v key %q quitting %v, want armed ctrl+d only", m.quitArmed, m.quitArmedKey, m.quitting)
	}
	if m.status != "press ctrl+d again to quit" {
		t.Fatalf("status = %q, want ctrl+d confirmation", m.status)
	}

	updated, cmd = m.Update(tea.KeyMsg{Type: tea.KeyCtrlD})
	m = updated.(chatModel)
	if cmd == nil {
		t.Fatal("second ctrl+d with empty input should quit")
	}
	if !m.quitting {
		t.Fatal("model should be marked quitting")
	}
}

func TestChatCtrlDDoesNotExitWithDraftInput(t *testing.T) {
	m := chatModel{input: []rune("draft")}

	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyCtrlD})
	m = updated.(chatModel)
	if cmd != nil {
		t.Fatal("ctrl+d with draft input should not quit")
	}
	if m.quitting {
		t.Fatal("model should not be marked quitting")
	}
	if got := string(m.input); got != "draft" {
		t.Fatalf("input = %q, want unchanged", got)
	}
}

func TestChatCtrlDCancelsActiveRunWhenExiting(t *testing.T) {
	canceled := false
	m := chatModel{
		running: true,
		cancel: func() {
			canceled = true
		},
	}

	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyCtrlD})
	m = updated.(chatModel)
	if cmd != nil {
		t.Fatal("first ctrl+d should only arm quit")
	}
	if canceled {
		t.Fatal("first ctrl+d should not cancel active run")
	}

	updated, cmd = m.Update(tea.KeyMsg{Type: tea.KeyCtrlD})
	m = updated.(chatModel)
	if cmd == nil {
		t.Fatal("second ctrl+d with empty input should quit")
	}
	if !canceled {
		t.Fatal("second ctrl+d should cancel active run before quitting")
	}
	if !m.quitting {
		t.Fatal("model should be marked quitting")
	}
}

func TestChatClearCommandResetsConversation(t *testing.T) {
	m := chatModel{
		ctx:      context.Background(),
		chatID:   "old",
		messages: []api.Message{{Role: "user", Content: "hello"}},
		entries:  []chatEntry{{role: "user", content: "hello"}},
		queued:   []string{"later"},
		input:    []rune("/clear"),
		opts: Options{
			NewChat: func(context.Context) (string, error) {
				return "new", nil
			},
		},
	}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("clear command should not return a command")
	}

	fm := updated.(chatModel)
	if fm.chatID != "new" {
		t.Fatalf("chatID = %q, want new", fm.chatID)
	}
	if len(fm.messages) != 0 || len(fm.entries) != 0 || len(fm.queued) != 0 {
		t.Fatalf("conversation was not cleared: messages=%d entries=%d queued=%d", len(fm.messages), len(fm.entries), len(fm.queued))
	}
	if fm.status != "cleared" {
		t.Fatalf("status = %q, want cleared", fm.status)
	}
}

func TestChatNewCommandStartsFreshChat(t *testing.T) {
	m := chatModel{
		ctx:             context.Background(),
		chatID:          "old",
		messages:        []api.Message{{Role: "user", Content: "hello"}},
		entries:         []chatEntry{{role: "user", content: "hello"}},
		contextTokens:   42,
		contextEstimate: true,
		input:           []rune("/new"),
		opts: Options{
			NewChat: func(context.Context) (string, error) {
				return "fresh", nil
			},
		},
	}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("new command should not return a command")
	}

	fm := updated.(chatModel)
	if fm.chatID != "fresh" {
		t.Fatalf("chatID = %q, want fresh", fm.chatID)
	}
	if len(fm.messages) != 0 || len(fm.entries) != 0 {
		t.Fatalf("conversation was not reset: messages=%d entries=%d", len(fm.messages), len(fm.entries))
	}
	if fm.contextTokens != 0 {
		t.Fatalf("contextTokens = %d, want 0", fm.contextTokens)
	}
	if fm.status != "new chat" {
		t.Fatalf("status = %q, want new chat", fm.status)
	}
}

func TestChatCompactCommandShowsSummary(t *testing.T) {
	compactor := &chatTestCompactor{
		progress: []int{12},
		result: coreagent.CompactionResult{
			Messages: []api.Message{
				{Role: "user", Content: "recent request"},
				compactionToolCallMessage(),
				{Role: "tool", ToolName: coreagent.CompactionToolName, ToolCallID: coreagent.CompactionToolCallID, Content: coreagent.CompactionSummaryMessagePrefix + "old work summary"},
			},
			Compacted: true,
			Due:       true,
			Summary:   "old work summary",
		},
	}
	m := chatModel{
		ctx:          context.Background(),
		chatID:       "chat-1",
		messages:     []api.Message{{Role: "user", Content: "old request"}},
		input:        []rune("/compact"),
		boundedFrame: true,
		opts: Options{
			Model:     "test",
			Compactor: compactor,
		},
	}

	updated, cmd := m.handleSubmit()
	if cmd == nil {
		t.Fatal("compact command should return a command")
	}
	fm := updated.(chatModel)
	if fm.status != "compacting" {
		t.Fatalf("status = %q, want compacting", fm.status)
	}
	if !fm.compacting {
		t.Fatal("compact command should mark model as compacting")
	}
	if got := fm.activityLabel(); got != "Compacting" {
		t.Fatalf("activityLabel = %q, want Compacting", got)
	}

	updated, _ = fm.Update(nextChatMsg(t, fm.compactEvents))
	fm = updated.(chatModel)
	if got := fm.activityLabel(); got != "Compacting 12 tokens" {
		t.Fatalf("activityLabel = %q, want Compacting 12 tokens", got)
	}

	updated, _ = fm.Update(nextChatMsg(t, fm.compactEvents))
	fm = updated.(chatModel)

	if !compactor.request.Force {
		t.Fatal("manual compaction should be forced")
	}
	if compactor.request.ContinueTask {
		t.Fatal("manual compaction should not add the automatic continue-task instruction")
	}
	if fm.compacting {
		t.Fatal("compacting should be cleared after completion")
	}
	if fm.status != "compacted" {
		t.Fatalf("status = %q, want compacted", fm.status)
	}
	if len(fm.messages) != 3 || fm.messages[0].Role != "user" || fm.messages[2].Role != "tool" || !strings.Contains(fm.messages[2].Content, "old work summary") {
		t.Fatalf("compacted messages = %#v", fm.messages)
	}
	transcript := stripANSI(fm.renderTranscript(100))
	if !strings.Contains(transcript, "Compacted summary") {
		t.Fatalf("summary row should be visible after compacting: %q", transcript)
	}
	if strings.Contains(transcript, "Compacted summary done") {
		t.Fatalf("summary row should not include done word: %q", transcript)
	}
	if strings.Contains(transcript, "old work summary") || strings.Contains(transcript, "Conversation summary:") {
		t.Fatalf("summary body should be collapsed after compacting: %q", transcript)
	}

	updated, _ = fm.Update(tea.KeyMsg{Type: tea.KeyCtrlO})
	fm = updated.(chatModel)
	view := stripANSI(fm.renderTranscript(100))
	if !strings.Contains(view, "old work summary") {
		t.Fatalf("expanded transcript should show compacted summary body: %q", view)
	}
}

func TestChatCompactCommandShowsSkippedReason(t *testing.T) {
	compactor := &chatTestCompactor{
		result: coreagent.CompactionResult{
			Messages: []api.Message{{Role: "user", Content: "only request"}},
			Due:      true,
			Reason:   "compaction is unavailable",
		},
	}
	m := chatModel{
		ctx:      context.Background(),
		messages: []api.Message{{Role: "user", Content: "only request"}},
		input:    []rune("/compact"),
		opts:     Options{Compactor: compactor},
	}

	updated, cmd := m.handleSubmit()
	if cmd == nil {
		t.Fatal("compact command should return a command")
	}
	fm := updated.(chatModel)
	updated, _ = fm.Update(nextChatMsg(t, fm.compactEvents))
	fm = updated.(chatModel)

	if fm.status != "compact skipped" {
		t.Fatalf("status = %q, want compact skipped", fm.status)
	}
	transcript := stripANSI(fm.renderTranscript(100))
	if !strings.Contains(transcript, "compaction is unavailable") || strings.Contains(transcript, "/new") {
		t.Fatalf("skip message = %q", transcript)
	}
}

func TestChatAgentEventsUpdateAssistantEntry(t *testing.T) {
	m := chatModel{}

	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventMessageStarted})
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventThinkingDelta, Thinking: "thinking"})
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventMessageDelta, Content: "done"})

	if len(m.entries) != 1 {
		t.Fatalf("entries = %d, want 1", len(m.entries))
	}
	entry := m.entries[0]
	if entry.role != "assistant" {
		t.Fatalf("role = %q, want assistant", entry.role)
	}
	if entry.detail != "" {
		t.Fatalf("detail = %q, want no rendered thinking", entry.detail)
	}
	if entry.content != "done" {
		t.Fatalf("content = %q, want done", entry.content)
	}
	if strings.Contains(stripANSI(m.renderTranscript(80)), "thinking") {
		t.Fatalf("thinking text should not render in transcript: %q", stripANSI(m.renderTranscript(80)))
	}
}
