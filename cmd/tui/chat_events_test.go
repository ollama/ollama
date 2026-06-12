package tui

import (
	"context"

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

func TestChatRunDoneStartsQueuedMessage(t *testing.T) {
	m := chatModel{
		ctx:     context.Background(),
		running: true,
		queued:  []string{"next prompt"},
		opts: ChatOptions{
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

func TestChatThinkingShowsTokenCount(t *testing.T) {
	m := chatModel{running: true}

	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventThinkingDelta, Thinking: "abcdefgh"})

	if got := m.activityLabel(); got != "thinking 2 tokens" {
		t.Fatalf("activityLabel = %q, want thinking 2 tokens", got)
	}
	if strings.Contains(stripANSI(m.renderTranscript(80)), "abcdefgh") {
		t.Fatalf("thinking text should not render in transcript: %q", stripANSI(m.renderTranscript(80)))
	}

	response := api.ChatResponse{Metrics: api.Metrics{EvalCount: 12}}
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventThinkingDelta, Thinking: "more", Response: &response})
	if got := m.activityLabel(); got != "thinking 12 tokens" {
		t.Fatalf("activityLabel = %q, want thinking 12 tokens", got)
	}

	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventMessageDelta, Content: "done"})
	if m.thinking || m.thinkingTokens != 0 {
		t.Fatalf("thinking state was not cleared: thinking=%v tokens=%d", m.thinking, m.thinkingTokens)
	}
}

func TestChatFooterShowsContextAndCompactionOnlyWhenNear(t *testing.T) {
	m := chatModel{
		width:           120,
		height:          24,
		contextTokens:   50,
		contextEstimate: true,
		opts: ChatOptions{
			Options:             map[string]any{"num_ctx": 100},
			CompactionThreshold: 0.75,
		},
	}

	view := stripANSI(m.View())
	if !strings.Contains(view, "ctx ~50/100 (50%)") {
		t.Fatalf("view missing context pressure: %q", view)
	}
	if strings.Contains(view, "compact at") || strings.Contains(view, "compact due") {
		t.Fatalf("view should hide distant compaction point: %q", view)
	}

	m.contextTokens = 65
	view = stripANSI(m.View())
	if !strings.Contains(view, "ctx ~65/100 (65%)") || !strings.Contains(view, "compact at 75") {
		t.Fatalf("view should show compaction point near threshold: %q", view)
	}

	m.contextTokens = 75
	view = stripANSI(m.View())
	if !strings.Contains(view, "compact due at 75") {
		t.Fatalf("view should show compaction due at threshold: %q", view)
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
		opts: ChatOptions{
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
		opts: ChatOptions{
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
		opts: ChatOptions{
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

func TestChatVerboseRunDoneRendersModelMetrics(t *testing.T) {
	m := chatModel{
		opts:    ChatOptions{Verbose: true},
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
	if got := strings.Count(view, "thinking 7 tokens"); got != 1 {
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
	if got := strings.Count(view, "compacting 42 tokens"); got != 1 {
		t.Fatalf("compacting activity rendered %d times, want 1:\n%s", got, view)
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
	if m.quitArmed || m.quitting {
		t.Fatalf("quit state = armed %v quitting %v, want false", m.quitArmed, m.quitting)
	}
	if m.status != "input cleared" {
		t.Fatalf("status = %q, want input cleared", m.status)
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
	if m.status != "input cleared" {
		t.Fatalf("status = %q, want input cleared", m.status)
	}
}

func TestChatCtrlCRequiresSecondPressToQuit(t *testing.T) {
	m := chatModel{}

	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyCtrlC})
	m = updated.(chatModel)
	if cmd != nil {
		t.Fatal("first idle ctrl+c should only arm quit")
	}
	if !m.quitArmed || m.quitting {
		t.Fatalf("quit state = armed %v quitting %v, want armed only", m.quitArmed, m.quitting)
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
	if m.status != "ready" {
		t.Fatalf("status = %q, want ready", m.status)
	}
	if got := string(m.input); got != "x" {
		t.Fatalf("input = %q, want x", got)
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
		opts: ChatOptions{
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
		opts: ChatOptions{
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
				{Role: "user", Content: "Conversation summary:\nold work summary"},
				{Role: "user", Content: "recent request"},
			},
			Compacted: true,
			Due:       true,
			Summary:   "old work summary",
		},
	}
	m := chatModel{
		ctx:      context.Background(),
		chatID:   "chat-1",
		messages: []api.Message{{Role: "user", Content: "old request"}},
		input:    []rune("/compact"),
		opts: ChatOptions{
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
	if got := fm.activityLabel(); got != "compacting" {
		t.Fatalf("activityLabel = %q, want compacting", got)
	}

	updated, _ = fm.Update(nextChatMsg(t, fm.compactEvents))
	fm = updated.(chatModel)
	if got := fm.activityLabel(); got != "compacting 12 tokens" {
		t.Fatalf("activityLabel = %q, want compacting 12 tokens", got)
	}

	updated, _ = fm.Update(nextChatMsg(t, fm.compactEvents))
	fm = updated.(chatModel)

	if !compactor.request.Force {
		t.Fatal("manual compaction should be forced")
	}
	if fm.compacting {
		t.Fatal("compacting should be cleared after completion")
	}
	if fm.status != "compacted" {
		t.Fatalf("status = %q, want compacted", fm.status)
	}
	if len(fm.messages) != 2 || fm.messages[0].Role != "user" || !strings.Contains(fm.messages[0].Content, "old work summary") {
		t.Fatalf("compacted messages = %#v", fm.messages)
	}
	transcript := stripANSI(fm.renderTranscript(100))
	if !strings.Contains(transcript, "Compacted summary done") {
		t.Fatalf("summary row should be visible after compacting: %q", transcript)
	}
	if strings.Contains(transcript, "old work summary") || strings.Contains(transcript, "Conversation summary:") {
		t.Fatalf("summary body should be collapsed after compacting: %q", transcript)
	}

	updated, _ = fm.Update(tea.KeyMsg{Type: tea.KeyCtrlO})
	fm = updated.(chatModel)
	transcript = stripANSI(fm.renderTranscript(100))
	if !strings.Contains(transcript, "old work summary") {
		t.Fatalf("expanded summary should show body: %q", transcript)
	}
}

func TestChatCompactCommandSuggestsNewWhenSkipped(t *testing.T) {
	compactor := &chatTestCompactor{
		result: coreagent.CompactionResult{
			Messages: []api.Message{{Role: "user", Content: "only request"}},
			Due:      true,
			Reason:   "not enough older messages to compact",
		},
	}
	m := chatModel{
		ctx:      context.Background(),
		messages: []api.Message{{Role: "user", Content: "only request"}},
		input:    []rune("/compact"),
		opts:     ChatOptions{Compactor: compactor},
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
	if !strings.Contains(transcript, "not enough older messages to compact") || !strings.Contains(transcript, "/new") {
		t.Fatalf("skip message should explain /new: %q", transcript)
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
