package chat

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	coreagent "github.com/ollama/ollama/agent"
	"github.com/ollama/ollama/api"
)

func TestChatHelpCommandShowsV1Commands(t *testing.T) {
	m := chatModel{input: []rune("/help")}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("help command should not return a command")
	}

	fm := updated.(chatModel)
	if len(fm.entries) != 1 {
		t.Fatalf("entries = %d, want 1", len(fm.entries))
	}
	for _, want := range []string{
		"**Commands**",
		"- `/model`: switch models",
		"- `/think`: set thinking mode",
		"- `/system [on|off]`: show or set the built-in system prompt",
		"- `/compact`: summarize older context",
		"- `/help`: show commands",
		"- `/bye`: exit",
		"- `/prompt`: show full prompt, tools, and messages",
		"- `/save <filename>`: save request JSON; saved as <filename>.json",
		"**Shortcuts**",
		"- `shift+enter`: insert a newline",
		"- `shift+tab`: toggle permission mode",
	} {
		if !strings.Contains(fm.entries[0].content, want) {
			t.Fatalf("help output missing %q:\n%s", want, fm.entries[0].content)
		}
	}
	for _, removed := range []string{"/history", "/load", "/raw", "/resume", "/set", "/show", "/verbose"} {
		if strings.Contains(fm.entries[0].content, removed) {
			t.Fatalf("removed command %q should stay hidden from help:\n%s", removed, fm.entries[0].content)
		}
	}
}

func TestChatNewCommandRepaintsFromTop(t *testing.T) {
	m := chatModel{
		input:            []rune("/new"),
		flowPrintedLines: 4,
		entries:          []chatEntry{{role: "assistant", content: "old transcript"}},
		messages:         []api.Message{{Role: "user", Content: "old prompt"}},
		approvalState:    testApprovalState(true, map[string]bool{"edit": true}),
		opts:             Options{AllowAllTools: true},
		permissionNotice: "full access enabled",
	}

	updated, cmd := m.handleSubmit()
	if cmd == nil {
		t.Fatal("/new should return a repaint command")
	}
	m = updated.(chatModel)
	if m.status != "new chat" {
		t.Fatalf("status = %q, want new chat", m.status)
	}
	if len(m.entries) != 0 || len(m.messages) != 0 {
		t.Fatalf("chat was not reset: entries=%#v messages=%#v", m.entries, m.messages)
	}
	if m.flowPrintedLines != 0 {
		t.Fatalf("flowPrintedLines = %d, want 0", m.flowPrintedLines)
	}
	if m.approvalState.AllGranted() || m.opts.AllowAllTools || m.approvalState.Allows("edit") || m.permissionNotice != "" {
		t.Fatalf("permissions were not reset: allowAll=%v opts=%v editAllowed=%v notice=%q", m.approvalState.AllGranted(), m.opts.AllowAllTools, m.approvalState.Allows("edit"), m.permissionNotice)
	}
	if msg := cmd(); msg == nil {
		t.Fatal("repaint command returned nil")
	}
}

func TestChatNewCommandPreservesLaunchFullAccessDefault(t *testing.T) {
	m := chatModel{
		input:           []rune("/new"),
		defaultAllowAll: true,
		approvalState:   testApprovalState(false, map[string]bool{"edit": true}),
	}

	updated, _ := m.handleSubmit()
	fm := updated.(chatModel)
	if !fm.approvalState.AllGranted() || !fm.opts.AllowAllTools {
		t.Fatalf("full access default was not restored: allowAll=%v opts=%v", fm.approvalState.AllGranted(), fm.opts.AllowAllTools)
	}
	fm.approvalState.Set(false, nil)
	if fm.approvalState.Allows("edit") {
		t.Fatal("edit scope should be cleared")
	}
}

func TestChatSaveCommandRequiresFilename(t *testing.T) {
	m := chatModel{
		input: []rune("/save"),
		opts:  Options{Model: "llama3.2"},
	}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("/save should not start a command")
	}
	fm := updated.(chatModel)
	if fm.status != "error" {
		t.Fatalf("status = %q, want error", fm.status)
	}
	if len(fm.entries) != 1 || !strings.Contains(fm.entries[0].content, "usage: /save <filename>") {
		t.Fatalf("entries = %#v, want usage error", fm.entries)
	}
}

func TestChatSaveCommandWritesRequestJSON(t *testing.T) {
	dir := t.TempDir()
	m := chatModel{
		input:      []rune("/save request"),
		workingDir: dir,
		opts: Options{
			Model:        "llama3.2",
			SystemPrompt: "You are Ollama.",
		},
		messages: []api.Message{{Role: "user", Content: "hello"}},
	}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("/save redirect should not start a command")
	}
	fm := updated.(chatModel)
	if fm.status != "saved" {
		t.Fatalf("status = %q, want saved", fm.status)
	}

	data, err := os.ReadFile(filepath.Join(dir, "request.json"))
	if err != nil {
		t.Fatal(err)
	}
	raw := string(data)
	for _, want := range []string{
		`"model": "llama3.2"`,
		`"role": "system"`,
		`"content": "You are Ollama."`,
		`"role": "user"`,
		`"content": "hello"`,
	} {
		if !strings.Contains(raw, want) {
			t.Fatalf("saved request missing %q:\n%s", want, raw)
		}
	}
	if got := fm.entries[len(fm.entries)-1].content; got != "saved as request.json" {
		t.Fatalf("save entry = %q, want saved filename", got)
	}
}

func TestChatSaveCommandRejectsPath(t *testing.T) {
	dir := t.TempDir()
	m := chatModel{
		input:      []rune("/save ../request"),
		workingDir: dir,
		opts:       Options{Model: "llama3.2"},
	}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("invalid /save should not start a command")
	}
	fm := updated.(chatModel)
	if fm.status != "error" {
		t.Fatalf("status = %q, want error", fm.status)
	}
	if _, err := os.Stat(filepath.Join(filepath.Dir(dir), "request.json")); !os.IsNotExist(err) {
		t.Fatalf("/save wrote outside working dir, stat err = %v", err)
	}
}

func TestChatPromptCommandOpensPromptDebugScreen(t *testing.T) {
	registry := &coreagent.Registry{}
	registry.Register(chatTestTool{})
	toolArgs := api.NewToolCallFunctionArguments()
	toolArgs.Set("query", "show me everything")
	m := chatModel{
		input:  []rune("/prompt"),
		width:  100,
		height: 20,
		opts: Options{
			Model:               "llama3.2",
			SystemPrompt:        "You are Ollama.",
			Tools:               registry,
			ContextWindowTokens: 1024,
		},
		messages: []api.Message{
			{Role: "user", Content: "hello\nsecond line"},
			{
				Role:     "assistant",
				Content:  "I'll call a tool.",
				Thinking: "I should call the fake tool first.",
				ToolCalls: []api.ToolCall{{
					ID: "call-1",
					Function: api.ToolCallFunction{
						Name:      "fake_tool",
						Arguments: toolArgs,
					},
				}},
			},
			{Role: "tool", ToolName: "fake_tool", ToolCallID: "call-1", Content: "tool result line 1\ntool result line 2"},
		},
	}

	updated, cmd := m.handleSubmit()
	if cmd == nil {
		t.Fatal("/prompt should enable managed prompt screen")
	}
	fm := updated.(chatModel)
	if fm.promptDebug == nil {
		t.Fatal("/prompt should open prompt debug screen")
	}
	if len(fm.entries) != 0 {
		t.Fatalf("/prompt should not append a transcript entry: %#v", fm.entries)
	}
	out := stripANSI(fm.View())
	body := stripANSI(strings.Join(fm.promptDebugLines(160), "\n"))
	for _, want := range []string{
		"Prompt",
		"full request preview",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("/prompt view missing %q:\n%s", want, out)
		}
	}
	for _, want := range []string{
		"model: llama3.2",
		"estimated prompt:",
		"/ 1024 tokens",
		"messages: 4",
		"tools: 1",
		"Tools",
		"1. fake_tool",
		"description:",
		"does test work",
		"parameters: object",
		"1. system",
		"You are Ollama.",
		"2. user",
		"hello",
		"second line",
		"3. assistant",
		"thinking:",
		"I should call the fake tool first.",
		"content:",
		"I'll call a tool.",
		"tool call 1: fake_tool",
		"id: call-1",
		"arguments:",
		"query: show me everything",
		"4. tool:fake_tool",
		"tool_name: fake_tool",
		"tool_call_id: call-1",
		"tool result",
		"tool result line 1",
		"tool result line 2",
	} {
		if !strings.Contains(body, want) {
			t.Fatalf("/prompt output missing %q:\n%s", want, body)
		}
	}
	assistantStart := strings.Index(body, "3. assistant")
	if assistantStart < 0 {
		t.Fatalf("/prompt output missing assistant message:\n%s", body)
	}
	assistantBody := body[assistantStart:]
	thinkingIndex := strings.Index(assistantBody, "thinking:")
	contentIndex := strings.Index(assistantBody, "content:")
	if thinkingIndex < 0 || contentIndex < 0 {
		t.Fatalf("/prompt assistant output missing thinking/content labels:\n%s", body)
	}
	if thinkingIndex > contentIndex {
		t.Fatalf("/prompt assistant thinking should render before content:\n%s", body)
	}
	for _, unwanted := range []string{`"query":`, `"name": "fake_tool"`} {
		if strings.Contains(body, unwanted) {
			t.Fatalf("/prompt output should be rendered, not raw JSON; found %q:\n%s", unwanted, body)
		}
	}
	messagesIndex := strings.Index(body, "Messages")
	toolsIndex := strings.Index(body, "Tools")
	if messagesIndex < 0 || toolsIndex < 0 || toolsIndex < messagesIndex {
		t.Fatalf("/prompt should render tools after messages:\n%s", body)
	}

	updated, cmd = fm.Update(tea.KeyMsg{Type: tea.KeyEsc})
	if cmd == nil {
		t.Fatal("closing prompt debug should disable prompt mouse mode")
	}
	fm = updated.(chatModel)
	if fm.promptDebug != nil {
		t.Fatal("esc should close prompt debug screen")
	}
}

func TestChatPromptDebugCapsToolResultPreview(t *testing.T) {
	longResult := strings.Repeat("x", maxPromptDebugToolResultRunes+25) + "tail-marker"
	m := chatModel{
		width:  120,
		height: 20,
		promptDebug: &chatPromptDebug{
			request: api.ChatRequest{
				Model: "llama3.2",
				Messages: []api.Message{{
					Role:       "tool",
					ToolName:   "bash",
					ToolCallID: "call-1",
					Content:    longResult,
				}},
			},
		},
	}

	body := stripANSI(strings.Join(m.promptDebugLines(120), "\n"))
	if strings.Contains(body, "tail-marker") {
		t.Fatalf("/prompt should cap rendered tool results:\n%s", body)
	}
	if !strings.Contains(body, "...") {
		t.Fatalf("/prompt capped tool result should show ellipsis:\n%s", body)
	}
	if got := strings.Count(body, "x"); got != maxPromptDebugToolResultRunes-3 {
		t.Fatalf("rendered tool result x count = %d, want %d:\n%s", got, maxPromptDebugToolResultRunes-3, body)
	}
	if got := m.promptDebug.request.Messages[0].Content; got != longResult {
		t.Fatal("/prompt rendering should not mutate the request")
	}
}

func TestChatPromptDebugMouseWheelScrolls(t *testing.T) {
	m := chatModel{
		width:  80,
		height: 8,
		opts:   Options{Model: "llama3.2", ContextWindowTokens: 1024},
		promptDebug: &chatPromptDebug{
			request: api.ChatRequest{Model: "llama3.2"},
			tokens:  10,
		},
	}
	for range 30 {
		m.promptDebug.request.Messages = append(m.promptDebug.request.Messages, api.Message{Role: "user", Content: "line"})
	}
	if m.promptDebugMaxScroll() == 0 {
		t.Fatal("test setup should produce a scrollable prompt debug screen")
	}

	updated, _ := m.Update(tea.MouseMsg{Type: tea.MouseWheelDown})
	m = updated.(chatModel)
	if m.promptDebug.scroll == 0 {
		t.Fatal("mouse wheel down should scroll prompt debug screen")
	}

	updated, _ = m.Update(tea.MouseMsg{Type: tea.MouseWheelUp})
	m = updated.(chatModel)
	if m.promptDebug.scroll != 0 {
		t.Fatalf("mouse wheel up should return prompt debug screen to top, got scroll %d", m.promptDebug.scroll)
	}
}

func TestTruncateInputLineUsesDisplayWidth(t *testing.T) {
	line := truncateInputLine(strings.Repeat("界", 10), 10)
	if got := lipgloss.Width(line); got > 10 {
		t.Fatalf("line %q width = %d, want <= 10", line, got)
	}
}

func TestRenderInputBoxTruncationUsesSingleContinuationMarker(t *testing.T) {
	lines := renderInputBoxLines("one two three four five six seven", len("one two three four five six seven"), 16, 1, "")
	rendered := strings.Join(lines, "\n")
	if strings.Contains(rendered, "... ...") {
		t.Fatalf("input rendered duplicate continuation marker: %q", rendered)
	}
	if strings.Contains(rendered, "one two") {
		t.Fatalf("input should keep the latest truncated line: %q", rendered)
	}
}

type shiftEnterCSITestMsg string

func (m shiftEnterCSITestMsg) String() string {
	return string(m)
}

func TestChatInputHandlesShiftEnterCSIMessage(t *testing.T) {
	m := chatModel{input: []rune("line one")}

	updated, _ := m.Update(shiftEnterCSITestMsg("?CSI[49 51 59 50 117]?"))
	m = updated.(chatModel)
	if got := string(m.input); got != "line one\n" {
		t.Fatalf("input = %q, want newline inserted", got)
	}
}

func TestChatInputAcceptsSpace(t *testing.T) {
	m := chatModel{}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("hello")})
	m = updated.(chatModel)
	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeySpace, Runes: []rune(" ")})
	m = updated.(chatModel)
	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("world")})
	m = updated.(chatModel)

	if got := string(m.input); got != "hello world" {
		t.Fatalf("input = %q, want hello world", got)
	}
}

func TestChatCloudModelDefaultToolRoundsAreUnlimited(t *testing.T) {
	const formerDefaultLimit = 100
	client := &chatToolLoopClient{toolRounds: formerDefaultLimit + 1}
	registry := &coreagent.Registry{}
	registry.Register(chatTestTool{})
	m := chatModel{
		ctx: context.Background(),
		opts: Options{
			Model:         "test:cloud",
			Client:        client,
			Tools:         registry,
			AllowAllTools: true,
		},
	}

	updated, cmd := m.startRun("keep going")
	m = updated.(chatModel)
	if cmd == nil {
		t.Fatal("startRun should start a cloud model run")
	}
	done := waitForRunDone(t, m.events)
	if done.err != nil {
		t.Fatalf("cloud run returned error: %v", done.err)
	}
	if client.calls != formerDefaultLimit+2 {
		t.Fatalf("client calls = %d, want %d", client.calls, formerDefaultLimit+2)
	}
}

func TestChatLargePasteUsesPlaceholderAndExpandsOnSubmit(t *testing.T) {
	pasted := strings.Repeat("line\n", pastedTextPlaceholderMinLines-1) + "line"
	m := chatModel{
		ctx: context.Background(),
		opts: Options{
			Model:  "test",
			Client: chatTestClient{},
		},
	}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(pasted), Paste: true})
	m = updated.(chatModel)
	if got, want := string(m.input), "[Pasted text #1 +8 lines]"; got != want {
		t.Fatalf("input = %q, want %q", got, want)
	}

	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = updated.(chatModel)
	if cmd == nil {
		t.Fatal("enter should start a run")
	}
	done := waitForRunDone(t, m.events)
	if done.err != nil {
		t.Fatal(done.err)
	}
	if done.result == nil || len(done.result.Messages) < 1 || done.result.Messages[0].Content != pasted {
		t.Fatalf("messages = %#v, want expanded pasted text", done.result)
	}
}

func TestChatBackspaceDeletesWholePastedTextPlaceholder(t *testing.T) {
	m := chatModel{
		input: []rune("use [Pasted text #1 +8 lines]"),
		inputPastedTexts: []chatInputPastedText{{
			placeholder: "[Pasted text #1 +8 lines]",
			content:     "hidden",
		}},
	}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyBackspace})
	m = updated.(chatModel)
	if got := string(m.input); got != "use " {
		t.Fatalf("input after backspace = %q, want pasted text placeholder removed", got)
	}
	if got := len(m.inputPastedTexts); got != 0 {
		t.Fatalf("pasted texts after backspace = %d, want 0", got)
	}
}

func TestInitialPromptHistoryLoadsFromMessages(t *testing.T) {
	history := initialPromptHistory(context.Background(), Options{
		Messages: []api.Message{
			{Role: "user", Content: "old prompt"},
			{Role: "assistant", Content: "answer"},
			{Role: "user", Content: "new prompt"},
		},
	})

	if got, want := strings.Join(history, "|"), "old prompt|new prompt"; got != want {
		t.Fatalf("history = %#v, want %s", history, want)
	}
}

func TestSkillCommandsListAndPersistSyntheticToolCall(t *testing.T) {
	dir := t.TempDir()
	skillDir := filepath.Join(dir, "release-notes")
	if err := os.MkdirAll(skillDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(skillDir, "SKILL.md"), []byte("---\nname: release-notes\ndescription: Draft release notes.\n---\nUse concise bullets."), 0o644); err != nil {
		t.Fatal(err)
	}
	catalog, err := coreagent.DiscoverSkills(dir)
	if err != nil {
		t.Fatal(err)
	}

	m := chatModel{opts: Options{Skills: catalog}, input: []rune("/skills")}
	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("/skills should not start a model run")
	}
	m = updated.(chatModel)
	if len(m.entries) != 1 || !strings.Contains(m.entries[0].content, "release-notes") {
		t.Fatalf("/skills entries = %#v", m.entries)
	}

	m = chatModel{ctx: context.Background(), opts: Options{Model: "test", Skills: catalog, Client: chatTestClient{}}, input: []rune("/release-notes")}
	updated, cmd = m.handleSubmit()
	if cmd == nil {
		t.Fatal("/<skill-name> should continue the chat with the loaded instructions")
	}
	m = updated.(chatModel)
	events := m.events
	for {
		msg, ok := <-events
		if !ok {
			t.Fatal("skill run closed before it finished")
		}
		updated, _ = m.Update(msg)
		m = updated.(chatModel)
		if _, ok := msg.(chatRunDoneMsg); ok {
			break
		}
	}
	if len(m.messages) != 4 {
		t.Fatalf("synthetic messages = %#v", m.messages)
	}
	call := m.messages[1]
	result := m.messages[2]
	if call.Role != "assistant" || len(call.ToolCalls) != 1 || call.ToolCalls[0].Function.Name != "skill" || !strings.HasPrefix(call.ToolCalls[0].ID, "call_skill_") {
		t.Fatalf("synthetic call = %#v", call)
	}
	if result.Role != "tool" || result.ToolCallID != call.ToolCalls[0].ID || !strings.Contains(result.Content, "Use concise bullets.") {
		t.Fatalf("synthetic result = %#v", result)
	}
	entries := entriesFromMessages(m.messages)
	if len(entries) != 3 || entries[1].toolID != call.ToolCalls[0].ID || entries[1].detail != "skill" || entries[1].args["name"] != "release-notes" {
		t.Fatalf("round-trip entries = %#v", entries)
	}
}

func TestSkillsImportReloadsCatalogRegistryAndSystemPrompt(t *testing.T) {
	before := writeTestSkillCatalog(t)
	dir := t.TempDir()
	if err := os.Mkdir(filepath.Join(dir, "from-codex"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "from-codex", "SKILL.md"), []byte("---\nname: from-codex\ndescription: Imported skill.\n---\nImported instructions."), 0o644); err != nil {
		t.Fatal(err)
	}
	after, err := coreagent.DiscoverSkills(dir)
	if err != nil {
		t.Fatal(err)
	}
	registry := &coreagent.Registry{}
	var reloaded, rebuilt, prompted bool
	m := chatModel{
		ctx: context.Background(),
		opts: Options{
			Model:  "test",
			Skills: before,
			ImportSkills: func(source string) (coreagent.SkillImportResult, error) {
				if source != "codex" {
					t.Fatalf("source = %q", source)
				}
				return coreagent.SkillImportResult{Source: source, SourceDir: "/source", Imported: []string{"from-codex"}}, nil
			},
			ReloadSkills: func() (*coreagent.SkillCatalog, error) {
				reloaded = true
				return after, nil
			},
			ToolRegistryForModel: func(context.Context, string) *coreagent.Registry {
				rebuilt = true
				return registry
			},
			SystemPromptForModel: func(_ context.Context, _ string, got *coreagent.Registry, _ bool) string {
				prompted = got == registry
				return after.SystemContext()
			},
		},
		input: []rune("/skills import codex"),
	}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("skills import should not start a model run")
	}
	m = updated.(chatModel)
	if !reloaded || !rebuilt || !prompted {
		t.Fatalf("reload=%v rebuilt=%v prompted=%v", reloaded, rebuilt, prompted)
	}
	if m.opts.Skills != after || m.opts.Tools != registry || !strings.Contains(m.opts.SystemPrompt, "from-codex") {
		t.Fatalf("reloaded options = %#v", m.opts)
	}
	if m.status != "skills reloaded" || len(m.entries) != 1 || !strings.Contains(m.entries[0].content, "Imported 1 skill") {
		t.Fatalf("import result = status %q entries %#v", m.status, m.entries)
	}
}

func TestSkillsImportUsage(t *testing.T) {
	m := chatModel{input: []rune("/skills import")}
	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("invalid skills import should not start a model run")
	}
	m = updated.(chatModel)
	if len(m.entries) != 1 || m.entries[0].role != "error" || !strings.Contains(m.entries[0].content, "usage: /skills [import codex|claude|pi]") {
		t.Fatalf("entries = %#v", m.entries)
	}
}

func TestSkillSlashCommandPromptBecomesUserMessage(t *testing.T) {
	catalog := writeTestSkillCatalog(t)
	m := chatModel{ctx: context.Background(), opts: Options{Model: "test", Skills: catalog, Client: chatTestClient{}}, input: []rune("/release-notes draft the v1.2 notes")}
	updated, cmd := m.handleSubmit()
	if cmd == nil {
		t.Fatal("/<skill-name> <prompt> should start a run")
	}
	m = updated.(chatModel)
	for {
		msg, ok := <-m.events
		if !ok {
			t.Fatal("skill run closed before it finished")
		}
		updated, _ = m.Update(msg)
		m = updated.(chatModel)
		if _, ok := msg.(chatRunDoneMsg); ok {
			break
		}
	}
	if len(m.messages) < 1 || m.messages[0].Role != "user" || m.messages[0].Content != "draft the v1.2 notes" {
		t.Fatalf("user message = %#v, want the prompt", m.messages[0])
	}
	// The skill still loads as a synthetic tool call right after the user turn.
	if len(m.messages) < 3 || m.messages[1].Role != "assistant" || len(m.messages[1].ToolCalls) != 1 || m.messages[1].ToolCalls[0].Function.Name != "skill" {
		t.Fatalf("synthetic skill call missing: %#v", m.messages)
	}
}

func TestChatSkillSubmitWhileActiveRunKeepsActiveState(t *testing.T) {
	catalog := writeTestSkillCatalog(t)
	for _, state := range []struct {
		name       string
		running    bool
		compacting bool
	}{
		{name: "running", running: true},
		{name: "compacting", compacting: true},
	} {
		t.Run(state.name, func(t *testing.T) {
			events := make(chan tea.Msg)
			cancel := func() {}
			m := chatModel{
				opts:       Options{Skills: catalog},
				input:      []rune("/release-notes draft notes"),
				running:    state.running,
				compacting: state.compacting,
				events:     events,
				cancel:     cancel,
			}

			updated, cmd := m.handleSubmit()
			if cmd != nil {
				t.Fatal("skill submit should not start another run while active")
			}
			got := updated.(chatModel)
			if got.events != events || got.cancel == nil || got.running != state.running || got.compacting != state.compacting {
				t.Fatalf("active run state changed: %#v", got)
			}
			if string(got.input) != "/release-notes draft notes" {
				t.Fatalf("input = %q, want skill invocation preserved", got.input)
			}
			if got.status != "wait for current response" {
				t.Fatalf("status = %q", got.status)
			}
		})
	}
}

func writeTestSkillCatalog(t *testing.T) *coreagent.SkillCatalog {
	t.Helper()
	dir := t.TempDir()
	skillDir := filepath.Join(dir, "release-notes")
	if err := os.MkdirAll(skillDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(skillDir, "SKILL.md"), []byte("---\nname: release-notes\ndescription: Draft release notes.\n---\nUse concise bullets."), 0o644); err != nil {
		t.Fatal(err)
	}
	catalog, err := coreagent.DiscoverSkills(dir)
	if err != nil {
		t.Fatal(err)
	}
	return catalog
}

func TestSkillSlashCommandAppearsInCompletions(t *testing.T) {
	catalog := writeTestSkillCatalog(t)
	m := chatModel{opts: Options{Skills: catalog}, input: []rune("/re")}

	lines := stripANSI(strings.Join(m.slashCommandLines(80), "\n"))
	if !strings.Contains(lines, "/release-notes") || !strings.Contains(lines, "Draft release notes.") {
		t.Fatalf("suggestions missing /release-notes: %q", lines)
	}
}

func TestSkillsImportSlashCompletions(t *testing.T) {
	for _, test := range []struct {
		input string
		want  []string
	}{
		{input: "/skills", want: []string{"/skills", "/skills import"}},
		{input: "/skills impo", want: []string{"/skills import"}},
		{input: "/skills import ", want: []string{"/skills import codex", "/skills import claude", "/skills import pi"}},
		{input: "/skills import c", want: []string{"/skills import codex", "/skills import claude"}},
		{input: "/skills import pi", want: []string{"/skills import pi"}},
	} {
		t.Run(test.input, func(t *testing.T) {
			m := chatModel{input: []rune(test.input)}
			completions := m.slashCompletions()
			got := make([]string, 0, len(completions))
			for _, completion := range completions {
				got = append(got, completion.value)
			}
			if strings.Join(got, "\n") != strings.Join(test.want, "\n") {
				t.Fatalf("completions = %#v, want %#v", got, test.want)
			}
		})
	}
}

func TestSkillSlashPromptHidesCommandCompletions(t *testing.T) {
	catalog := writeTestSkillCatalog(t)
	for _, input := range []string{"/release-notes ", "/release-notes draft the release notes"} {
		t.Run(input, func(t *testing.T) {
			m := chatModel{opts: Options{Skills: catalog}, input: []rune(input)}
			if lines := m.completionLines(80); len(lines) != 0 {
				t.Fatalf("completion lines = %#v, want none", lines)
			}
		})
	}
}

func TestSkillSlashNameResolvesAndRejectsArgsAndUnknown(t *testing.T) {
	catalog := writeTestSkillCatalog(t)
	m := &chatModel{opts: Options{Skills: catalog}}

	if name, _, ok := m.skillSlashInvocation("/release-notes"); !ok || name != "release-notes" {
		t.Fatalf("/release-notes = %q %v, want release-notes true", name, ok)
	}
	if name, prompt, ok := m.skillSlashInvocation("/release-notes draft notes"); !ok || name != "release-notes" || prompt != "draft notes" {
		t.Fatalf("/release-notes draft notes = %q %q %v, want release-notes / draft notes / true", name, prompt, ok)
	}
	if _, _, ok := m.skillSlashInvocation("/no-such-skill"); ok {
		t.Fatal("unknown skill should not resolve")
	}
	// A built-in command sharing a prefix must not be claimed as a skill.
	if _, _, ok := m.skillSlashInvocation("/skills"); ok {
		t.Fatal("/skills should resolve to the built-in, not a skill")
	}

	// Unknown slash input that is not a skill stays an unknown command.
	m2 := chatModel{opts: Options{Skills: catalog}, input: []rune("/no-such-skill")}
	updated, cmd := m2.handleSubmit()
	if cmd != nil {
		t.Fatal("unknown slash command should not start a run")
	}
	m2 = updated.(chatModel)
	if len(m2.entries) != 1 || m2.entries[0].role != "error" || !strings.Contains(m2.entries[0].content, "Unknown command") {
		t.Fatalf("entries = %#v, want unknown command", m2.entries)
	}
}

func TestChatDeletedSlashCommandsAreUnknown(t *testing.T) {
	for _, command := range []string{"/clear", "/copy", "/copy-all", "/launch", "/history", "/load", "/raw", "/resume", "/set", "/show", "/verbose"} {
		t.Run(command, func(t *testing.T) {
			m := chatModel{input: []rune(command)}

			updated, cmd := m.handleSubmit()
			if cmd != nil {
				t.Fatal("deleted slash command should not return a command")
			}
			m = updated.(chatModel)
			if len(m.entries) != 1 || m.entries[0].role != "error" || !strings.Contains(m.entries[0].content, "Unknown command") {
				t.Fatalf("entries = %#v, want unknown command error", m.entries)
			}
		})
	}
}

func TestChatViewRendersSlashCommandSuggestions(t *testing.T) {
	m := chatModel{
		input:  []rune("/"),
		width:  80,
		height: 18,
	}

	view := stripANSI(m.View())
	for _, want := range []string{"/model", "/new", "/think", "/tools", "/system"} {
		if !strings.Contains(view, want) {
			t.Fatalf("view missing %s suggestion: %q", want, view)
		}
	}
	for _, removed := range []string{"/clear", "/copy", "/copy-all", "/history", "/load", "/raw", "/resume", "/set", "/show", "/verbose"} {
		if strings.Contains(view, removed) {
			t.Fatalf("bare slash should hide removed command %s: %q", removed, view)
		}
	}
	if got := len(m.slashCommandLines(80)); got != maxSlashCompletions {
		t.Fatalf("slash suggestions = %d, want %d", got, maxSlashCompletions)
	}
}

func TestChatToolsCommandTogglesToolRegistry(t *testing.T) {
	registry := &coreagent.Registry{}
	registry.Register(chatTestTool{})
	calls := 0
	m := chatModel{
		ctx:   context.Background(),
		input: []rune("/tools"),
		opts: Options{
			Model: "llama3.2",
			Tools: registry,
			ToolRegistryForModel: func(context.Context, string) *coreagent.Registry {
				calls++
				return registry
			},
		},
	}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("/tools should not start a command")
	}
	m = updated.(chatModel)
	if !m.opts.ToolsDisabled || m.opts.Tools == nil {
		t.Fatalf("tools off state = disabled:%v tools:%#v", m.opts.ToolsDisabled, m.opts.Tools)
	}
	if len(m.entries) != 0 {
		t.Fatalf("/tools should only update action space, got entries:%#v", m.entries)
	}
	req, _ := m.requestPreview()
	if got := len(req.Tools); got != 0 {
		t.Fatalf("request preview tools = %d, want 0", got)
	}

	m.input = []rune("/tools")
	updated, cmd = m.handleSubmit()
	if cmd != nil {
		t.Fatal("/tools should not start a command")
	}
	m = updated.(chatModel)
	if m.opts.ToolsDisabled || m.opts.Tools == nil {
		t.Fatalf("tools on state = disabled:%v tools:%#v", m.opts.ToolsDisabled, m.opts.Tools)
	}
	if calls != 1 {
		t.Fatalf("tool registry calls = %d, want 1", calls)
	}
	if len(m.entries) != 0 {
		t.Fatalf("/tools should only update action space, got entries:%#v", m.entries)
	}
	req, _ = m.requestPreview()
	if got := len(req.Tools); got != 1 {
		t.Fatalf("request preview tools = %d, want 1", got)
	}
}

func TestChatToolsCommandRefreshesCapabilityAwareSystemPrompt(t *testing.T) {
	registry := &coreagent.Registry{}
	registry.Register(chatTestTool{})
	m := chatModel{
		ctx: context.Background(),
		opts: Options{
			Model: "test",
			Tools: registry,
			SystemPromptForModel: func(_ context.Context, _ string, _ *coreagent.Registry, disabled bool) string {
				if disabled {
					return "tools disabled"
				}
				return "tools enabled"
			},
		},
	}

	updated, _ := m.handleToolsCommand("")
	m = updated.(chatModel)
	if m.opts.SystemPrompt != "tools disabled" {
		t.Fatalf("system prompt = %q, want disabled prompt", m.opts.SystemPrompt)
	}
	updated, _ = m.handleToolsCommand("")
	m = updated.(chatModel)
	if m.opts.SystemPrompt != "tools enabled" {
		t.Fatalf("system prompt = %q, want enabled prompt", m.opts.SystemPrompt)
	}
}

func TestChatToolsCommandUsage(t *testing.T) {
	m := chatModel{input: []rune("/tools off")}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("invalid /tools should not start a command")
	}
	m = updated.(chatModel)
	if m.status != "error" || len(m.entries) != 1 || !strings.Contains(m.entries[0].content, "usage: /tools") {
		t.Fatalf("invalid /tools result = status:%q entries:%#v", m.status, m.entries)
	}
}

func TestChatSystemCommandControlsBuiltInSystemPrompt(t *testing.T) {
	client := &chatCaptureClient{}
	m := chatModel{
		ctx:   context.Background(),
		input: []rune("/system"),
		opts: Options{
			Model:        "test",
			Client:       client,
			SystemPrompt: "canonical agent prompt",
		},
	}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("/system should not start a run")
	}
	m = updated.(chatModel)
	if len(m.entries) != 1 || m.entries[0].role != "slash" || m.entries[0].content != "Built-in system prompt is on.\n\ncanonical agent prompt\n\nWarning: Changing the system prompt during a session breaks the prompt cache." {
		t.Fatalf("/system entry = %#v", m.entries)
	}

	m.input = []rune("/system off")
	updated, _ = m.handleSubmit()
	m = updated.(chatModel)
	if !m.systemPromptDisabled || m.status != "system prompt off" {
		t.Fatalf("/system off state = disabled:%v status:%q", m.systemPromptDisabled, m.status)
	}
	m.input = []rune("/system")
	updated, _ = m.handleSubmit()
	m = updated.(chatModel)
	if got := m.entries[len(m.entries)-1].content; got != "Built-in system prompt is off.\n\ncanonical agent prompt\n\nWarning: Changing the system prompt during a session breaks the prompt cache." {
		t.Fatalf("/system off entry = %q", got)
	}
	updated, cmd = m.startRun("hello")
	if cmd == nil {
		t.Fatal("run after /system off should start")
	}
	m = updated.(chatModel)
	if done := waitForRunDone(t, m.events); done.err != nil {
		t.Fatalf("run after /system off: %v", done.err)
	}
	if len(client.requests) != 1 || len(client.requests[0].Messages) != 1 || client.requests[0].Messages[0].Role != "user" {
		t.Fatalf("request after /system off = %#v", client.requests)
	}

	m.input = []rune("/system ON")
	updated, _ = m.handleSubmit()
	m = updated.(chatModel)
	if m.systemPromptDisabled || m.status != "system prompt on" {
		t.Fatalf("/system on state = disabled:%v status:%q", m.systemPromptDisabled, m.status)
	}
	updated, cmd = m.startRun("hello again")
	if cmd == nil {
		t.Fatal("run after /system on should start")
	}
	m = updated.(chatModel)
	if done := waitForRunDone(t, m.events); done.err != nil {
		t.Fatalf("run after /system on: %v", done.err)
	}
	if len(client.requests) != 2 {
		t.Fatalf("client requests = %d, want 2", len(client.requests))
	}
	request := client.requests[1]
	if len(request.Messages) != 2 || request.Messages[0].Role != "system" || request.Messages[0].Content != "canonical agent prompt" {
		t.Fatalf("request after /system on = %#v", request.Messages)
	}

	m.input = []rune("/system sometimes")
	updated, _ = m.handleSubmit()
	m = updated.(chatModel)
	if m.status != "error" || len(m.entries) == 0 || m.entries[len(m.entries)-1].content != "usage: /system [on|off]" {
		t.Fatalf("invalid /system result = status:%q entries:%#v", m.status, m.entries)
	}
}

func TestChatSystemCommandArgumentCompletions(t *testing.T) {
	for _, tt := range []struct {
		input string
		want  []string
	}{
		{input: "/system ", want: []string{"/system on", "/system off"}},
		{input: "/system o", want: []string{"/system on", "/system off"}},
		{input: "/system on", want: []string{"/system on"}},
	} {
		t.Run(tt.input, func(t *testing.T) {
			m := chatModel{input: []rune(tt.input)}
			completions := m.slashCompletions()
			if len(completions) != len(tt.want) {
				t.Fatalf("completions = %#v, want %d", completions, len(tt.want))
			}
			for i, want := range tt.want {
				if completions[i].value != want {
					t.Fatalf("completion %d = %q, want %q", i, completions[i].value, want)
				}
			}
		})
	}

	m := chatModel{input: []rune("/system ")}
	lines := stripANSI(strings.Join(m.slashCommandLines(80), "\n"))
	for _, want := range []string{"on", "enable the built-in system prompt", "off", "disable the built-in system prompt"} {
		if !strings.Contains(lines, want) {
			t.Fatalf("/system option suggestions missing %q: %q", want, lines)
		}
	}
	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	if cmd != nil {
		t.Fatal("selecting /system on should not submit the command")
	}
	m = updated.(chatModel)
	if got := string(m.input); got != "/system on" {
		t.Fatalf("input = %q, want /system on", got)
	}

	m.input = []rune("/system maybe")
	completions := m.slashCompletions()
	if len(completions) != 1 || completions[0].label != "No matching options" {
		t.Fatalf("invalid argument completions = %#v", completions)
	}
}

func TestChatSlashCommandSuggestionsIncludePromptAndSave(t *testing.T) {
	for _, tt := range []struct {
		input       string
		command     string
		description string
	}{
		{input: "/pr", command: "/prompt", description: "show full prompt, tools, and messages"},
		{input: "/sa", command: "/save", description: "save request JSON; saved as <filename>.json"},
	} {
		t.Run(tt.command, func(t *testing.T) {
			m := chatModel{input: []rune(tt.input)}

			lines := stripANSI(strings.Join(m.slashCommandLines(80), "\n"))
			if !strings.Contains(lines, tt.command) || !strings.Contains(lines, tt.description) {
				t.Fatalf("suggestions missing %s: %q", tt.command, lines)
			}
		})
	}
}

func TestChatSlashCommandSuggestionsIncludeThink(t *testing.T) {
	m := chatModel{input: []rune("/th")}

	lines := stripANSI(strings.Join(m.slashCommandLines(80), "\n"))
	if !strings.Contains(lines, "/think") || !strings.Contains(lines, "set thinking mode") {
		t.Fatalf("suggestions missing /think: %q", lines)
	}
}

func TestChatEnterFillsSelectedSlashCommandBeforeSubmitting(t *testing.T) {
	m := chatModel{input: []rune("/th")}

	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = updated.(chatModel)
	if cmd != nil {
		t.Fatal("filling a slash command should not return a command")
	}
	if got := string(m.input); got != "/think" {
		t.Fatalf("input = %q, want completed command", got)
	}
	if m.thinkPicker != nil {
		t.Fatal("filling a slash command should not open its picker")
	}

	updated, cmd = m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = updated.(chatModel)
	if cmd != nil {
		t.Fatal("think command should not return a command")
	}
	if m.thinkPicker == nil {
		t.Fatal("second enter should submit the completed /think command")
	}
}

func TestChatEnterSubmitsExactSlashCommandAliases(t *testing.T) {
	t.Run("help", func(t *testing.T) {
		m := chatModel{input: []rune("/?")}

		updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
		if cmd != nil {
			t.Fatal("help alias should not return a command")
		}
		m = updated.(chatModel)
		if len(m.entries) != 1 || m.entries[0].role != "slash" {
			t.Fatalf("entries = %#v, want help output", m.entries)
		}
		if got := string(m.input); got != "" {
			t.Fatalf("input = %q, want cleared after submitting alias", got)
		}
	})

	t.Run("exit", func(t *testing.T) {
		m := chatModel{input: []rune("/exit")}

		updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
		if cmd == nil {
			t.Fatal("exit alias should return the quit command")
		}
		m = updated.(chatModel)
		if !m.quitting {
			t.Fatal("exit alias should quit without filling /bye first")
		}
		if got := string(m.input); got != "" {
			t.Fatalf("input = %q, want cleared after submitting alias", got)
		}
	})
}

func TestChatSlashCommandsRunWhileModelResponds(t *testing.T) {
	m := chatModel{running: true, input: []rune("/help")}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("help command should not return a command")
	}
	m = updated.(chatModel)
	if len(m.entries) != 1 || m.entries[0].role != "slash" {
		t.Fatalf("entries = %#v, want immediate slash output", m.entries)
	}
}

func TestChatMessageSubmitWhileRunningPreservesDraft(t *testing.T) {
	attachment := chatInputAttachment{placeholder: "[image 1]", kind: "image"}
	pasted := chatInputPastedText{placeholder: "[pasted 1]", content: "long paste"}
	m := chatModel{
		running:          true,
		input:            []rune("next prompt [image 1] [pasted 1]"),
		inputCursor:      4,
		inputCursorSet:   true,
		inputAttachments: []chatInputAttachment{attachment},
		inputPastedTexts: []chatInputPastedText{pasted},
	}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("busy submit should not start a command")
	}
	m = updated.(chatModel)
	if got := string(m.input); got != "next prompt [image 1] [pasted 1]" {
		t.Fatalf("input = %q, want draft preserved", got)
	}
	if m.inputCursor != 4 || !m.inputCursorSet {
		t.Fatalf("cursor not preserved: cursor=%d set=%v", m.inputCursor, m.inputCursorSet)
	}
	if len(m.inputAttachments) != 1 || m.inputAttachments[0].placeholder != attachment.placeholder || m.inputAttachments[0].kind != attachment.kind {
		t.Fatalf("attachments not preserved: %#v", m.inputAttachments)
	}
	if len(m.inputPastedTexts) != 1 || m.inputPastedTexts[0] != pasted {
		t.Fatalf("pasted text not preserved: %#v", m.inputPastedTexts)
	}
	if len(m.entries) != 0 {
		t.Fatalf("busy submit should not add entries: %#v", m.entries)
	}
}

func TestChatThinkCommandOpensPicker(t *testing.T) {
	m := chatModel{input: []rune("/think")}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("think command should not return a command")
	}
	m = updated.(chatModel)
	if m.thinkPicker == nil {
		t.Fatal("think picker should open")
	}
	if view := stripANSI(m.renderThinkPicker(80)); !strings.Contains(view, "Thinking mode") || !strings.Contains(view, "high") {
		t.Fatalf("think picker view missing options: %q", view)
	}
}

func TestChatThinkCommandSetsModes(t *testing.T) {
	for _, tt := range []struct {
		input string
		want  any
	}{
		{input: "/think on", want: true},
		{input: "/think off", want: false},
		{input: "/think high", want: "high"},
	} {
		t.Run(tt.input, func(t *testing.T) {
			m := chatModel{input: []rune(tt.input)}
			updated, cmd := m.handleSubmit()
			if cmd != nil {
				t.Fatal("think command should not return a command")
			}
			m = updated.(chatModel)
			if m.opts.Think == nil || m.opts.Think.Value != tt.want {
				t.Fatalf("think = %#v, want %#v", m.opts.Think, tt.want)
			}
		})
	}
}

func TestChatViewRendersFileMentionSuggestions(t *testing.T) {
	dir := t.TempDir()
	if err := os.Mkdir(filepath.Join(dir, "cmd"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "README.md"), []byte("hi"), 0o644); err != nil {
		t.Fatal(err)
	}
	m := chatModel{
		workingDir: dir,
		input:      []rune("open @"),
		width:      80,
		height:     18,
	}

	view := stripANSI(m.View())
	if !strings.Contains(view, "@cmd/") || !strings.Contains(view, "@README.md") {
		t.Fatalf("file mention suggestions missing: %q", view)
	}
}

func TestChatFileMentionSuggestionsFilterAndComplete(t *testing.T) {
	dir := t.TempDir()
	if err := os.Mkdir(filepath.Join(dir, "cmd"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "README.md"), []byte("hi"), 0o644); err != nil {
		t.Fatal(err)
	}
	m := chatModel{
		workingDir: dir,
		input:      []rune("open @REA"),
	}

	lines := stripANSI(strings.Join(m.completionLines(80), "\n"))
	if !strings.Contains(lines, "@README.md") || strings.Contains(lines, "@cmd/") {
		t.Fatalf("filtered file suggestions = %q", lines)
	}
	m.applyCompletion()
	if got := string(m.input); got != "open @README.md " {
		t.Fatalf("completed input = %q", got)
	}
}
