package chat

import (
	"context"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	coreagent "github.com/ollama/ollama/agent"
)

func testApprovalRequest() coreagent.ApprovalRequest {
	return coreagent.ApprovalRequest{
		WorkingDir: "/repo",
		Calls: []coreagent.ApprovalToolCall{{
			ToolCallID:    "call-1",
			ToolName:      "edit",
			Args:          map[string]any{"path": "note.txt"},
			ApprovalScope: "edit",
		}},
	}
}

func testApprovalState(allowAll bool, scopes map[string]bool) *coreagent.ApprovalState {
	state := &coreagent.ApprovalState{}
	state.Set(allowAll, scopes)
	return state
}

func TestChatApprovalApprovesOnce(t *testing.T) {
	reply := make(chan coreagent.Approval, 1)
	m := chatModel{
		approvalPrompt: &chatApprovalPrompt{
			request: testApprovalRequest(),
			reply:   reply,
		},
		events: make(chan tea.Msg),
	}

	updated, cmd := m.updateApprovalPrompt(tea.KeyMsg{Type: tea.KeyEnter})
	if cmd == nil {
		t.Fatal("approval should resume waiting for agent events")
	}
	fm := updated.(chatModel)
	if fm.approvalPrompt != nil {
		t.Fatal("approval prompt should close")
	}
	result := <-reply
	if !result.Allow || result.AllowAll {
		t.Fatalf("approval = %#v, want allow once", result)
	}
}

func TestChatApprovalAllowsTool(t *testing.T) {
	reply := make(chan coreagent.Approval, 1)
	m := chatModel{
		approvalPrompt: &chatApprovalPrompt{
			request: testApprovalRequest(),
			reply:   reply,
			cursor:  1,
		},
		events: make(chan tea.Msg),
	}

	updated, _ := m.updateApprovalPrompt(tea.KeyMsg{Type: tea.KeyEnter})
	fm := updated.(chatModel)
	if fm.allowAllToolsEnabled() {
		t.Fatal("allowing a tool should not enable full access")
	}
	if !fm.approvalState.Allows("edit") {
		t.Fatal("edit scope was not saved")
	}
	result := <-reply
	if !result.Allow || result.AllowAll || len(result.AllowScopes) != 1 || result.AllowScopes[0] != "edit" {
		t.Fatalf("approval = %#v, want per-tool approval", result)
	}
}

func TestChatApprovalLabelsSecondChoiceAsPerTool(t *testing.T) {
	lines := stripANSI(strings.Join(renderApprovalChoices(testApprovalRequest(), 1, 80), "\n"))
	if !strings.Contains(lines, "2. Always allow Edit") {
		t.Fatalf("approval choices = %q, want per-tool option", lines)
	}
	if strings.Contains(lines, "Approve all") {
		t.Fatalf("approval choices = %q, should not offer approve all as option 2", lines)
	}
}

func TestChatApprovalLabelsShellChoiceAsCommandScoped(t *testing.T) {
	request := coreagent.ApprovalRequest{
		WorkingDir: "/repo",
		Calls: []coreagent.ApprovalToolCall{{
			ToolCallID:    "call-1",
			ToolName:      "bash",
			Args:          map[string]any{"command": "pwd"},
			ApprovalScope: "bash\x00pwd",
		}},
	}
	lines := stripANSI(strings.Join(renderApprovalChoices(request, 1, 80), "\n"))
	if !strings.Contains(lines, "2. Always allow this command") {
		t.Fatalf("approval choices = %q, want command-scoped option", lines)
	}
	if strings.Contains(lines, "Always allow Bash") {
		t.Fatalf("approval choices = %q, should not offer top-level Bash approval", lines)
	}
}

func TestChatApprovalUsesShellNameForPermissionPrompt(t *testing.T) {
	request := coreagent.ApprovalRequest{
		WorkingDir: "/repo",
		Calls: []coreagent.ApprovalToolCall{{
			ToolCallID:    "call-1",
			ToolName:      "bash",
			Args:          map[string]any{"command": "pwd"},
			ApprovalScope: "bash\x00pwd",
		}},
	}

	detail := stripANSI(approvalRequestDetail(request, 80))
	if !strings.Contains(detail, "$ pwd") {
		t.Fatalf("approval detail should show command prompt, got %q", detail)
	}

	m := chatModel{}
	m.upsertApprovalToolEntries(request)
	if len(m.entries) != 1 {
		t.Fatalf("entries = %#v", m.entries)
	}
	line := stripANSI(toolStatusLine(m.entries[0]))
	if !strings.Contains(line, `Bash("pwd")`) || !strings.Contains(line, "needs approval") {
		t.Fatalf("approval status line = %q", line)
	}
}

func TestChatApprovalRendersSkillLoad(t *testing.T) {
	request := coreagent.ApprovalRequest{
		WorkingDir: "/repo",
		Calls: []coreagent.ApprovalToolCall{{
			ToolCallID:    "call-skill-1",
			ToolName:      "skill",
			Args:          map[string]any{"name": "release-notes"},
			ApprovalScope: "skill",
		}},
	}

	lines := stripANSI(strings.Join((&chatModel{approvalPrompt: &chatApprovalPrompt{request: request}}).renderApprovalPromptLines(80), "\n"))
	for _, want := range []string{"name: release-notes", "2. Always allow skill"} {
		if !strings.Contains(lines, want) {
			t.Fatalf("skill approval prompt missing %q:\n%s", want, lines)
		}
	}

	m := chatModel{}
	m.upsertApprovalToolEntries(request)
	if len(m.entries) != 1 || !strings.Contains(stripANSI(toolStatusLine(m.entries[0])), `skill("release-notes") needs approval`) {
		t.Fatalf("skill approval entry = %#v", m.entries)
	}
}

func TestChatApprovalPromptOmitsDuplicateBatchDetails(t *testing.T) {
	request := coreagent.ApprovalRequest{
		WorkingDir: "/repo",
		Calls: []coreagent.ApprovalToolCall{
			{
				ToolCallID:    "call-1",
				ToolName:      "bash",
				Args:          map[string]any{"command": "git rev-parse --abbrev-ref HEAD"},
				ApprovalScope: "bash\x00git rev-parse --abbrev-ref HEAD",
			},
			{
				ToolCallID:    "call-2",
				ToolName:      "bash",
				Args:          map[string]any{"command": "git branch -a"},
				ApprovalScope: "bash\x00git branch -a",
			},
		},
	}
	m := chatModel{
		approvalPrompt: &chatApprovalPrompt{request: request},
	}

	lines := stripANSI(strings.Join(m.renderApprovalPromptLines(120), "\n"))
	if strings.Contains(lines, `Bash("git rev-parse --abbrev-ref HEAD")`) || strings.Contains(lines, `Bash("git branch -a")`) {
		t.Fatalf("batched approval prompt should not duplicate visible tool rows:\n%s", lines)
	}
	for _, want := range []string{"1. Approve once", "2. Always allow these requests", "3. Deny"} {
		if !strings.Contains(lines, want) {
			t.Fatalf("batched approval prompt missing %q:\n%s", want, lines)
		}
	}
}

func TestChatApprovalKeepsQueuedBatchCallsVisible(t *testing.T) {
	reply := make(chan coreagent.Approval, 1)
	request := coreagent.ApprovalRequest{
		WorkingDir: "/repo",
		Calls: []coreagent.ApprovalToolCall{
			{
				ToolCallID:    "call-1",
				ToolName:      "bash",
				Args:          map[string]any{"command": "git rev-parse --abbrev-ref HEAD"},
				ApprovalScope: "bash\x00git rev-parse --abbrev-ref HEAD",
			},
			{
				ToolCallID:    "call-2",
				ToolName:      "bash",
				Args:          map[string]any{"command": "git branch -a"},
				ApprovalScope: "bash\x00git branch -a",
			},
		},
	}
	m := chatModel{
		running: true,
		events:  make(chan tea.Msg),
	}
	m.openApprovalPrompt(chatApprovalPromptMsg{request: request, reply: reply})

	updated, _ := m.resolveApprovalPrompt(chatApprovalChoice{allow: true})
	m = updated.(chatModel)
	m.applyAgentEvent(coreagent.Event{
		Type:       coreagent.EventToolStarted,
		ToolCallID: "call-1",
		ToolName:   "bash",
		Args:       request.Calls[0].Args,
	})
	m.applyAgentEvent(coreagent.Event{
		Type:       coreagent.EventToolFinished,
		ToolCallID: "call-1",
		ToolName:   "bash",
		Args:       request.Calls[0].Args,
		Content:    "parth-agent-tui\n",
	})
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventMessageDelta, Content: "I'm on branch parth-agent-tui."})

	transcript := stripANSI(m.renderTranscript(180))
	for _, want := range []string{
		`Bash("git rev-parse --abbrev-ref HEAD")`,
		`Bash("git branch -a")`,
		"I'm on branch parth-agent-tui.",
	} {
		if !strings.Contains(transcript, want) {
			t.Fatalf("transcript missing %q:\n%s", want, transcript)
		}
	}
}

func TestChatApprovalPromptRepaintsFlowTranscript(t *testing.T) {
	reply := make(chan coreagent.Approval, 1)
	request := coreagent.ApprovalRequest{
		WorkingDir: "/repo",
		Calls: []coreagent.ApprovalToolCall{
			{
				ToolCallID:    "call-1",
				ToolName:      "web_fetch",
				Args:          map[string]any{"url": "https://parthsareen.com/"},
				ApprovalScope: "web_fetch",
			},
			{
				ToolCallID:    "call-2",
				ToolName:      "web_fetch",
				Args:          map[string]any{"url": "https://github.com/ParthSareen"},
				ApprovalScope: "web_fetch",
			},
		},
	}
	m := chatModel{
		running:          true,
		width:            160,
		flowPrintedLines: 1,
		entries: []chatEntry{
			{role: "user", content: "research parth"},
		},
	}

	updated, cmd := m.Update(chatApprovalPromptMsg{request: request, reply: reply})
	if cmd == nil {
		t.Fatal("opening approval should repaint flow transcript")
	}
	fm := updated.(chatModel)
	transcript := stripANSI(fm.renderTranscript(160))
	for _, want := range []string{
		`Web Fetch("https://parthsareen.com/") needs approval`,
		`Web Fetch("https://github.com/ParthSareen") needs approval`,
	} {
		if !strings.Contains(transcript, want) {
			t.Fatalf("transcript missing %q:\n%s", want, transcript)
		}
	}
}

func TestChatApprovalResolutionRepaintsFlowTranscript(t *testing.T) {
	reply := make(chan coreagent.Approval, 1)
	request := coreagent.ApprovalRequest{
		WorkingDir: "/repo",
		Calls: []coreagent.ApprovalToolCall{
			{
				ToolCallID:    "call-1",
				ToolName:      "web_fetch",
				Args:          map[string]any{"url": "https://parthsareen.com/"},
				ApprovalScope: "web_fetch",
			},
			{
				ToolCallID:    "call-2",
				ToolName:      "web_fetch",
				Args:          map[string]any{"url": "https://github.com/ParthSareen"},
				ApprovalScope: "web_fetch",
			},
		},
	}
	m := chatModel{
		running: true,
		width:   160,
		events:  make(chan tea.Msg),
		entries: []chatEntry{
			{role: "user", content: "research parth"},
		},
	}
	m.openApprovalPrompt(chatApprovalPromptMsg{request: request, reply: reply})
	printed := len(m.transcriptLines(160))
	m.flowPrintedLines = printed

	updated, cmd := m.resolveApprovalPrompt(chatApprovalChoice{allow: true})
	if cmd == nil {
		t.Fatal("approval resolution should keep waiting for agent events")
	}
	fm := updated.(chatModel)
	if fm.flowPrintedLines >= printed {
		t.Fatalf("approval resolution should repaint and hold queued rows, flowPrintedLines = %d, was %d", fm.flowPrintedLines, printed)
	}
	if result := <-reply; !result.Allow {
		t.Fatalf("approval = %#v, want allow", result)
	}
}

func TestChatApprovalBatchCollapsesAtNextToolBoundary(t *testing.T) {
	reply := make(chan coreagent.Approval, 1)
	request := coreagent.ApprovalRequest{
		WorkingDir: "/repo",
		Calls: []coreagent.ApprovalToolCall{
			{
				ToolCallID:    "call-1",
				ToolName:      "bash",
				Args:          map[string]any{"command": "git rev-parse --abbrev-ref HEAD"},
				ApprovalScope: "bash\x00git rev-parse --abbrev-ref HEAD",
			},
			{
				ToolCallID:    "call-2",
				ToolName:      "bash",
				Args:          map[string]any{"command": "git branch -a"},
				ApprovalScope: "bash\x00git branch -a",
			},
		},
	}
	m := chatModel{
		running: true,
		events:  make(chan tea.Msg),
	}
	m.openApprovalPrompt(chatApprovalPromptMsg{request: request, reply: reply})

	updated, _ := m.resolveApprovalPrompt(chatApprovalChoice{allow: true})
	m = updated.(chatModel)
	for _, call := range request.Calls {
		m.applyAgentEvent(coreagent.Event{
			Type:       coreagent.EventToolStarted,
			ToolCallID: call.ToolCallID,
			ToolName:   call.ToolName,
			Args:       call.Args,
		})
		m.applyAgentEvent(coreagent.Event{
			Type:       coreagent.EventToolFinished,
			ToolCallID: call.ToolCallID,
			ToolName:   call.ToolName,
			Args:       call.Args,
			Content:    "ok\n",
		})
	}
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventMessageDelta, Content: "I'm on branch parth-agent-tui."})

	transcript := stripANSI(m.renderTranscript(180))
	if strings.Contains(transcript, "Ran 2 commands") {
		t.Fatalf("completed batch should stay expanded until the next tool boundary:\n%s", transcript)
	}
	if !strings.Contains(transcript, `Bash("git branch -a")`) {
		t.Fatalf("completed batch should keep concrete command rows before the next boundary:\n%s", transcript)
	}

	m.applyAgentEvent(coreagent.Event{
		Type:       coreagent.EventToolStarted,
		ToolCallID: "call-3",
		ToolName:   "bash",
		Args:       map[string]any{"command": "git status --short"},
	})
	transcript = stripANSI(m.renderTranscript(180))
	if !strings.Contains(transcript, "Ran 2 commands") {
		t.Fatalf("completed batch should collapse when a new tool starts:\n%s", transcript)
	}
	if !strings.Contains(transcript, `Bash("git status --short")`) {
		t.Fatalf("new running command should remain concrete after previous batch collapses:\n%s", transcript)
	}
}

func TestChatApprovalPrompterCancels(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	result, err := (chatApprovalPrompter{ch: make(chan tea.Msg)}).PromptApproval(ctx, testApprovalRequest())
	if err != nil {
		t.Fatal(err)
	}
	if result.Allow || result.Reason == "" {
		t.Fatalf("approval = %#v, want canceled denial", result)
	}
}

func TestChatApprovalControllerAutoApprovesAfterFullAccessToggle(t *testing.T) {
	events := make(chan tea.Msg, 1)
	state := testApprovalState(false, nil)
	controller := newChatApprovalController(events, state)
	state.GrantAll()

	result, err := controller.PromptApproval(context.Background(), testApprovalRequest())
	if err != nil {
		t.Fatal(err)
	}
	if !result.Allow || !result.AllowAll {
		t.Fatalf("approval = %#v, want full-access approval", result)
	}
	select {
	case msg := <-events:
		t.Fatalf("approval UI event should not be sent after full access toggle: %#v", msg)
	default:
	}
}

func TestChatPermissionToggleSyncsRunningApprovalController(t *testing.T) {
	events := make(chan tea.Msg, 1)
	state := testApprovalState(false, nil)
	m := chatModel{
		approvalState:      state,
		approvalController: newChatApprovalController(events, state),
	}

	updated, _ := m.togglePermissionMode()
	fm := updated.(chatModel)
	result, err := fm.approvalController.PromptApproval(context.Background(), testApprovalRequest())
	if err != nil {
		t.Fatal(err)
	}
	if !result.Allow || !result.AllowAll {
		t.Fatalf("approval = %#v, want full-access approval", result)
	}
}

func TestChatPermissionToggleFromFullAccessRequiresReviewInRunningController(t *testing.T) {
	events := make(chan tea.Msg, 1)
	state := testApprovalState(true, nil)
	m := chatModel{
		approvalState:      state,
		approvalController: newChatApprovalController(events, state),
	}

	updated, _ := m.togglePermissionMode()
	fm := updated.(chatModel)
	if fm.allowAllToolsEnabled() {
		t.Fatal("full access should be disabled")
	}

	resultCh := make(chan coreagent.Approval, 1)
	go func() {
		result, err := fm.approvalController.PromptApproval(context.Background(), testApprovalRequest())
		if err != nil {
			resultCh <- coreagent.Approval{Reason: err.Error()}
			return
		}
		resultCh <- result
	}()

	select {
	case msg := <-events:
		prompt, ok := msg.(chatApprovalPromptMsg)
		if !ok {
			t.Fatalf("event = %#v, want approval prompt", msg)
		}
		prompt.reply <- coreagent.Approval{Reason: "denied"}
	case <-time.After(time.Second):
		t.Fatal("expected approval prompt after toggling from full access to review")
	}

	result := <-resultCh
	if result.Allow {
		t.Fatalf("approval = %#v, want review prompt result", result)
	}
}

func TestChatApprovalPromptSkippedWhenFullAccessEnabledInFlight(t *testing.T) {
	reply := make(chan coreagent.Approval, 1)
	// Full access is on by the time the buffered approval request reaches the
	// UI (toggled after the agent sent the request but before Update ran).
	// The stale prompt must not surface; the request is auto-approved.
	m := chatModel{approvalState: testApprovalState(true, nil), running: true}

	updated, _ := m.Update(chatApprovalPromptMsg{request: testApprovalRequest(), reply: reply})
	fm := updated.(chatModel)

	if fm.approvalPrompt != nil {
		t.Fatalf("approval prompt = %#v, want nil (full access on)", fm.approvalPrompt)
	}
	if got := fm.status; got == "approval required" {
		t.Fatalf("status = %q, should not show approval required", got)
	}
	select {
	case result := <-reply:
		if !result.Allow || !result.AllowAll {
			t.Fatalf("approval = %#v, want full-access approval", result)
		}
	default:
		t.Fatal("expected auto-approval sent on the reply channel")
	}
}
