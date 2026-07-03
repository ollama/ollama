package chat

import (
	"context"
	"strings"
	"testing"

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
	if fm.allowAllTools {
		t.Fatal("allowing a tool should not enable full access")
	}
	if !fm.allowedScopes["edit"] {
		t.Fatalf("allowed scopes = %#v, want edit", fm.allowedScopes)
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
	controller := newChatApprovalController(events, false, nil)
	controller.set(true, nil)

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
	m := chatModel{
		approvalController: newChatApprovalController(events, false, nil),
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
