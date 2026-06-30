package chat

import (
	"context"
	"testing"

	tea "github.com/charmbracelet/bubbletea"

	coreagent "github.com/ollama/ollama/agent"
)

func testApprovalRequest() coreagent.ApprovalRequest {
	return coreagent.ApprovalRequest{
		WorkingDir: "/repo",
		Calls: []coreagent.ApprovalToolCall{{
			ToolCallID: "call-1",
			ToolName:   "edit",
			Args:       map[string]any{"path": "note.txt"},
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

func TestChatApprovalApprovesAll(t *testing.T) {
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
	if !fm.allowAllTools {
		t.Fatal("approving all should enable full access")
	}
	result := <-reply
	if !result.Allow || !result.AllowAll {
		t.Fatalf("approval = %#v, want allow all", result)
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
