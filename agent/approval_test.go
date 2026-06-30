package agent

import (
	"context"
	"strings"
	"testing"
)

type recordingApprovalPrompter struct {
	requests []ApprovalRequest
	results  []ApprovalResult
}

func (p *recordingApprovalPrompter) PromptApproval(_ context.Context, request ApprovalRequest) (ApprovalResult, error) {
	p.requests = append(p.requests, request)
	if len(p.results) == 0 {
		return ApprovalResult{Decision: ApprovalAllowCurrent}, nil
	}
	result := p.results[0]
	p.results = p.results[1:]
	return result, nil
}

func TestApprovalManagerAllowsWhenNoCallsNeedApproval(t *testing.T) {
	prompter := &recordingApprovalPrompter{}
	manager := NewApprovalManager(ApprovalManagerOptions{Prompter: prompter})

	result, err := manager.AuthorizeTools(context.Background(), ApprovalRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Decision != ApprovalAllowCurrent {
		t.Fatalf("decision = %q, want allow_current", result.Decision)
	}
	if len(prompter.requests) != 0 {
		t.Fatalf("prompts = %d, want 0", len(prompter.requests))
	}
}

func TestApprovalManagerDeniesWithoutPrompter(t *testing.T) {
	manager := NewApprovalManager(ApprovalManagerOptions{})

	result, err := manager.AuthorizeTools(context.Background(), ApprovalRequest{
		Calls: []ApprovalToolCall{{ToolCallID: "call-1", ToolName: "bash"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Decision != ApprovalDeny {
		t.Fatalf("decision = %q, want deny", result.Decision)
	}
	if !strings.Contains(result.Reason, "no approval prompter") {
		t.Fatalf("reason = %q", result.Reason)
	}
}

func TestApprovalManagerPromptsForCurrentRequest(t *testing.T) {
	prompter := &recordingApprovalPrompter{}
	manager := NewApprovalManager(ApprovalManagerOptions{Prompter: prompter})
	req := ApprovalRequest{
		WorkingDir: "/repo",
		Calls: []ApprovalToolCall{
			{ToolCallID: "call-1", ToolName: "bash", Args: map[string]any{"command": "pwd"}},
			{ToolCallID: "call-2", ToolName: "edit", Args: map[string]any{"path": "README.md"}},
		},
	}

	result, err := manager.AuthorizeTools(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if result.Decision != ApprovalAllowCurrent {
		t.Fatalf("decision = %q, want allow_current", result.Decision)
	}
	if len(prompter.requests) != 1 {
		t.Fatalf("prompts = %d, want 1", len(prompter.requests))
	}
	if got := prompter.requests[0]; got.WorkingDir != req.WorkingDir || len(got.Calls) != 2 {
		t.Fatalf("request = %#v, want current request details", got)
	}
}

func TestApprovalManagerAllowAllSkipsFuturePrompts(t *testing.T) {
	prompter := &recordingApprovalPrompter{
		results: []ApprovalResult{{Decision: ApprovalAllowAll}},
	}
	manager := NewApprovalManager(ApprovalManagerOptions{Prompter: prompter})
	req := ApprovalRequest{
		Calls: []ApprovalToolCall{{ToolCallID: "call-1", ToolName: "bash"}},
	}

	result, err := manager.AuthorizeTools(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if result.Decision != ApprovalAllowAll {
		t.Fatalf("decision = %q, want allow_all", result.Decision)
	}
	result, err = manager.AuthorizeTools(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if result.Decision != ApprovalAllowCurrent {
		t.Fatalf("second decision = %q, want allow_current", result.Decision)
	}
	if len(prompter.requests) != 1 {
		t.Fatalf("prompts = %d, want 1", len(prompter.requests))
	}
}

func TestAutoAllowApprovalAllowsCurrentRequest(t *testing.T) {
	result, err := AutoAllowApproval{}.AuthorizeTools(context.Background(), ApprovalRequest{
		Calls: []ApprovalToolCall{{ToolCallID: "call-1", ToolName: "bash"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Decision != ApprovalAllowCurrent {
		t.Fatalf("decision = %q, want allow_current", result.Decision)
	}
}
