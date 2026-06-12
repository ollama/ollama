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
		return ApprovalResult{Decision: ApprovalAllowOnce}, nil
	}
	result := p.results[0]
	p.results = p.results[1:]
	return result, nil
}

func TestApprovalManagerAllowsSafeToolsWithoutPrompt(t *testing.T) {
	prompter := &recordingApprovalPrompter{}
	manager := NewApprovalManager(ApprovalManagerOptions{Prompter: prompter})

	result, err := manager.Approve(context.Background(), ApprovalRequest{
		ToolName:   "read",
		Args:       map[string]any{"path": "README.md"},
		WorkingDir: t.TempDir(),
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Decision != ApprovalAllowOnce {
		t.Fatalf("decision = %q, want allow_once", result.Decision)
	}
	if len(prompter.requests) != 0 {
		t.Fatalf("safe tool prompted: %#v", prompter.requests)
	}
}

func TestApprovalManagerDeniesEscapingPath(t *testing.T) {
	manager := NewApprovalManager(ApprovalManagerOptions{})

	result, err := manager.Approve(context.Background(), ApprovalRequest{
		ToolName:   "edit",
		Args:       map[string]any{"path": "../outside.txt"},
		WorkingDir: t.TempDir(),
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Decision != ApprovalDeny {
		t.Fatalf("decision = %q, want deny", result.Decision)
	}
	if !strings.Contains(result.Reason, "path escapes working directory") {
		t.Fatalf("reason = %q", result.Reason)
	}
}

func TestApprovalManagerPromptsForEdit(t *testing.T) {
	prompter := &recordingApprovalPrompter{}
	manager := NewApprovalManager(ApprovalManagerOptions{Prompter: prompter})

	result, err := manager.Approve(context.Background(), ApprovalRequest{
		ToolName:   "edit",
		Args:       map[string]any{"path": "note.txt", "old_text": "old", "new_text": "new"},
		WorkingDir: t.TempDir(),
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Decision != ApprovalAllowOnce {
		t.Fatalf("decision = %q, want allow_once", result.Decision)
	}
	if len(prompter.requests) != 1 {
		t.Fatalf("prompts = %d, want 1", len(prompter.requests))
	}
	request := prompter.requests[0]
	if request.Risk != ApprovalRiskMedium {
		t.Fatalf("risk = %q, want medium", request.Risk)
	}
	if !strings.Contains(strings.Join(request.Reasons, " "), "writes to a file") {
		t.Fatalf("reasons = %#v", request.Reasons)
	}
}

func TestApprovalManagerHeadlessDeniesPromptRequiredTools(t *testing.T) {
	manager := NewApprovalManager(ApprovalManagerOptions{})

	result, err := manager.Approve(context.Background(), ApprovalRequest{
		ToolName: "bash",
		Args:     map[string]any{"command": "pwd"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Decision != ApprovalDeny {
		t.Fatalf("decision = %q, want deny", result.Decision)
	}
	if !strings.Contains(result.Reason, "--auto-approve-tools") {
		t.Fatalf("reason = %q", result.Reason)
	}
}

func TestApprovalManagerSessionAllowList(t *testing.T) {
	prompter := &recordingApprovalPrompter{
		results: []ApprovalResult{{Decision: ApprovalAllowSession}},
	}
	manager := NewApprovalManager(ApprovalManagerOptions{Prompter: prompter})
	request := ApprovalRequest{
		ToolName: "bash",
		Args:     map[string]any{"command": "go test ./agent"},
	}

	result, err := manager.Approve(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if result.Decision != ApprovalAllowSession {
		t.Fatalf("decision = %q, want allow_session", result.Decision)
	}
	result, err = manager.Approve(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if result.Decision != ApprovalAllowOnce {
		t.Fatalf("second decision = %q, want allow_once", result.Decision)
	}
	if len(prompter.requests) != 1 {
		t.Fatalf("prompts = %d, want 1", len(prompter.requests))
	}
}

func TestBashApprovalClassifiesHighRiskShell(t *testing.T) {
	evaluation := DefaultApprovalPolicy{}.EvaluateApproval(context.Background(), ApprovalRequest{
		ToolName: "bash",
		Args:     map[string]any{"command": "cd / && rm -rf tmp"},
	})

	if !evaluation.RequirePrompt {
		t.Fatal("bash should require prompt")
	}
	if evaluation.Risk != ApprovalRiskHigh {
		t.Fatalf("risk = %q, want high", evaluation.Risk)
	}
	reasons := strings.Join(evaluation.Reasons, " ")
	for _, want := range []string{"changes directory", "control operator", "removes files"} {
		if !strings.Contains(reasons, want) {
			t.Fatalf("reasons = %#v, want %q", evaluation.Reasons, want)
		}
	}
}
