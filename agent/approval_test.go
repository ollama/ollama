package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

type recordingApprovalPrompter struct {
	requests []ApprovalRequest
	results  []ApprovalResult
}

type allowWithoutPromptPolicy struct{}

type approvalRequiredTestTool struct{}

func (p *recordingApprovalPrompter) PromptApproval(_ context.Context, request ApprovalRequest) (ApprovalResult, error) {
	p.requests = append(p.requests, request)
	if len(p.results) == 0 {
		return ApprovalResult{Decision: ApprovalAllowOnce}, nil
	}
	result := p.results[0]
	p.results = p.results[1:]
	return result, nil
}

func (allowWithoutPromptPolicy) EvaluateApproval(context.Context, ApprovalRequest) ApprovalEvaluation {
	return ApprovalEvaluation{Decision: ApprovalAllowOnce, Risk: ApprovalRiskLow}
}

func (approvalRequiredTestTool) Name() string {
	return "approval_required"
}

func (approvalRequiredTestTool) Description() string {
	return "requires approval"
}

func (approvalRequiredTestTool) Schema() api.ToolFunction {
	return api.ToolFunction{Name: "approval_required"}
}

func (approvalRequiredTestTool) Execute(context.Context, ToolContext, map[string]any) (ToolResult, error) {
	return ToolResult{Content: "ok"}, nil
}

func (approvalRequiredTestTool) RequiresApproval(map[string]any) bool {
	return true
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

func TestApprovalManagerToolRequiredOverridePromptsInApprove(t *testing.T) {
	prompter := &recordingApprovalPrompter{}
	manager := NewApprovalManager(ApprovalManagerOptions{Policy: allowWithoutPromptPolicy{}, Prompter: prompter})
	tool := approvalRequiredTestTool{}
	request := ApprovalRequest{
		ToolName:             tool.Name(),
		Args:                 map[string]any{},
		ToolApprovalRequired: ToolRequiresApproval(tool, nil),
	}

	if !manager.RequiresApproval(context.Background(), tool, request) {
		t.Fatal("tool-required approval should require a prompt")
	}
	result, err := manager.Approve(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if result.Decision != ApprovalAllowOnce {
		t.Fatalf("decision = %q, want allow_once", result.Decision)
	}
	if len(prompter.requests) != 1 {
		t.Fatalf("prompts = %d, want 1", len(prompter.requests))
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

func TestApprovalManagerSanitizesEditSummary(t *testing.T) {
	evaluation := evaluateEditApproval(ApprovalRequest{
		ToolName:   "edit",
		Args:       map[string]any{"path": "notes/\x1b[31mred\nfile.txt"},
		WorkingDir: t.TempDir(),
	})
	if strings.ContainsAny(evaluation.Summary, "\n\r\x1b") {
		t.Fatalf("summary contains control characters: %q", evaluation.Summary)
	}
	if !strings.Contains(evaluation.Summary, "notes/red file.txt") {
		t.Fatalf("summary = %q, want sanitized path", evaluation.Summary)
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

func TestBashApprovalClassifiesDynamicShellEvasions(t *testing.T) {
	tests := []struct {
		name   string
		cmd    string
		reason string
	}{
		{name: "function declaration", cmd: "f() { rm -rf /; } && f", reason: "defines shell functions"},
		{name: "eval", cmd: `eval "$cmd"`, reason: "evaluates shell code"},
		{name: "variable command name", cmd: "$DANGER --flag", reason: "dynamic command name"},
		{name: "command substitution command name", cmd: "$(echo rm) -rf /", reason: "dynamic command name"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			evaluation := DefaultApprovalPolicy{}.EvaluateApproval(context.Background(), ApprovalRequest{
				ToolName: "bash",
				Args:     map[string]any{"command": tt.cmd},
			})
			if !evaluation.RequirePrompt {
				t.Fatal("bash evasion should require prompt")
			}
			if evaluation.Risk != ApprovalRiskHigh {
				t.Fatalf("risk = %q, want high", evaluation.Risk)
			}
			if reasons := strings.Join(evaluation.Reasons, " "); !strings.Contains(reasons, tt.reason) {
				t.Fatalf("reasons = %#v, want %q", evaluation.Reasons, tt.reason)
			}
		})
	}
}

func TestBashApprovalClassifiesDestructiveGitWithGlobalOptions(t *testing.T) {
	tests := []struct {
		name   string
		cmd    string
		reason string
	}{
		{name: "git reset hard after cwd", cmd: "git -C /tmp reset --hard", reason: "runs destructive git reset"},
		{name: "git reset hard after config", cmd: "git -c core.autocrlf=false reset --hard", reason: "runs destructive git reset"},
		{name: "git clean after work tree", cmd: "git --work-tree=/tmp clean -fdx", reason: "runs destructive git clean"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			evaluation := DefaultApprovalPolicy{}.EvaluateApproval(context.Background(), ApprovalRequest{
				ToolName: "bash",
				Args:     map[string]any{"command": tt.cmd},
			})
			if evaluation.Risk != ApprovalRiskHigh {
				t.Fatalf("risk = %q, want high", evaluation.Risk)
			}
			if reasons := strings.Join(evaluation.Reasons, " "); !strings.Contains(reasons, tt.reason) {
				t.Fatalf("reasons = %#v, want %q", evaluation.Reasons, tt.reason)
			}
		})
	}
}

func TestBashApprovalClassifiesFindMutations(t *testing.T) {
	tests := []struct {
		name   string
		cmd    string
		reason string
	}{
		{name: "delete", cmd: "find . -name '*.tmp' -delete", reason: "deletes files via find"},
		{name: "exec", cmd: `find . -name '*.tmp' -exec rm -rf {} \;`, reason: "executes commands via find"},
		{name: "exec nested destructive command", cmd: `find . -name '*.tmp' -exec rm -rf {} \;`, reason: "removes files destructively"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			evaluation := DefaultApprovalPolicy{}.EvaluateApproval(context.Background(), ApprovalRequest{
				ToolName: "bash",
				Args:     map[string]any{"command": tt.cmd},
			})
			if evaluation.Risk != ApprovalRiskHigh {
				t.Fatalf("risk = %q, want high", evaluation.Risk)
			}
			if reasons := strings.Join(evaluation.Reasons, " "); !strings.Contains(reasons, tt.reason) {
				t.Fatalf("reasons = %#v, want %q", evaluation.Reasons, tt.reason)
			}
		})
	}
}

func TestWebApprovalRequiresPrompt(t *testing.T) {
	tests := []struct {
		name    string
		tool    string
		args    map[string]any
		summary string
	}{
		{
			name:    "search",
			tool:    "web_search",
			args:    map[string]any{"query": "Ollama agents"},
			summary: "Web Search wants to search for \"Ollama agents\"",
		},
		{
			name:    "fetch",
			tool:    "web_fetch",
			args:    map[string]any{"url": "https://ollama.com"},
			summary: "Web Fetch wants to fetch https://ollama.com",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			evaluation := DefaultApprovalPolicy{}.EvaluateApproval(context.Background(), ApprovalRequest{
				ToolName: tt.tool,
				Args:     tt.args,
			})
			if !evaluation.RequirePrompt {
				t.Fatal("web tool should require prompt")
			}
			if evaluation.Decision != "" && evaluation.Decision != ApprovalAllowOnce {
				t.Fatalf("decision = %q, want allow once", evaluation.Decision)
			}
			if evaluation.Risk != ApprovalRiskMedium {
				t.Fatalf("risk = %q, want medium", evaluation.Risk)
			}
			if evaluation.Summary != tt.summary {
				t.Fatalf("summary = %q, want %q", evaluation.Summary, tt.summary)
			}
		})
	}
}

func TestWebApprovalDeniesMissingArgs(t *testing.T) {
	for _, tool := range []string{"web_search", "web_fetch"} {
		evaluation := DefaultApprovalPolicy{}.EvaluateApproval(context.Background(), ApprovalRequest{
			ToolName: tool,
			Args:     map[string]any{},
		})
		if evaluation.Decision != ApprovalDeny {
			t.Fatalf("%s missing args decision = %q, want deny", tool, evaluation.Decision)
		}
		if evaluation.Risk != ApprovalRiskHigh {
			t.Fatalf("%s missing args risk = %q, want high", tool, evaluation.Risk)
		}
	}
}
