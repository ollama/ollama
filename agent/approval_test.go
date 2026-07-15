package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

type mockTool struct {
	name string
}

func (m mockTool) Name() string        { return m.name }
func (m mockTool) Description() string { return "" }
func (m mockTool) Schema() api.ToolFunction {
	return api.ToolFunction{Name: m.name}
}
func (m mockTool) Execute(context.Context, ToolContext, map[string]any) (ToolResult, error) {
	return ToolResult{}, nil
}

func TestToolApprovalScopeUsesScopedTool(t *testing.T) {
	shellTool := mockScopedTool{
		mockTool: mockTool{name: "bash"},
		scope: func(args map[string]any) string {
			if cmd, ok := args["command"].(string); ok {
				cmd = strings.TrimSpace(cmd)
				if cmd != "" {
					return "bash\x00" + cmd
				}
			}
			return "bash"
		}}
	plainTool := mockTool{name: "edit"}

	tests := []struct {
		tool Tool
		name string
		args map[string]any
		want string
	}{
		{shellTool, "bash", map[string]any{"command": " pwd "}, "bash\x00pwd"},
		{shellTool, "bash", map[string]any{"command": "Get-ChildItem"}, "bash\x00Get-ChildItem"},
		{plainTool, "edit", map[string]any{"path": "README.md"}, "edit"},
	}
	for _, tt := range tests {
		if got := toolApprovalScope(tt.tool, tt.name, tt.args); got != tt.want {
			t.Fatalf("toolApprovalScope(%q) = %q, want %q", tt.name, got, tt.want)
		}
	}
}

type mockScopedTool struct {
	mockTool
	scope func(args map[string]any) string
}

func (m mockScopedTool) ApprovalScope(args map[string]any) string {
	return m.scope(args)
}

func TestSessionApplyApprovalScopes(t *testing.T) {
	session := &Session{}
	result := Approval{AllowScopes: []string{"edit", "bash\x00pwd", " "}}

	session.applyApproval(&result)

	if !result.Allow {
		t.Fatal("scoped approval should allow the current request")
	}
	if !session.allows("edit") || !session.allows("bash\x00pwd") {
		t.Fatal("scoped approval was not saved")
	}
	if session.allows("bash") || session.allows("bash\x00ls") {
		t.Fatal("shell approval was too broad")
	}
	if session.ApprovalState.AllGranted() {
		t.Fatal("allow all = true, want false for scoped approval")
	}
}

func TestSessionApplyApprovalAllowAll(t *testing.T) {
	session := &Session{}
	result := Approval{AllowAll: true}

	session.applyApproval(&result)

	if !result.Allow || !session.allows("anything") {
		t.Fatalf("allow all = %v result = %#v, want allow all", session.ApprovalState.AllGranted(), result)
	}
}
