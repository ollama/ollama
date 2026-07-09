package agent

import "testing"

func TestApprovalRequestScopesShellCommandsToExactCommand(t *testing.T) {
	req := ApprovalRequest{}
	req.AddToolCall("call-1", "bash", map[string]any{"command": " pwd "})
	req.AddToolCall("call-2", "powershell", map[string]any{"command": "Get-ChildItem"})
	req.AddToolCall("call-3", "edit", map[string]any{"path": "README.md"})

	tests := []struct {
		index int
		want  string
	}{
		{index: 0, want: "bash\x00pwd"},
		{index: 1, want: "powershell\x00Get-ChildItem"},
		{index: 2, want: "edit"},
	}
	for _, tt := range tests {
		if got := req.Calls[tt.index].ApprovalScope; got != tt.want {
			t.Fatalf("call %d approval scope = %q, want %q", tt.index, got, tt.want)
		}
	}
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
	if session.ApprovalState.AllowAll() {
		t.Fatal("allow all = true, want false for scoped approval")
	}
}

func TestSessionApplyApprovalAllowAll(t *testing.T) {
	session := &Session{}
	result := Approval{AllowAll: true}

	session.applyApproval(&result)

	if !result.Allow || !session.allows("anything") {
		t.Fatalf("allow all = %v result = %#v, want allow all", session.ApprovalState.AllowAll(), result)
	}
}
