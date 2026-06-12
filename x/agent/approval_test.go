package agent

import (
	"strings"
	"testing"
)

func TestApprovalManager_IsAllowed(t *testing.T) {
	am := NewApprovalManager()

	// Initially nothing is allowed
	if am.IsAllowed("test_tool", nil) {
		t.Error("expected test_tool to not be allowed initially")
	}

	// Add to allowlist
	am.AddToAllowlist("test_tool", nil)

	// Now it should be allowed
	if !am.IsAllowed("test_tool", nil) {
		t.Error("expected test_tool to be allowed after AddToAllowlist")
	}

	// Other tools should still not be allowed
	if am.IsAllowed("other_tool", nil) {
		t.Error("expected other_tool to not be allowed")
	}
}

func TestApprovalManager_Reset(t *testing.T) {
	am := NewApprovalManager()

	am.AddToAllowlist("tool1", nil)
	am.AddToAllowlist("tool2", nil)

	if !am.IsAllowed("tool1", nil) || !am.IsAllowed("tool2", nil) {
		t.Error("expected tools to be allowed")
	}

	am.Reset()

	if am.IsAllowed("tool1", nil) || am.IsAllowed("tool2", nil) {
		t.Error("expected tools to not be allowed after Reset")
	}
}

func TestApprovalManager_AllowedTools(t *testing.T) {
	am := NewApprovalManager()

	tools := am.AllowedTools()
	if len(tools) != 0 {
		t.Errorf("expected 0 allowed tools, got %d", len(tools))
	}

	am.AddToAllowlist("tool1", nil)
	am.AddToAllowlist("tool2", nil)

	tools = am.AllowedTools()
	if len(tools) != 2 {
		t.Errorf("expected 2 allowed tools, got %d", len(tools))
	}
}

func TestAllowlistKey(t *testing.T) {
	tests := []struct {
		name     string
		toolName string
		args     map[string]any
		expected string
	}{
		{
			name:     "web_search tool",
			toolName: "web_search",
			args:     map[string]any{"query": "test"},
			expected: "web_search",
		},
		{
			name:     "bash tool with command",
			toolName: "bash",
			args:     map[string]any{"command": "ls -la"},
			expected: "bash:ls -la",
		},
		{
			name:     "bash tool without command",
			toolName: "bash",
			args:     map[string]any{},
			expected: "bash",
		},
		{
			name:     "other tool",
			toolName: "custom_tool",
			args:     map[string]any{"param": "value"},
			expected: "custom_tool",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := AllowlistKey(tt.toolName, tt.args)
			if result != tt.expected {
				t.Errorf("AllowlistKey(%s, %v) = %s, expected %s",
					tt.toolName, tt.args, result, tt.expected)
			}
		})
	}
}

func TestApprovalManager_PathTraversalBlocked(t *testing.T) {
	am := NewApprovalManager()

	// Allow "cat tools/file.go" - creates prefix "cat:tools/"
	am.AddToAllowlist("bash", map[string]any{"command": "cat tools/file.go"})

	// Path traversal attack: should NOT be allowed
	if am.IsAllowed("bash", map[string]any{"command": "cat tools/../../etc/passwd"}) {
		t.Error("SECURITY: path traversal attack should NOT be allowed")
	}

	// Another traversal variant
	if am.IsAllowed("bash", map[string]any{"command": "cat tools/../../../etc/shadow"}) {
		t.Error("SECURITY: deep path traversal should NOT be allowed")
	}

	// Valid subdirectory access should still work
	if !am.IsAllowed("bash", map[string]any{"command": "cat tools/subdir/file.go"}) {
		t.Error("expected cat tools/subdir/file.go to be allowed")
	}

	// Safe ".." that normalizes to within allowed directory should work
	// tools/subdir/../other.go normalizes to tools/other.go which is under tools/
	if !am.IsAllowed("bash", map[string]any{"command": "cat tools/subdir/../other.go"}) {
		t.Error("expected cat tools/subdir/../other.go to be allowed (normalizes to tools/other.go)")
	}
}

func TestApprovalManager_PrefixAllowlist(t *testing.T) {
	am := NewApprovalManager()

	// Allow "cat tools/file.go"
	am.AddToAllowlist("bash", map[string]any{"command": "cat tools/file.go"})

	// Should allow other files in same directory
	if !am.IsAllowed("bash", map[string]any{"command": "cat tools/other.go"}) {
		t.Error("expected cat tools/other.go to be allowed via prefix")
	}

	// Should not allow different directory
	if am.IsAllowed("bash", map[string]any{"command": "cat src/main.go"}) {
		t.Error("expected cat src/main.go to NOT be allowed")
	}

	// Should not allow different command in same directory
	if am.IsAllowed("bash", map[string]any{"command": "rm tools/file.go"}) {
		t.Error("expected rm tools/file.go to NOT be allowed (rm is not a safe command)")
	}
}

func TestApprovalManager_HierarchicalPrefixAllowlist(t *testing.T) {
	am := NewApprovalManager()

	// Allow "cat tools/file.go" - this creates prefix "cat:tools/"
	am.AddToAllowlist("bash", map[string]any{"command": "cat tools/file.go"})

	// Should allow subdirectories (hierarchical matching)
	if !am.IsAllowed("bash", map[string]any{"command": "cat tools/subdir/file.go"}) {
		t.Error("expected cat tools/subdir/file.go to be allowed via hierarchical prefix")
	}

	// Should allow deeply nested subdirectories
	if !am.IsAllowed("bash", map[string]any{"command": "cat tools/a/b/c/deep.go"}) {
		t.Error("expected cat tools/a/b/c/deep.go to be allowed via hierarchical prefix")
	}

	// Should still allow same directory
	if !am.IsAllowed("bash", map[string]any{"command": "cat tools/another.go"}) {
		t.Error("expected cat tools/another.go to be allowed")
	}

	// Should NOT allow different base directory
	if am.IsAllowed("bash", map[string]any{"command": "cat src/main.go"}) {
		t.Error("expected cat src/main.go to NOT be allowed")
	}

	// Should NOT allow different command even in subdirectory
	if am.IsAllowed("bash", map[string]any{"command": "ls tools/subdir/"}) {
		t.Error("expected ls tools/subdir/ to NOT be allowed (different command)")
	}

	// Should NOT allow similar but different directory name
	if am.IsAllowed("bash", map[string]any{"command": "cat toolsbin/file.go"}) {
		t.Error("expected cat toolsbin/file.go to NOT be allowed (different directory)")
	}
}

func TestApprovalManager_HierarchicalPrefixAllowlist_CrossPlatform(t *testing.T) {
	am := NewApprovalManager()

	// Allow with forward slashes (Unix-style)
	am.AddToAllowlist("bash", map[string]any{"command": "cat tools/file.go"})

	// Should work with backslashes too (Windows-style) - normalized internally
	if !am.IsAllowed("bash", map[string]any{"command": "cat tools\\subdir\\file.go"}) {
		t.Error("expected cat tools\\subdir\\file.go to be allowed via hierarchical prefix (Windows path)")
	}

	// Mixed slashes should also work
	if !am.IsAllowed("bash", map[string]any{"command": "cat tools\\a/b\\c/deep.go"}) {
		t.Error("expected mixed slash path to be allowed via hierarchical prefix")
	}
}

func TestMatchesHierarchicalPrefix(t *testing.T) {
	am := NewApprovalManager()

	// Add prefix for "cat:tools/"
	am.prefixes["cat:tools/"] = true

	tests := []struct {
		name     string
		prefix   string
		expected bool
	}{
		{
			name:     "exact match",
			prefix:   "cat:tools/",
			expected: true, // exact match also passes HasPrefix - caller handles exact match first
		},
		{
			name:     "subdirectory",
			prefix:   "cat:tools/subdir/",
			expected: true,
		},
		{
			name:     "deeply nested",
			prefix:   "cat:tools/a/b/c/",
			expected: true,
		},
		{
			name:     "different base directory",
			prefix:   "cat:src/",
			expected: false,
		},
		{
			name:     "different command same path",
			prefix:   "ls:tools/",
			expected: false,
		},
		{
			name:     "similar directory name",
			prefix:   "cat:toolsbin/",
			expected: false,
		},
		{
			name:     "invalid prefix format",
			prefix:   "cattools",
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := am.matchesHierarchicalPrefix(tt.prefix)
			if result != tt.expected {
				t.Errorf("matchesHierarchicalPrefix(%q) = %v, expected %v",
					tt.prefix, result, tt.expected)
			}
		})
	}
}

func TestFormatApprovalResult(t *testing.T) {
	tests := []struct {
		name     string
		toolName string
		args     map[string]any
		result   ApprovalResult
		contains string
	}{
		{
			name:     "approved bash",
			toolName: "bash",
			args:     map[string]any{"command": "ls"},
			result:   ApprovalResult{Decision: ApprovalOnce},
			contains: "bash: ls",
		},
		{
			name:     "denied web_search",
			toolName: "web_search",
			args:     map[string]any{"query": "test"},
			result:   ApprovalResult{Decision: ApprovalDeny},
			contains: "Denied",
		},
		{
			name:     "always allowed",
			toolName: "bash",
			args:     map[string]any{"command": "pwd"},
			result:   ApprovalResult{Decision: ApprovalAlways},
			contains: "Always allowed",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := FormatApprovalResult(tt.toolName, tt.args, tt.result)
			if result == "" {
				t.Error("expected non-empty result")
			}
			// Just check it contains expected substring
			// (can't check exact string due to ANSI codes)
		})
	}
}

func TestFormatDenyResult(t *testing.T) {
	result := FormatDenyResult("bash", "")
	if result != "User denied execution of bash." {
		t.Errorf("unexpected result: %s", result)
	}

	result = FormatDenyResult("bash", "too dangerous")
	if result != "User denied execution of bash. Reason: too dangerous" {
		t.Errorf("unexpected result: %s", result)
	}
}

func newSelectorStateForTest() *selectorState {
	return &selectorState{
		selected:   0,
		termWidth:  80,
		boxWidth:   60,
		innerWidth: 56,
	}
}

func TestHandleSelectorInput_TypingStartsDenyEdit(t *testing.T) {
	state := newSelectorStateForTest()

	outcome, changed := handleSelectorInput(state, []byte("a"))
	if outcome.done {
		t.Fatal("typing should not finish the selector")
	}
	if !changed {
		t.Fatal("typing should change selector state")
	}
	if state.selected != 2 {
		t.Fatalf("selected = %d, want 2", state.selected)
	}
	if !state.editingDeny {
		t.Fatal("typing should enter deny edit mode")
	}
	if state.denyReason != "a" {
		t.Fatalf("denyReason = %q, want %q", state.denyReason, "a")
	}

	outcome, changed = handleSelectorInput(state, []byte("12"))
	if outcome.done {
		t.Fatal("typing digits in deny edit mode should not finish the selector")
	}
	if !changed {
		t.Fatal("typing digits in deny edit mode should update the reason")
	}
	if state.denyReason != "a12" {
		t.Fatalf("denyReason = %q, want %q", state.denyReason, "a12")
	}
}

func TestHandleSelectorInput_TabStartsDenyEdit(t *testing.T) {
	state := newSelectorStateForTest()
	state.selected = 1

	outcome, changed := handleSelectorInput(state, []byte{'\t'})
	if outcome.done {
		t.Fatal("tab should not finish the selector")
	}
	if !changed {
		t.Fatal("tab should enter deny edit mode")
	}
	if state.selected != 2 || !state.editingDeny {
		t.Fatalf("tab should select deny and start editing, got selected=%d editing=%v", state.selected, state.editingDeny)
	}

	_, changed = handleSelectorInput(state, []byte("2"))
	if !changed {
		t.Fatal("digit should append to deny reason while editing")
	}
	if state.denyReason != "2" {
		t.Fatalf("denyReason = %q, want %q", state.denyReason, "2")
	}
}

func TestHandleSelectorInput_UpClearsReasonAndMovesSelection(t *testing.T) {
	state := newSelectorStateForTest()
	state.selected = 2
	state.editingDeny = true
	state.denyReason = "not safe 123"

	outcome, changed := handleSelectorInput(state, []byte{27, '[', 'A'})
	if outcome.done {
		t.Fatal("up should not finish the selector")
	}
	if !changed {
		t.Fatal("up should change selector state")
	}
	if state.selected != 1 {
		t.Fatalf("selected = %d, want 1", state.selected)
	}
	if state.editingDeny {
		t.Fatal("up should exit deny edit mode")
	}
	if state.denyReason != "" {
		t.Fatalf("denyReason = %q, want empty string", state.denyReason)
	}
}

func TestHandleSelectorInput_EscapeClearsReason(t *testing.T) {
	state := newSelectorStateForTest()
	state.selected = 2
	state.editingDeny = true
	state.denyReason = "because 42"

	outcome, changed := handleSelectorInput(state, []byte{27})
	if outcome.done {
		t.Fatal("escape should not finish the selector")
	}
	if !changed {
		t.Fatal("escape should clear deny input")
	}
	if state.editingDeny {
		t.Fatal("escape should exit deny edit mode")
	}
	if state.denyReason != "" {
		t.Fatalf("denyReason = %q, want empty string", state.denyReason)
	}
}

func TestGetHintText_DropsTabEditCopy(t *testing.T) {
	state := newSelectorStateForTest()

	hint := getHintText(state)
	if strings.Contains(hint, "tab edit deny reason") {
		t.Fatalf("hint should not contain deny edit copy, got %q", hint)
	}
	if !strings.Contains(hint, "1-3 quick select") {
		t.Fatalf("hint = %q, expected quick select guidance", hint)
	}
	if strings.Contains(hint, "esc clear") || strings.Contains(hint, "up exits deny edit") {
		t.Fatalf("hint should stay short, got %q", hint)
	}
}

func TestDenyReasonPrefixText(t *testing.T) {
	idle := newSelectorStateForTest()
	if got := denyReasonPrefixText(idle); got != "3. Deny (tab to edit)" {
		t.Fatalf("denyReasonPrefixText(idle) = %q", got)
	}

	editing := newSelectorStateForTest()
	editing.editingDeny = true
	if got := denyReasonPrefixText(editing); got != "3. Deny: " {
		t.Fatalf("denyReasonPrefixText(editing) = %q", got)
	}
}

func TestDenyReasonPrefixDisplay(t *testing.T) {
	idle := newSelectorStateForTest()
	if got := denyReasonPrefixDisplay(idle); got != "3. Deny \033[90m(tab to edit)\033[0m" {
		t.Fatalf("denyReasonPrefixDisplay(idle) = %q", got)
	}

	editing := newSelectorStateForTest()
	editing.editingDeny = true
	if got := denyReasonPrefixDisplay(editing); got != "3. Deny: " {
		t.Fatalf("denyReasonPrefixDisplay(editing) = %q", got)
	}
}

func TestGetDenyReasonLines_WrapsWithinWidth(t *testing.T) {
	state := newSelectorStateForTest()
	state.denyReason = "this deny reason should wrap across multiple lines and stay inside the selector width"

	lines := getDenyReasonLines(state)
	if len(lines) < 2 {
		t.Fatalf("expected wrapped deny reason, got %v", lines)
	}

	maxWidth := denyReasonWrapWidth(state)
	for _, line := range lines {
		if len(line) > maxWidth {
			t.Fatalf("line %q exceeds wrap width %d", line, maxWidth)
		}
	}
}

func TestWarningLineCount_WrapsLongWarnings(t *testing.T) {
	state := newSelectorStateForTest()
	state.termWidth = 30
	state.warnings = []string{"command flagged as suspicious: Writes to a device"}

	lines := getWarningContentLines(state, state.warnings[0])
	if len(lines) < 2 {
		t.Fatalf("expected wrapped warning lines, got %v", lines)
	}

	if got := warningLineCount(state); got != len(lines)+1 {
		t.Fatalf("warningLineCount = %d, want %d", got, len(lines)+1)
	}
}

func TestGetWrappedToolLines_WrapsLongDisplayLines(t *testing.T) {
	state := newSelectorStateForTest()
	state.termWidth = 28
	state.toolDisplay = "Command: echo this is a very long command line that should wrap"

	lines := getWrappedToolLines(state)
	if len(lines) < 2 {
		t.Fatalf("expected wrapped tool display lines, got %v", lines)
	}

	for _, line := range lines {
		if len(line) > displayWrapWidth(state) {
			t.Fatalf("line %q exceeds wrap width %d", line, displayWrapWidth(state))
		}
	}
}

func TestCalculateTotalLines_IncludesWrappedDenyReason(t *testing.T) {
	state := newSelectorStateForTest()
	state.toolDisplay = "Bash: dangerous"

	baseLines := calculateTotalLines(state)

	state.denyReason = strings.Repeat("wrapped reason ", 8)
	wrappedLines := calculateTotalLines(state)

	if wrappedLines <= baseLines {
		t.Fatalf("wrappedLines = %d, want > %d", wrappedLines, baseLines)
	}
}

func TestSelectorClearLineCount_UsesLargerHeight(t *testing.T) {
	tests := []struct {
		name     string
		previous int
		next     int
		want     int
	}{
		{name: "grow", previous: 8, next: 11, want: 11},
		{name: "shrink", previous: 11, next: 8, want: 11},
		{name: "same", previous: 9, next: 9, want: 9},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := selectorClearLineCount(tt.previous, tt.next); got != tt.want {
				t.Fatalf("selectorClearLineCount(%d, %d) = %d, want %d", tt.previous, tt.next, got, tt.want)
			}
		})
	}
}

func TestSelectorClearLineCount_GrowingWrappedReasonClearsFullHeight(t *testing.T) {
	state := newSelectorStateForTest()
	state.toolDisplay = "Bash: suspicious"
	state.warnings = []string{"command flagged as suspicious"}
	state.totalLines = calculateTotalLines(state)

	state.denyReason = strings.Repeat("wrapped reason ", 8)
	nextTotalLines := calculateTotalLines(state)
	clearLines := selectorClearLineCount(state.totalLines, nextTotalLines)

	if clearLines != nextTotalLines {
		t.Fatalf("clearLines = %d, want %d when selector grows", clearLines, nextTotalLines)
	}
}
