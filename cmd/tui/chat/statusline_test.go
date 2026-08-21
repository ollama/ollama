package chat

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func TestJustifyThreeColumnsPlacesSegments(t *testing.T) {
	got := justifyThreeColumns("left", "mid", "right", 20)
	if !strings.HasPrefix(got, "left") {
		t.Fatalf("expected left segment at start: %q", got)
	}
	if !strings.HasSuffix(got, "right") {
		t.Fatalf("expected right segment at end: %q", got)
	}
	if !strings.Contains(got, "mid") {
		t.Fatalf("expected center segment present: %q", got)
	}
	if got := len([]rune(got)); got != 20 {
		t.Fatalf("width = %d, want 20", got)
	}
}

func TestJustifyThreeColumnsDropsCenterWhenTight(t *testing.T) {
	got := justifyThreeColumns("left", "this center is far too long to fit", "right", 12)
	if strings.Contains(got, "center") {
		t.Fatalf("center should have been dropped: %q", got)
	}
	if !strings.HasPrefix(got, "left") || !strings.HasSuffix(got, "right") {
		t.Fatalf("left/right should remain: %q", got)
	}
}

func TestJustifyThreeColumnsTruncatesRightRatherThanDroppingIt(t *testing.T) {
	// Regression test: the right segment (which now carries the model name)
	// used to be dropped entirely when it didn't fit alongside left, hiding
	// the model name on moderately narrow terminals. It should be truncated
	// instead, so at least part of it stays visible.
	got := justifyThreeColumns("dir (review)", "", "kimi-k2.7-code:cloud (auto) - 0m00s", 20)
	if got == "dir (review)"+strings.Repeat(" ", 20-len("dir (review)")) {
		t.Fatalf("right segment should not be fully dropped when it can be truncated: %q", got)
	}
	if len([]rune(got)) != 20 {
		t.Fatalf("width = %d, want 20: %q", len([]rune(got)), got)
	}
}

func TestJustifyThreeColumnsTruncatesLeftAsLastResort(t *testing.T) {
	got := justifyThreeColumns("a very long left segment indeed", "", "", 10)
	if len([]rune(got)) != 10 {
		t.Fatalf("width = %d, want 10: %q", len([]rune(got)), got)
	}
}

func TestContextPercentUnknownWindow(t *testing.T) {
	m := chatModel{}
	if _, ok := m.contextPercent(); ok {
		t.Fatal("contextPercent should report unknown when window size is 0")
	}
}

func TestContextPercentComputesRoundedValue(t *testing.T) {
	m := chatModel{
		contextTokens: 100,
		opts:          Options{ContextWindowTokens: 200},
	}
	percent, ok := m.contextPercent()
	if !ok {
		t.Fatal("contextPercent should be known when window size is set")
	}
	if percent != 50 {
		t.Fatalf("percent = %d, want 50", percent)
	}
}

func TestStatuslineBarColorThresholds(t *testing.T) {
	cases := []struct {
		percent int
		want    string
	}{
		{0, chatAnsiBrightBlack},
		{50, chatAnsiBrightBlack},
		{51, chatAnsiYellow},
		{65, chatAnsiYellow},
		{66, chatAnsiOrange},
		{85, chatAnsiOrange},
		{86, chatAnsiRed},
		{100, chatAnsiRed},
	}
	for _, c := range cases {
		if got := statuslineBarColor(c.percent); got != c.want {
			t.Errorf("statuslineBarColor(%d) = %q, want %q", c.percent, got, c.want)
		}
	}
}

func TestRenderElapsedFormatsMinutesAndHours(t *testing.T) {
	cases := []struct {
		d    time.Duration
		want string
	}{
		{0, "0m00s"},
		{9 * time.Second, "0m09s"},
		{754 * time.Second, "12m34s"},
		{61*time.Minute + 2*time.Minute, "1h03m"},
		{-5 * time.Second, "0m00s"},
	}
	for _, c := range cases {
		if got := renderElapsed(c.d); got != c.want {
			t.Errorf("renderElapsed(%v) = %q, want %q", c.d, got, c.want)
		}
	}
}

func TestCompactingLabelWithAndWithoutTokens(t *testing.T) {
	m := chatModel{compacting: true}
	if got := m.compactingLabel(); got != "Compacting" {
		t.Fatalf("compactingLabel() = %q, want %q", got, "Compacting")
	}

	m.compactingTokens = 1234
	if got := m.compactingLabel(); got != "Compacting 1234 tokens" {
		t.Fatalf("compactingLabel() = %q, want %q", got, "Compacting 1234 tokens")
	}
}

func TestAccessLevelNameReflectsApprovalState(t *testing.T) {
	m := chatModel{approvalState: testApprovalState(true, nil)}
	if got := m.accessLevelName(); got != "full-access" {
		t.Fatalf("accessLevelName() = %q, want full-access", got)
	}

	m.approvalState = testApprovalState(false, nil)
	if got := m.accessLevelName(); got != "review" {
		t.Fatalf("accessLevelName() = %q, want review", got)
	}
}

func TestRenderNativeStatuslineEmptyWithoutModel(t *testing.T) {
	m := chatModel{}
	if got := m.renderNativeStatusline(80); got != "" {
		t.Fatalf("renderNativeStatusline() = %q, want empty when no model is set", got)
	}
}

func TestRenderNativeStatuslineShowsCompactingInsteadOfBar(t *testing.T) {
	m := chatModel{
		opts:             Options{Model: "llama3.2", ContextWindowTokens: 8192},
		contextTokens:    100,
		compacting:       true,
		compactingTokens: 42,
		sessionStartedAt: time.Now(),
	}
	got := stripANSI(m.renderNativeStatusline(120))
	if !strings.Contains(got, "Compacting 42 tokens") {
		t.Fatalf("expected compacting label in statusline, got %q", got)
	}
	if strings.Contains(got, "% ctx") {
		t.Fatalf("should not show the context bar while compacting: %q", got)
	}
}

func TestRenderNativeStatuslineShowsContextBarWhenIdle(t *testing.T) {
	m := chatModel{
		opts:             Options{Model: "llama3.2", ContextWindowTokens: 8192},
		contextTokens:    100,
		sessionStartedAt: time.Now(),
	}
	got := stripANSI(m.renderNativeStatusline(120))
	if !strings.Contains(got, "% ctx") {
		t.Fatalf("expected abbreviated ctx label, got %q", got)
	}
	if strings.Contains(got, "context used") {
		t.Fatalf("label should be abbreviated to \"ctx\", not the old \"context used\": %q", got)
	}
}

func TestSerializeStateIncludesExpandedFields(t *testing.T) {
	m := chatModel{
		opts:             Options{Model: "llama3.2", ContextWindowTokens: 8192},
		contextTokens:    100,
		workingDir:       "/tmp/project",
		chatID:           "chat-1",
		approvalState:    testApprovalState(true, nil),
		compacting:       true,
		compactingTokens: 7,
		sessionStartedAt: time.Now().Add(-10 * time.Second),
	}

	var state map[string]any
	if err := json.Unmarshal(m.serializeState(), &state); err != nil {
		t.Fatalf("serializeState produced invalid JSON: %v", err)
	}

	for _, field := range []string{
		"model", "status", "context_tokens", "working_dir", "chat_id",
		"access_level", "compacting", "compacting_tokens",
		"session_elapsed_seconds", "version",
		"context_window_size", "context_used_percentage",
	} {
		if _, ok := state[field]; !ok {
			t.Errorf("serializeState missing field %q: %v", field, state)
		}
	}
	if state["access_level"] != "full-access" {
		t.Errorf("access_level = %v, want full-access", state["access_level"])
	}
	if state["compacting"] != true {
		t.Errorf("compacting = %v, want true", state["compacting"])
	}
}

func TestSerializeStateOmitsContextWindowFieldsWhenUnknown(t *testing.T) {
	m := chatModel{opts: Options{Model: "llama3.2"}}

	var state map[string]any
	if err := json.Unmarshal(m.serializeState(), &state); err != nil {
		t.Fatalf("serializeState produced invalid JSON: %v", err)
	}
	if _, ok := state["context_window_size"]; ok {
		t.Errorf("context_window_size should be omitted when window size is unknown: %v", state)
	}
	if _, ok := state["context_used_percentage"]; ok {
		t.Errorf("context_used_percentage should be omitted when window size is unknown: %v", state)
	}
}

func TestStatuslineSignatureChangesWithAccessLevel(t *testing.T) {
	m := chatModel{
		opts:          Options{Model: "llama3.2"},
		approvalState: testApprovalState(false, nil),
	}
	before := m.statuslineSignature()

	m.approvalState = testApprovalState(true, nil)
	after := m.statuslineSignature()

	if before == after {
		t.Fatal("signature should differ when access level changes")
	}
}

func TestStatuslineSignatureStableWhenNothingChanges(t *testing.T) {
	m := chatModel{
		opts:          Options{Model: "llama3.2"},
		approvalState: testApprovalState(true, nil),
	}
	if m.statuslineSignature() != m.statuslineSignature() {
		t.Fatal("signature should be stable across calls with no state change")
	}
}
