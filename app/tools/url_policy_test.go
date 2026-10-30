//go:build windows || darwin

package tools

import "testing"

func TestDirectURLsFromText_RejectsChangedToolArgument(t *testing.T) {
	ctx := WithAllowedDirectURLs(t.Context(), "summarize https://attacker.example/x")

	if allowedDirectURL(ctx, "https://attacker.example/x!!!!") {
		t.Fatal("expected changed tool argument to be rejected")
	}
}

func TestDirectURLsFromText_ExtractsMarkdownCodeSpanURL(t *testing.T) {
	ctx := WithAllowedDirectURLs(t.Context(), "summarize `https://example.com/privacy`")

	if !allowedDirectURL(ctx, "https://example.com/privacy") {
		t.Fatal("expected URL wrapped in backticks to be allowed")
	}
}
