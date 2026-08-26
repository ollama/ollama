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

func TestDirectURLsFromText_AllowsURLEndingInPunctuation(t *testing.T) {
	const u = "https://en.wikipedia.org/wiki/Go_(programming_language)"
	ctx := WithAllowedDirectURLs(t.Context(), "please summarise "+u)

	if !allowedDirectURL(ctx, u) {
		t.Fatal("expected URL ending in a closing paren to be allowed")
	}
}

func TestDirectURLsFromText_AllowsURLTrimmedFromProse(t *testing.T) {
	ctx := WithAllowedDirectURLs(t.Context(), "summarize https://example.com/privacy.")

	if !allowedDirectURL(ctx, "https://example.com/privacy") {
		t.Fatal("expected URL without the sentence's trailing period to be allowed")
	}
}

func TestAddAllowedDirectURL_KeepsTrailingPunctuation(t *testing.T) {
	const u = "https://en.wikipedia.org/wiki/Go_(programming_language)"
	ctx := WithAllowedDirectURLs(t.Context(), "no urls here")
	addAllowedDirectURL(ctx, u)

	if !allowedDirectURL(ctx, u) {
		t.Fatal("expected discovered link to be allowed unmodified")
	}
}
