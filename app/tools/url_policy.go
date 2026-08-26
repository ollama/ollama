//go:build windows || darwin

package tools

import (
	"context"
	"regexp"
	"strings"
)

type directURLContextKey struct{}

var directURLPattern = regexp.MustCompile("https?://[^\\s<>\"'`]+")

// trailingPunctuation is punctuation a URL can pick up from the prose around it.
const trailingPunctuation = ".,;:!?)]}"

func WithAllowedDirectURLs(ctx context.Context, text string) context.Context {
	allowed := make(map[string]struct{})
	for _, match := range directURLPattern.FindAllString(text, -1) {
		// A URL extracted from prose may have swallowed the punctuation that
		// followed it, but it may also genuinely end in punctuation, as
		// Wikipedia disambiguation URLs do. Allow both readings.
		addAllowedDirectURLToMap(allowed, match)
		addAllowedDirectURLToMap(allowed, strings.TrimRight(match, trailingPunctuation))
	}
	return context.WithValue(ctx, directURLContextKey{}, allowed)
}

func addAllowedDirectURL(ctx context.Context, raw string) {
	allowed, _ := ctx.Value(directURLContextKey{}).(map[string]struct{})
	addAllowedDirectURLToMap(allowed, raw)
}

func addAllowedDirectURLToMap(allowed map[string]struct{}, raw string) {
	if allowed == nil {
		return
	}

	raw = normalizeDirectURL(raw)
	if raw == "" {
		return
	}

	allowed[raw] = struct{}{}
}

func allowedDirectURL(ctx context.Context, raw string) bool {
	allowed, _ := ctx.Value(directURLContextKey{}).(map[string]struct{})
	raw = normalizeDirectURL(raw)
	if raw == "" {
		return false
	}

	_, ok := allowed[raw]
	return ok
}

func normalizeDirectURL(raw string) string {
	raw = strings.TrimSpace(raw)

	if !strings.HasPrefix(raw, "http://") && !strings.HasPrefix(raw, "https://") {
		return ""
	}

	return raw
}
