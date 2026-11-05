//go:build windows || darwin

package tools

import (
	"context"
	"regexp"
	"strings"
)

type directURLContextKey struct{}

var directURLPattern = regexp.MustCompile("https?://[^\\s<>\"'`]+")

func WithAllowedDirectURLs(ctx context.Context, text string) context.Context {
	allowed := make(map[string]struct{})
	for _, match := range directURLPattern.FindAllString(text, -1) {
		addAllowedDirectURLToMap(allowed, match)
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

	raw = cleanDirectURL(raw)
	if raw == "" {
		return
	}

	allowed[raw] = struct{}{}
}

func allowedDirectURL(ctx context.Context, raw string) bool {
	allowed, _ := ctx.Value(directURLContextKey{}).(map[string]struct{})
	cleaned := cleanDirectURL(raw)
	if cleaned == "" || cleaned != raw {
		return false
	}

	_, ok := allowed[cleaned]
	return ok
}

func cleanDirectURL(raw string) string {
	raw = strings.TrimSpace(raw)
	raw = strings.TrimRight(raw, ".,;:!?)]}")

	if !strings.HasPrefix(raw, "http://") && !strings.HasPrefix(raw, "https://") {
		return ""
	}

	return raw
}
