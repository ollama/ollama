package tools

import (
	"errors"
	"testing"
)

func TestWebSearchTool_Name(t *testing.T) {
	tool := &WebSearchTool{}
	if tool.Name() != "web_search" {
		t.Errorf("expected name 'web_search', got '%s'", tool.Name())
	}
}

func TestWebSearchTool_Description(t *testing.T) {
	tool := &WebSearchTool{}
	if tool.Description() == "" {
		t.Error("expected non-empty description")
	}
}

func TestWebSearchTool_Execute_MissingQuery(t *testing.T) {
	tool := &WebSearchTool{}

	// Test with no query
	_, err := tool.Execute(map[string]any{})
	if err == nil {
		t.Error("expected error for missing query")
	}

	// Test with empty query
	_, err = tool.Execute(map[string]any{"query": ""})
	if err == nil {
		t.Error("expected error for empty query")
	}
}

func TestErrWebSearchAuthRequired(t *testing.T) {
	// Test that the error type exists and can be checked with errors.Is
	err := ErrWebSearchAuthRequired
	if err == nil {
		t.Fatal("ErrWebSearchAuthRequired should not be nil")
	}

	if err.Error() != "web search requires authentication" {
		t.Errorf("unexpected error message: %s", err.Error())
	}

	// Test that errors.Is works
	wrappedErr := errors.New("wrapped: " + err.Error())
	if errors.Is(wrappedErr, ErrWebSearchAuthRequired) {
		t.Error("wrapped error should not match with errors.Is")
	}

	if !errors.Is(ErrWebSearchAuthRequired, ErrWebSearchAuthRequired) {
		t.Error("ErrWebSearchAuthRequired should match itself with errors.Is")
	}
}
