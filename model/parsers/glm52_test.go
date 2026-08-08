package parsers

import (
	"reflect"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestGLM52ParserAdd(t *testing.T) {
	parser := GLM52Parser{}
	parser.Init([]api.Tool{
		tool("calculate", map[string]api.ToolProperty{
			"count":   {Type: api.PropertyType{"integer"}},
			"enabled": {Type: api.PropertyType{"boolean"}},
		}),
	}, nil, nil)

	// GLM-5.2 maintains GLM-4.7 parsing with enhanced response handling
	content, thinking, calls, err := parser.Add("plan</think>Answer<tool_call>calculate<arg_key>count</arg_key><arg_value>3</arg_value><arg_key>enabled</arg_key><arg_value>true</arg_value></tool_call>", true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if thinking != "plan" {
		t.Fatalf("expected thinking 'plan', got %q", thinking)
	}
	if content != "Answer" {
		t.Fatalf("expected content 'Answer', got %q", content)
	}
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(calls))
	}
	expectedArgs := args(`{"count": 3, "enabled": true}`)
	if !toolCallEqual(api.ToolCall{Function: api.ToolCallFunction{Arguments: calls[0].Function.Arguments}}, api.ToolCall{Function: api.ToolCallFunction{Arguments: expectedArgs}}) {
		t.Fatalf("expected args %#v, got %#v", expectedArgs.ToMap(), calls[0].Function.Arguments.ToMap())
	}
}

func TestGLM52ParserComplexPrompt(t *testing.T) {
	parser := GLM52Parser{}
	parser.Init([]api.Tool{
		tool("search", map[string]api.ToolProperty{
			"query": {Type: api.PropertyType{"string"}},
		}),
	}, nil, nil)

	// Test complex prompt handling with extended thinking
	content, thinking, calls, err := parser.Add("deep analysis of the problem</think>Found the answer<tool_call>search<arg_key>query</arg_key><arg_value>advanced search query</arg_value></tool_call>", true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if thinking != "deep analysis of the problem" {
		t.Fatalf("expected thinking with complex content, got %q", thinking)
	}
	if content != "Found the answer" {
		t.Fatalf("expected content 'Found the answer', got %q", content)
	}
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(calls))
	}
}

func TestGLM52ParserNoThinkingContent(t *testing.T) {
	parser := GLM52Parser{}
	parser.Init(nil, nil, nil)

	// Test response without thinking content
	content, thinking, calls, err := parser.Add("</think>Plain answer", true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if thinking != "" {
		t.Fatalf("expected empty thinking, got %q", thinking)
	}
	if content != "Plain answer" {
		t.Fatalf("expected 'Plain answer', got %q", content)
	}
	if len(calls) != 0 {
		t.Fatalf("expected 0 tool calls, got %d", len(calls))
	}
}

func TestGLM52ParserThinkingDisabled(t *testing.T) {
	parser := GLM52Parser{}
	thinkValue := api.ThinkValue{Type: api.ThinkValueBool, Bool: false}
	parser.Init(nil, nil, &thinkValue)

	// When thinking is disabled, model should not output thinking tags
	content, thinking, _, err := parser.Add("Direct answer without thinking", true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if thinking != "" {
		t.Fatalf("expected empty thinking when disabled, got %q", thinking)
	}
	if content != "Direct answer without thinking" {
		t.Fatalf("expected direct content, got %q", content)
	}
}

func TestGLM52ParserEmptyResponse(t *testing.T) {
	parser := GLM52Parser{}
	parser.Init(nil, nil, nil)

	// Test handling of empty responses (the main bug fix)
	content, thinking, calls, err := parser.Add("", true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if thinking != "" {
		t.Fatalf("expected empty thinking for empty response, got %q", thinking)
	}
	if content != "" {
		t.Fatalf("expected empty content, got %q", content)
	}
	if len(calls) != 0 {
		t.Fatalf("expected 0 tool calls, got %d", len(calls))
	}
}

func TestGLM52ParserPartialResponse(t *testing.T) {
	parser := GLM52Parser{}
	parser.Init(nil, nil, nil)

	// Test handling of incomplete/partial responses
	content1, thinking1, _, err := parser.Add("partial th", false)
	if err != nil {
		t.Fatalf("parse failed on partial 1: %v", err)
	}

	content2, thinking2, _, err := parser.Add("inking content</think>response", true)
	if err != nil {
		t.Fatalf("parse failed on partial 2: %v", err)
	}

	// Combined should equal complete response
	totalThinking := thinking1 + thinking2
	totalContent := content1 + content2
	if totalThinking != "partial thinking content" {
		t.Fatalf("expected complete thinking 'partial thinking content', got %q", totalThinking)
	}
	if totalContent != "response" {
		t.Fatalf("expected complete content 'response', got %q", totalContent)
	}
}
