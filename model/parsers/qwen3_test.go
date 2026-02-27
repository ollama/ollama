package parsers

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestQwen3ParserThinkingEnabled(t *testing.T) {
	parser := &Qwen3Parser{hasThinkingSupport: true, defaultThinking: true}
	parser.Init(nil, nil, &api.ThinkValue{Value: true})

	content, thinking, calls, err := parser.Add("Let me think...</think>Answer.", true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	if thinking != "Let me think..." {
		t.Fatalf("expected thinking %q, got %q", "Let me think...", thinking)
	}
	if content != "Answer." {
		t.Fatalf("expected content %q, got %q", "Answer.", content)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls, got %d", len(calls))
	}
}

func TestQwen3ParserThinkingEnabledWithExplicitOpeningTag(t *testing.T) {
	parser := &Qwen3Parser{hasThinkingSupport: true, defaultThinking: true}
	parser.Init(nil, nil, &api.ThinkValue{Value: true})

	content, thinking, calls, err := parser.Add("<think>\nLet me think...</think>Answer.", true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	if thinking != "Let me think..." {
		t.Fatalf("expected thinking %q, got %q", "Let me think...", thinking)
	}
	if content != "Answer." {
		t.Fatalf("expected content %q, got %q", "Answer.", content)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls, got %d", len(calls))
	}
}

func TestQwen3ParserThinkingEnabledWithSplitOpeningTag(t *testing.T) {
	parser := &Qwen3Parser{hasThinkingSupport: true, defaultThinking: true}
	parser.Init(nil, nil, &api.ThinkValue{Value: true})

	content, thinking, calls, err := parser.Add("<thi", false)
	if err != nil {
		t.Fatalf("parse failed on first chunk: %v", err)
	}
	if content != "" || thinking != "" || len(calls) != 0 {
		t.Fatalf("expected no output for first chunk, got content=%q thinking=%q calls=%d", content, thinking, len(calls))
	}

	content, thinking, calls, err = parser.Add("nk>Let me think...</think>Answer.", true)
	if err != nil {
		t.Fatalf("parse failed on second chunk: %v", err)
	}
	if thinking != "Let me think..." {
		t.Fatalf("expected thinking %q, got %q", "Let me think...", thinking)
	}
	if content != "Answer." {
		t.Fatalf("expected content %q, got %q", "Answer.", content)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls, got %d", len(calls))
	}
}

func TestQwen3ParserThinkingDisabled(t *testing.T) {
	parser := &Qwen3Parser{hasThinkingSupport: false, defaultThinking: false}
	parser.Init(nil, nil, &api.ThinkValue{Value: false})

	content, thinking, calls, err := parser.Add("Direct answer", true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	if thinking != "" {
		t.Fatalf("expected no thinking, got %q", thinking)
	}
	if content != "Direct answer" {
		t.Fatalf("expected content %q, got %q", "Direct answer", content)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls, got %d", len(calls))
	}
}

func TestQwen3ParserNilThinkDefaultsToContentForInstructParser(t *testing.T) {
	parser := &Qwen3Parser{hasThinkingSupport: false, defaultThinking: false}
	parser.Init(nil, nil, nil)

	content, thinking, calls, err := parser.Add("Direct answer", true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	if thinking != "" {
		t.Fatalf("expected no thinking, got %q", thinking)
	}
	if content != "Direct answer" {
		t.Fatalf("expected content %q, got %q", "Direct answer", content)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls, got %d", len(calls))
	}
}

func TestQwen3ParserToolCall(t *testing.T) {
	parser := &Qwen3Parser{hasThinkingSupport: false, defaultThinking: false}
	parser.Init(nil, nil, &api.ThinkValue{Value: false})

	input := "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"location\":\"San Francisco\",\"unit\":\"celsius\"}}</tool_call>"
	content, thinking, calls, err := parser.Add(input, true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	if content != "" {
		t.Fatalf("expected empty content, got %q", content)
	}
	if thinking != "" {
		t.Fatalf("expected empty thinking, got %q", thinking)
	}
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(calls))
	}
	if calls[0].Function.Name != "get_weather" {
		t.Fatalf("expected tool name %q, got %q", "get_weather", calls[0].Function.Name)
	}

	location, ok := calls[0].Function.Arguments.Get("location")
	if !ok || location != "San Francisco" {
		t.Fatalf("expected location %q, got %v", "San Francisco", location)
	}
	unit, ok := calls[0].Function.Arguments.Get("unit")
	if !ok || unit != "celsius" {
		t.Fatalf("expected unit %q, got %v", "celsius", unit)
	}
}

func TestQwen3ParserThinkingWithToolCallBeforeThinkingClose(t *testing.T) {
	parser := &Qwen3Parser{hasThinkingSupport: true, defaultThinking: true}
	parser.Init(nil, nil, &api.ThinkValue{Value: true})

	input := "Let me think<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"location\":\"San Francisco\",\"unit\":\"celsius\"}}</tool_call>"
	content, thinking, calls, err := parser.Add(input, true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	if content != "" {
		t.Fatalf("expected empty content, got %q", content)
	}
	if thinking != "Let me think" {
		t.Fatalf("expected thinking %q, got %q", "Let me think", thinking)
	}
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(calls))
	}
	if calls[0].Function.Name != "get_weather" {
		t.Fatalf("expected tool name %q, got %q", "get_weather", calls[0].Function.Name)
	}
}

func TestQwen3ParserThinkingWithSplitToolOpenTag(t *testing.T) {
	parser := &Qwen3Parser{hasThinkingSupport: true, defaultThinking: true}
	parser.Init(nil, nil, &api.ThinkValue{Value: true})

	content, thinking, calls, err := parser.Add("Let me think<tool_ca", false)
	if err != nil {
		t.Fatalf("parse failed on first chunk: %v", err)
	}
	if content != "" || thinking != "Let me think" || len(calls) != 0 {
		t.Fatalf(
			"expected content=%q thinking=%q calls=%d, got content=%q thinking=%q calls=%d",
			"",
			"Let me think",
			0,
			content,
			thinking,
			len(calls),
		)
	}

	content, thinking, calls, err = parser.Add("ll>{\"name\":\"get_weather\",\"arguments\":{\"location\":\"SF\"}}</tool_call>", true)
	if err != nil {
		t.Fatalf("parse failed on second chunk: %v", err)
	}
	if content != "" {
		t.Fatalf("expected empty content, got %q", content)
	}
	if thinking != "" {
		t.Fatalf("expected no additional thinking on second chunk, got %q", thinking)
	}
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(calls))
	}
	if calls[0].Function.Name != "get_weather" {
		t.Fatalf("expected tool name %q, got %q", "get_weather", calls[0].Function.Name)
	}
}

func TestQwen35ParserRespectsNoThink(t *testing.T) {
	parser := ParserForName("qwen3.5")
	if parser == nil {
		t.Fatal("expected qwen3.5 parser")
	}

	parser.Init(nil, nil, &api.ThinkValue{Value: false})
	content, thinking, calls, err := parser.Add("Hello! How can I help you today?", true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	if thinking != "" {
		t.Fatalf("expected no thinking, got %q", thinking)
	}
	if content != "Hello! How can I help you today?" {
		t.Fatalf("expected content %q, got %q", "Hello! How can I help you today?", content)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls, got %d", len(calls))
	}
}
