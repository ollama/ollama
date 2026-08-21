package parsers

import (
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/api"
)

func TestBailingParserAdd(t *testing.T) {
	parser := BailingParser{}
	parser.Init([]api.Tool{
		tool("get_weather", map[string]api.ToolProperty{
			"location": {Type: api.PropertyType{"string"}},
		}),
	}, nil, nil)

	// The Bailing prompt ends with <think> when thinking is enabled, so the
	// model output starts directly with thinking content (no opening tag).
	content, thinking, calls, err := parser.Add("plan</think>I'll check.\n<tool_call>get_weather<arg_key>location</arg_key>\n<arg_value>Tokyo</arg_value>\n</tool_call>", true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if thinking != "plan" {
		t.Fatalf("expected thinking 'plan', got %q", thinking)
	}
	if content != "I'll check." {
		t.Fatalf("expected content \"I'll check.\", got %q", content)
	}
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(calls))
	}
	if calls[0].Function.Name != "get_weather" {
		t.Fatalf("expected tool 'get_weather', got %q", calls[0].Function.Name)
	}
	expectedArgs := args(`{"location": "Tokyo"}`)
	if diff := cmp.Diff(expectedArgs, calls[0].Function.Arguments, argsComparer); diff != "" {
		t.Fatalf("tool arguments mismatch (-want +got):\n%s", diff)
	}
}

func TestBailingParserNoThinkingContent(t *testing.T) {
	parser := BailingParser{}
	parser.Init(nil, nil, nil)

	// Thinking enabled but the model emits </think> immediately.
	content, thinking, calls, err := parser.Add("</think>Plain answer", true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if thinking != "" {
		t.Fatalf("expected empty thinking, got %q", thinking)
	}
	if content != "Plain answer" {
		t.Fatalf("expected content 'Plain answer', got %q", content)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls, got %d", len(calls))
	}
}

func TestBailingParserThinkingDisabled(t *testing.T) {
	parser := BailingParser{}
	// The prompt ends with <think></think>, so output is plain content.
	parser.Init(nil, nil, &api.ThinkValue{Value: false})

	content, thinking, calls, err := parser.Add("Plain answer", true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if thinking != "" {
		t.Fatalf("expected empty thinking, got %q", thinking)
	}
	if content != "Plain answer" {
		t.Fatalf("expected content 'Plain answer', got %q", content)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls, got %d", len(calls))
	}
}

func TestBailingParserPrefillContent(t *testing.T) {
	parser := BailingParser{}
	// Prefilled assistant content puts the parser directly in content mode.
	parser.Init(nil, &api.Message{Role: "assistant", Content: "The answer is"}, nil)

	content, thinking, _, err := parser.Add(" 42.", true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if thinking != "" {
		t.Fatalf("expected empty thinking, got %q", thinking)
	}
	if content != " 42." {
		t.Fatalf("expected content ' 42.', got %q", content)
	}
}

func TestBailingParserStreaming(t *testing.T) {
	parser := BailingParser{}
	parser.Init(nil, nil, nil)

	var thinkingSb, contentSb strings.Builder
	for _, chunk := range []string{"consider", " the question</th", "ink>Answer", " here"} {
		content, thinking, _, err := parser.Add(chunk, false)
		if err != nil {
			t.Fatalf("parse failed: %v", err)
		}
		thinkingSb.WriteString(thinking)
		contentSb.WriteString(content)
	}
	content, thinking, _, err := parser.Add("", true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	thinkingSb.WriteString(thinking)
	contentSb.WriteString(content)

	if thinkingSb.String() != "consider the question" {
		t.Fatalf("expected thinking 'consider the question', got %q", thinkingSb.String())
	}
	if contentSb.String() != "Answer here" {
		t.Fatalf("expected content 'Answer here', got %q", contentSb.String())
	}
}

func TestBailingParserPrefillInlineReasoningOnly(t *testing.T) {
	parser := BailingParser{}
	// Content carries only an inline think block; the renderer re-opens the
	// thinking block, so streamed output continues as thinking.
	parser.Init(nil, &api.Message{Role: "assistant", Content: "<think>foo</think>"}, nil)

	content, thinking, _, err := parser.Add("bar</think>answer", true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if thinking != "bar" {
		t.Fatalf("expected thinking 'bar', got %q", thinking)
	}
	if content != "answer" {
		t.Fatalf("expected content 'answer', got %q", content)
	}
}

func TestBailingParserPrefillInlineThinkWithContent(t *testing.T) {
	parser := BailingParser{}
	// The inline think block is closed and followed by content, so the
	// prompt ends mid-content.
	parser.Init(nil, &api.Message{Role: "assistant", Content: "<think>foo</think>partial"}, nil)

	content, thinking, _, err := parser.Add(" answer", true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if thinking != "" {
		t.Fatalf("expected empty thinking, got %q", thinking)
	}
	if content != " answer" {
		t.Fatalf("expected content ' answer', got %q", content)
	}
}

func TestBailingParserPrefillThinkingFieldDisabled(t *testing.T) {
	parser := BailingParser{}
	// With thinking disabled the renderer closes the prefilled think block,
	// so the model continues with content.
	parser.Init(nil, &api.Message{Role: "assistant", Thinking: "already thought"}, &api.ThinkValue{Value: false})

	content, thinking, _, err := parser.Add("answer", true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if thinking != "" {
		t.Fatalf("expected empty thinking, got %q", thinking)
	}
	if content != "answer" {
		t.Fatalf("expected content 'answer', got %q", content)
	}
}
