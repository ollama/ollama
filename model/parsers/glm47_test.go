package parsers

import (
	"reflect"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestGLM47ParserAdd(t *testing.T) {
	parser := GLM47Parser{}
	parser.Init([]api.Tool{
		tool("calculate", map[string]api.ToolProperty{
			"count":   {Type: api.PropertyType{"integer"}},
			"enabled": {Type: api.PropertyType{"boolean"}},
		}),
	}, nil, nil)

	// When thinking is enabled (thinkValue nil), the prompt ends with <think>,
	// so the model output does NOT include the opening <think> tag.
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

func TestGLM47ParserNoThinkingContent(t *testing.T) {
	parser := GLM47Parser{}
	parser.Init(nil, nil, nil)

	// When thinking is enabled but model has no thinking to output,
	// it should output </think> immediately followed by content.
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

func TestGLM47ParserThinkingDisabled(t *testing.T) {
	parser := GLM47Parser{}
	// When thinking is disabled, parser stays in LookingForThinkingOpen state
	parser.Init(nil, nil, &api.ThinkValue{Value: false})

	// Model outputs plain content (prompt ended with </think>)
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

func TestGLM47ParserToolCallEscaping(t *testing.T) {
	toolCall, err := parseGLM46ToolCall(glm46EventRawToolCall{raw: `exec
<arg_key>expr</arg_key>
<arg_value>a < b && c > d</arg_value>`}, nil)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	expected := api.ToolCall{
		Function: api.ToolCallFunction{
			Name:      "exec",
			Arguments: args(`{"expr": "a < b && c > d"}`),
		},
	}
	if !reflect.DeepEqual(toolCall, expected) {
		t.Fatalf("expected %#v, got %#v", expected, toolCall)
	}
}
