package parsers

// Tests for wrapper-less <function= tool calls (missing <tool_call> opener).

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func bareTestTools() []api.Tool {
	prop := api.NewToolPropertiesMap()
	prop.Set("cmd", api.ToolProperty{Type: api.PropertyType{"string"}})
	return []api.Tool{{
		Type: "function",
		Function: api.ToolFunction{
			Name: "exec_command",
			Parameters: api.ToolFunctionParameters{
				Type:       "object",
				Properties: prop,
			},
		},
	}}
}

const bareCall = "<function=exec_command>\n<parameter=cmd>\nls -la\n</parameter>\n</function>"

func TestQwen3CoderBareFunctionCallIsParsed(t *testing.T) {
	p := &Qwen3CoderParser{}
	p.Init(bareTestTools(), nil, nil)
	content, _, calls, err := p.Add(bareCall, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call, got %d (content=%q)", len(calls), content)
	}
	if calls[0].Function.Name != "exec_command" {
		t.Fatalf("wrong function name: %q", calls[0].Function.Name)
	}
	if got, _ := calls[0].Function.Arguments.Get("cmd"); got != "ls -la" {
		t.Fatalf("wrong cmd argument: %v", got)
	}
	if content != "" {
		t.Fatalf("tool call leaked into content: %q", content)
	}
}

func TestQwen3CoderBareCallSwallowsOrphanCloseTag(t *testing.T) {
	p := &Qwen3CoderParser{}
	p.Init(bareTestTools(), nil, nil)
	content, _, calls, err := p.Add(bareCall+"\n</tool_call>\ndone", true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(calls))
	}
	if content != "done" {
		t.Fatalf("expected trailing content %q, got %q", "done", content)
	}
}

func TestQwen3CoderBareCallStreamedInChunks(t *testing.T) {
	p := &Qwen3CoderParser{}
	p.Init(bareTestTools(), nil, nil)
	var calls []api.ToolCall
	var content string
	for i := 0; i < len(bareCall); i += 7 {
		end := min(i+7, len(bareCall))
		c, _, tc, err := p.Add(bareCall[i:end], end == len(bareCall))
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		content += c
		calls = append(calls, tc...)
	}
	if len(calls) != 1 || calls[0].Function.Name != "exec_command" {
		t.Fatalf("streamed bare call not parsed: calls=%v content=%q", calls, content)
	}
}

func TestQwen3CoderWrappedCallStillWorks(t *testing.T) {
	p := &Qwen3CoderParser{}
	p.Init(bareTestTools(), nil, nil)
	content, _, calls, err := p.Add("<tool_call>\n"+bareCall+"\n</tool_call>", true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(calls) != 1 || calls[0].Function.Name != "exec_command" {
		t.Fatalf("wrapped call regressed: calls=%v content=%q", calls, content)
	}
}

func TestQwen3CoderProseWithUnclosedFunctionTagFlushesAsContent(t *testing.T) {
	p := &Qwen3CoderParser{}
	p.Init(bareTestTools(), nil, nil)
	prose := "the tag <function=foo is documented here"
	content, _, calls, err := p.Add(prose, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(calls) != 0 {
		t.Fatalf("prose misparsed as tool call: %v", calls)
	}
	// upstream trims whitespace immediately before a (suspected) tool call, so
	// one space may be lost on downgrade — require the words, not the exact byte
	if content != prose && content != "the tag<function=foo is documented here" {
		t.Fatalf("prose lost: got %q want %q", content, prose)
	}
}

func TestQwen3CoderBareGarbagePayloadDowngradesToContent(t *testing.T) {
	p := &Qwen3CoderParser{}
	p.Init(bareTestTools(), nil, nil)
	garbage := "<function=<<not xml>></function>"
	content, _, calls, err := p.Add(garbage, true)
	if err != nil {
		t.Fatalf("expected downgrade, got error: %v", err)
	}
	if len(calls) != 0 {
		t.Fatalf("garbage parsed as call: %v", calls)
	}
	if content == "" {
		t.Fatalf("garbage payload dropped instead of downgraded")
	}
}
