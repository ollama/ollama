package parsers

import (
	"reflect"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestGraniteThinkingParserThinkingSplit(t *testing.T) {
	p := &GraniteThinkingParser{hasThinkingSupport: true}
	p.Init(nil, nil, &api.ThinkValue{Value: true})

	content, thinking, calls, err := p.Add("reasoning here</think>\nhello world", true)
	if err != nil {
		t.Fatalf("Add error: %v", err)
	}
	if thinking != "reasoning here" {
		t.Errorf("thinking = %q, want %q", thinking, "reasoning here")
	}
	if content != "hello world" {
		t.Errorf("content = %q, want %q", content, "hello world")
	}
	if len(calls) != 0 {
		t.Errorf("calls = %v, want none", calls)
	}
}

func TestGraniteThinkingParserThinkingDisabled(t *testing.T) {
	p := &GraniteThinkingParser{hasThinkingSupport: true}
	p.Init(nil, nil, &api.ThinkValue{Value: false})

	content, thinking, _, err := p.Add("hello world", true)
	if err != nil {
		t.Fatalf("Add error: %v", err)
	}
	if thinking != "" {
		t.Errorf("thinking = %q, want empty", thinking)
	}
	if content != "hello world" {
		t.Errorf("content = %q, want %q", content, "hello world")
	}
}

func TestGraniteThinkingParserToolCall(t *testing.T) {
	tools := []api.Tool{{Function: api.ToolFunction{Name: "get_weather"}}}
	p := &GraniteThinkingParser{hasThinkingSupport: true}
	p.Init(tools, nil, &api.ThinkValue{Value: false})

	raw := "<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>\n</tool_call>\n"
	content, thinking, calls, err := p.Add(raw, true)
	if err != nil {
		t.Fatalf("Add error: %v", err)
	}
	if content != "" {
		t.Errorf("content = %q, want empty", content)
	}
	if thinking != "" {
		t.Errorf("thinking = %q, want empty", thinking)
	}
	if len(calls) != 1 {
		t.Fatalf("calls = %v, want 1 call", calls)
	}
	if calls[0].Function.Name != "get_weather" {
		t.Errorf("call name = %q, want get_weather", calls[0].Function.Name)
	}
	got, _ := calls[0].Function.Arguments.Get("location")
	if !reflect.DeepEqual(got, "Paris") {
		t.Errorf("location arg = %v, want Paris", got)
	}
}

func TestGraniteThinkingParserStreamingAcrossChunks(t *testing.T) {
	p := &GraniteThinkingParser{hasThinkingSupport: true}
	p.Init(nil, nil, &api.ThinkValue{Value: true})

	var content, thinking string
	chunks := []string{"partial reason", "ing</thi", "nk>\nfinal ", "content"}
	for i, c := range chunks {
		cc, tt, _, err := p.Add(c, i == len(chunks)-1)
		if err != nil {
			t.Fatalf("Add error: %v", err)
		}
		content += cc
		thinking += tt
	}

	if thinking != "partial reasoning" {
		t.Errorf("thinking = %q, want %q", thinking, "partial reasoning")
	}
	if content != "final content" {
		t.Errorf("content = %q, want %q", content, "final content")
	}
}
