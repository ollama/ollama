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

func TestQwen3ParserToolCallIndexing(t *testing.T) {
	parser := &Qwen3Parser{hasThinkingSupport: false, defaultThinking: false}
	parser.Init(nil, nil, &api.ThinkValue{Value: false})

	input := `<tool_call>{"name":"first","arguments":{"a":"1"}}</tool_call>
<tool_call>{"name":"second","arguments":{"b":"2"}}</tool_call>
<tool_call>{"name":"third","arguments":{"c":"3"}}</tool_call>`
	_, _, calls, err := parser.Add(input, true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	want := []api.ToolCall{
		{Function: api.ToolCallFunction{Name: "first", Arguments: args(`{"a":"1"}`), Index: 0}},
		{Function: api.ToolCallFunction{Name: "second", Arguments: args(`{"b":"2"}`), Index: 1}},
		{Function: api.ToolCallFunction{Name: "third", Arguments: args(`{"c":"3"}`), Index: 2}},
	}
	if len(calls) != len(want) {
		t.Fatalf("expected %d calls, got %d", len(want), len(calls))
	}
	for i := range want {
		if !toolCallEqual(calls[i], want[i]) {
			t.Fatalf("call %d mismatch: got %#v, want %#v", i, calls[i], want[i])
		}
	}
}

func TestQwen3ParserToolCallIndexingStreaming(t *testing.T) {
	parser := &Qwen3Parser{hasThinkingSupport: false, defaultThinking: false}
	parser.Init(nil, nil, &api.ThinkValue{Value: false})

	var all []api.ToolCall

	_, _, calls, err := parser.Add(`<tool_call>{"name":"first","arguments":{"a":"1"}}</tool_call><tool_call>{"name":"second","arguments":{"b":"2"}`, false)
	if err != nil {
		t.Fatalf("step 1 parse failed: %v", err)
	}
	all = append(all, calls...)

	_, _, calls, err = parser.Add(`}</tool_call><tool_call>{"name":"third","arguments":{"c":"3"}}</tool_call>`, true)
	if err != nil {
		t.Fatalf("step 2 parse failed: %v", err)
	}
	all = append(all, calls...)

	want := []api.ToolCall{
		{Function: api.ToolCallFunction{Name: "first", Arguments: args(`{"a":"1"}`), Index: 0}},
		{Function: api.ToolCallFunction{Name: "second", Arguments: args(`{"b":"2"}`), Index: 1}},
		{Function: api.ToolCallFunction{Name: "third", Arguments: args(`{"c":"3"}`), Index: 2}},
	}
	if len(all) != len(want) {
		t.Fatalf("expected %d calls, got %d", len(want), len(all))
	}
	for i := range want {
		if !toolCallEqual(all[i], want[i]) {
			t.Fatalf("call %d mismatch: got %#v, want %#v", i, all[i], want[i])
		}
	}
}

func TestQwen3ParserToolCallIndexResetOnInit(t *testing.T) {
	parser := &Qwen3Parser{hasThinkingSupport: false, defaultThinking: false}
	parser.Init(nil, nil, &api.ThinkValue{Value: false})

	_, _, _, err := parser.Add(`<tool_call>{"name":"first","arguments":{"a":"1"}}</tool_call>`, true)
	if err != nil {
		t.Fatalf("first parse failed: %v", err)
	}

	parser.Init(nil, nil, &api.ThinkValue{Value: false})
	_, _, calls, err := parser.Add(`<tool_call>{"name":"second","arguments":{"b":"2"}}</tool_call>`, true)
	if err != nil {
		t.Fatalf("second parse failed: %v", err)
	}

	want := api.ToolCall{
		Function: api.ToolCallFunction{Name: "second", Arguments: args(`{"b":"2"}`), Index: 0},
	}
	if len(calls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(calls))
	}
	if !toolCallEqual(calls[0], want) {
		t.Fatalf("got %#v, want %#v", calls[0], want)
	}
}
