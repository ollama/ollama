package parsers

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestLFM3ParserStreamingToolCall(t *testing.T) {
	p := ParserForName("lfm3")
	if p == nil || !p.HasThinkingSupport() || !p.HasToolSupport() {
		t.Fatal("LFM3 parser must support thinking and tool calls")
	}
	tools := []api.Tool{{Function: api.ToolFunction{Name: "get_weather"}}}
	p.Init(tools, nil, &api.ThinkValue{Value: true})
	var content, thinking strings.Builder
	var calls []api.ToolCall
	chunks := []string{
		"<thi", "nk>Checking weather.</th", "ink>Using a tool.",
		"<tool_ca", "ll>\n<function=get_weather>\n<parameter=city>\nPar",
		"is\n</parameter>\n</function>\n</tool_call>",
	}
	for i, chunk := range chunks {
		c, thought, gotCalls, err := p.Add(chunk, i == len(chunks)-1)
		if err != nil {
			t.Fatal(err)
		}
		content.WriteString(c)
		thinking.WriteString(thought)
		calls = append(calls, gotCalls...)
	}
	if got := thinking.String(); got != "Checking weather." {
		t.Errorf("thinking = %q", got)
	}
	if got := content.String(); got != "Using a tool." {
		t.Errorf("content = %q", got)
	}
	if len(calls) != 1 || calls[0].Function.Name != "get_weather" {
		t.Fatalf("tool calls = %+v", calls)
	}
	if city, ok := calls[0].Function.Arguments.Get("city"); !ok || city != "Paris" {
		t.Errorf("city = %v, present %v", city, ok)
	}
}

func TestLFM3ParserEverySplitInReasoningAndToolCall(t *testing.T) {
	input := "<think>Check the result.</think>Answer. " +
		"<tool_call><function=get_weather><parameter=city>\nParis\n</parameter></function></tool_call>"
	tools := []api.Tool{{Function: api.ToolFunction{Name: "get_weather"}}}
	for split := 1; split < len(input); split++ {
		p := ParserForName("lfm3")
		p.Init(tools, nil, &api.ThinkValue{Value: true})
		var content, thinking strings.Builder
		var calls []api.ToolCall
		for part, chunk := range []string{input[:split], input[split:]} {
			c, thought, gotCalls, err := p.Add(chunk, part == 1)
			if err != nil {
				t.Fatalf("split %d part %d: %v", split, part, err)
			}
			content.WriteString(c)
			thinking.WriteString(thought)
			calls = append(calls, gotCalls...)
		}
		if thinking.String() != "Check the result." || content.String() != "Answer." {
			t.Errorf("split %d: thinking=%q content=%q", split, thinking.String(), content.String())
		}
		if len(calls) != 1 || calls[0].Function.Name != "get_weather" {
			t.Errorf("split %d: calls=%+v", split, calls)
			continue
		}
		if city, ok := calls[0].Function.Arguments.Get("city"); !ok || city != "Paris" {
			t.Errorf("split %d: city=%v present=%v", split, city, ok)
		}
	}
}

// Each case is a reply shape the LFM3 templates' prompt ending implies: the
// <think> prefill (thinking on), <think></think> (thinking off or Omni), and
// a continued trailing assistant message.
func TestLFM3ParsersFollowGenerationPrompt(t *testing.T) {
	tools := []api.Tool{{Function: api.ToolFunction{Name: "get_weather"}}}
	call := "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"
	on, off := &api.ThinkValue{Value: true}, &api.ThinkValue{Value: false}
	user := &api.Message{Role: "user", Content: "Q"}
	toolResult := &api.Message{Role: "tool", Content: "19C"}
	prefill := &api.Message{Role: "assistant", Content: "The answer is"}
	for _, tc := range []struct {
		parser, name, input, content, thinking string
		think                                  *api.ThinkValue
		last                                   *api.Message
		call                                   bool
	}{
		{parser: "lfm3", name: "default thinking", input: "Add.</think>\n\n4", thinking: "Add.", content: "4", last: user},
		{parser: "lfm3", name: "thinking on", input: "Add.</think>4", thinking: "Add.", content: "4", think: on, last: user},
		{parser: "lfm3", name: "thinking on tool call", input: "Need weather.</think>\n" + call, thinking: "Need weather.", think: on, last: user, call: true},
		{parser: "lfm3", name: "thinking on after tool result", input: "Got it.</think>It is 19C.", thinking: "Got it.", content: "It is 19C.", think: on, last: toolResult},
		{parser: "lfm3", name: "thinking off", input: "4", content: "4", think: off, last: user},
		{parser: "lfm3", name: "thinking off tool call", input: call, think: off, last: user, call: true},
		{parser: "lfm3", name: "continued assistant", input: " 42.", content: " 42.", last: prefill},
		{parser: "lfm3-omni", name: "answer", input: "4", content: "4", last: user},
		{parser: "lfm3-omni", name: "tool call", input: "Checking.\n" + call, content: "Checking.", last: user, call: true},
	} {
		t.Run(tc.parser+"/"+tc.name, func(t *testing.T) {
			for split := 0; split <= len(tc.input); split++ {
				p := ParserForName(tc.parser)
				p.Init(tools, tc.last, tc.think)
				var content, thinking strings.Builder
				var calls []api.ToolCall
				for i, chunk := range []string{tc.input[:split], tc.input[split:]} {
					c, th, cl, err := p.Add(chunk, i == 1)
					if err != nil {
						t.Fatalf("split %d: %v", split, err)
					}
					content.WriteString(c)
					thinking.WriteString(th)
					calls = append(calls, cl...)
				}
				if strings.TrimSpace(content.String()) != strings.TrimSpace(tc.content) || thinking.String() != tc.thinking {
					t.Fatalf("split %d: content=%q thinking=%q", split, content.String(), thinking.String())
				}
				if tc.call != (len(calls) == 1) || (tc.call && calls[0].Function.Name != "get_weather") {
					t.Fatalf("split %d: calls=%+v", split, calls)
				}
			}
		})
	}
}
