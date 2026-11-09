package parsers

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func FuzzLFM3AudioParser(f *testing.F) {
	for _, seed := range []struct {
		raw   string
		think int
	}{
		{"The quick brown fox.", 0},
		{"<think>Add them.</think>\n\n42", 0},
		{"\n37", 1},
		{"Reasoning.</think>Answer", 2},
		{"<think>Need weather.</think>\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>", 0},
	} {
		f.Add(seed.raw, seed.think)
	}
	tools := []api.Tool{{Function: api.ToolFunction{Name: "get_weather"}}}
	f.Fuzz(func(t *testing.T, raw string, think int) {
		if len(raw) > 4096 {
			t.Skip()
		}
		var value *api.ThinkValue
		switch think % 3 {
		case 1:
			value = &api.ThinkValue{Value: false}
		case 2:
			value = &api.ThinkValue{Value: true}
		}
		p := ParserForName("lfm3-audio")
		p.Init(tools, nil, value)
		split := len(raw) / 2
		_, t1, _, _ := p.Add(raw[:split], false)
		_, t2, _, _ := p.Add(raw[split:], true)
		// With thinking off nothing is returned as thinking. Tag removal is
		// covered by TestLFM3AudioParserEverySplit on token-level outputs;
		// arbitrary text can splice a tag out of plain characters around a
		// removed one, which a model emitting special tokens does not produce.
		if value != nil && !value.Bool() && t1+t2 != "" {
			t.Fatalf("thinking %q with thinking off", t1+t2)
		}
	})
}

func TestLFM3AudioParserEverySplit(t *testing.T) {
	tools := []api.Tool{{Function: api.ToolFunction{Name: "get_weather"}}}
	off := &api.ThinkValue{Value: false}
	on := &api.ThinkValue{Value: true}
	prefill := &api.Message{Role: "assistant", Content: "The answer is"}
	for _, tc := range []struct {
		name, input, content, thinking string
		think                          *api.ThinkValue
		last                           *api.Message
		call                           bool
	}{
		{name: "plain transcript", input: "The quick brown fox.", content: "The quick brown fox."},
		{name: "leading newline", input: "\nParis", content: "Paris"},
		{name: "optional thinking", input: "<think>Add them.</think>\n\n42", content: "42", thinking: "Add them."},
		{name: "thinking off prefill", input: "\n37", content: "37", think: off},
		{name: "thinking off explicit reasoning", input: "<think>Add them.</think>\n\n42", content: "42", think: off},
		{name: "thinking off stray close tag", input: "Left side.</think>Right side.", content: "Left side.\n\nRight side.", think: off},
		{name: "think-like text", input: "<thinker> rules", content: "<thinker> rules"},
		{name: "thinking on prefill", input: "Reasoning.</think>Answer", content: "Answer", thinking: "Reasoning.", think: on},
		{name: "continued assistant", input: " 42.", content: " 42.", think: on, last: prefill},
		{name: "continued assistant with think-like text", input: " <think> is a tag", content: " <think> is a tag", last: prefill},
		{name: "thinking then tool", input: "<think>Need weather.</think>\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>", thinking: "Need weather.", call: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for split := 0; split <= len(tc.input); split++ {
				p := ParserForName("lfm3-audio")
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
				if content.String() != tc.content || thinking.String() != tc.thinking {
					t.Fatalf("split %d: content=%q thinking=%q", split, content.String(), thinking.String())
				}
				if tc.call && (len(calls) != 1 || calls[0].Function.Name != "get_weather") {
					t.Fatalf("split %d: calls=%+v", split, calls)
				}
			}
		})
	}
}
