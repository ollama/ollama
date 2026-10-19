package parsers

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

// With thinking off, LFM3 can still reason after the <think></think> prefill,
// in an explicit <think> block or closed by a second </think>. The parser must
// keep every thinking tag and explicit reasoning out of the reply, return no
// thinking, and still extract tool calls; text the model writes before a stray
// </think> cannot be recognized while streaming and is not asserted here.
func TestLFM3ParserThinkingOff(t *testing.T) {
	tools := []api.Tool{{Function: api.ToolFunction{Name: "get_weather"}}}
	cases := []struct {
		name, input string
		// want is the exact reply where the output has no stray tag; contains
		// and absent check the rest.
		want     string
		contains []string
		absent   []string
		calls    int
	}{
		{name: "no tag", input: "The Seine runs through Paris.", want: "The Seine runs through Paris."},
		{name: "tag-like text", input: "Use <thin> wrappers.", want: "Use <thin> wrappers."},
		{name: "explicit reasoning block", input: "<think>Count the letters.</think>\n\n10", want: "10"},
		{name: "reasoning block mid reply", input: "Sure. <think>check this</think> Done.", contains: []string{"Sure.", "Done."}, absent: []string{"check this"}},
		{name: "unclosed reasoning block", input: "Answer: 3 <think>but maybe", contains: []string{"Answer: 3"}, absent: []string{"but maybe"}},
		{name: "tool call", input: "Checking.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>", want: "Checking.", calls: 1},
		{name: "tool call after stray tag", input: "Need weather.</think><tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>", absent: []string{"tool_call", "get_weather"}, calls: 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			for split := 0; split <= len(tc.input); split++ {
				p := ParserForName("lfm3")
				p.Init(tools, nil, &api.ThinkValue{Value: false})
				var content, thinking strings.Builder
				var calls []api.ToolCall
				for part, chunk := range []string{tc.input[:split], tc.input[split:]} {
					c, thought, got, err := p.Add(chunk, part == 1)
					if err != nil {
						t.Fatalf("split %d: %v", split, err)
					}
					content.WriteString(c)
					thinking.WriteString(thought)
					calls = append(calls, got...)
				}
				got := strings.TrimSpace(content.String())
				if strings.Contains(got, "<think>") || strings.Contains(got, "</think>") {
					t.Errorf("split %d: thinking tag in content %q", split, got)
				}
				if thinking.Len() != 0 {
					t.Errorf("split %d: thinking %q, want none", split, thinking.String())
				}
				if len(calls) != tc.calls {
					t.Errorf("split %d: %d tool calls, want %d", split, len(calls), tc.calls)
				}
				if tc.want != "" && got != tc.want {
					t.Errorf("split %d: content %q, want %q", split, got, tc.want)
				}
				for _, s := range tc.contains {
					if !strings.Contains(got, s) {
						t.Errorf("split %d: content %q, want it to contain %q", split, got, s)
					}
				}
				for _, s := range tc.absent {
					if strings.Contains(got, s) {
						t.Errorf("split %d: content %q, want no %q", split, got, s)
					}
				}
			}
		})
	}
}

// A stray </think> between text becomes a paragraph break, however the output
// is chunked, so the text on either side is not run together.
func TestLFM3ParserThinkingOffStrayClose(t *testing.T) {
	cases := []struct{ name, input, want string }{
		{name: "between text", input: "Left side.</think>Right side.", want: "Left side.\n\nRight side."},
		{name: "whitespace after tag", input: "Left side.</think>\n\n  Right side.", want: "Left side.\n\nRight side."},
		{name: "repeated tag", input: "Left side.</think>\n</think>Right side.", want: "Left side.\n\nRight side."},
		{name: "explicit block after tag", input: "Left side.</think><think>hidden</think>Right side.", want: "Left side.\n\nRight side."},
		{name: "leading tag", input: "\n</think>Right side.", want: "Right side."},
		{name: "trailing tag", input: "Left side.</think>\n", want: "Left side."},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			check := func(chunks []string) {
				t.Helper()
				p := ParserForName("lfm3")
				p.Init(nil, nil, &api.ThinkValue{Value: false})
				var content strings.Builder
				for i, chunk := range chunks {
					c, thinking, _, err := p.Add(chunk, i == len(chunks)-1)
					if err != nil || thinking != "" {
						t.Fatalf("chunks %q: thinking %q err %v", chunks, thinking, err)
					}
					content.WriteString(c)
				}
				if got := strings.TrimSpace(content.String()); got != tc.want {
					t.Fatalf("chunks %q: content %q, want %q", chunks, got, tc.want)
				}
			}
			for i := 0; i <= len(tc.input); i++ {
				for j := i; j <= len(tc.input); j++ {
					check([]string{tc.input[:i], tc.input[i:j], tc.input[j:]})
				}
			}
			chars := make([]string, len(tc.input))
			for i := range tc.input {
				chars[i] = tc.input[i : i+1]
			}
			check(chars)
		})
	}
}

// Output streams with thinking off; a suffix that could still become a tag is
// the only thing held back.
func TestLFM3ParserThinkingOffStreams(t *testing.T) {
	p := ParserForName("lfm3")
	p.Init(nil, nil, &api.ThinkValue{Value: false})
	if content, _, _, _ := p.Add("The answer is 42", false); content != "The answer is 42" {
		t.Fatalf("first chunk content %q, want it streamed", content)
	}
	if content, _, _, _ := p.Add(" and more </th", false); strings.Contains(content, "</th") || !strings.Contains(content, "and more") {
		t.Fatalf("content %q, want the text before the partial tag and not the tag", content)
	}
}

func TestLFM3ParserThinkingOnUnchanged(t *testing.T) {
	for _, think := range []*api.ThinkValue{nil, {Value: true}} {
		p := ParserForName("lfm3")
		p.Init(nil, nil, think)
		content, thinking, _, err := p.Add("Add them.</think>42", true)
		if err != nil || content != "42" || thinking != "Add them." {
			t.Fatalf("think %v: content %q thinking %q err %v, want 42 and Add them.", think, content, thinking, err)
		}
	}
}
