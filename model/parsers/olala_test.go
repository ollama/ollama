package parsers

import (
	"testing"

	"github.com/ollama/ollama/api"
)

type olalaResult struct {
	content, thinking string
	calls             []api.ToolCall
}

func olalaRun(t *testing.T, think *api.ThinkValue, last *api.Message, chunks []string) olalaResult {
	t.Helper()
	p := ParserForName("olala")
	if p == nil {
		t.Fatal("olala parser not registered")
	}
	p.Init(nil, last, think)
	var r olalaResult
	for i, c := range chunks {
		content, thinking, calls, err := p.Add(c, i == len(chunks)-1)
		if err != nil {
			t.Fatal(err)
		}
		r.content += content
		r.thinking += thinking
		r.calls = append(r.calls, calls...)
	}
	return r
}

// split into single bytes to exercise partial markers
func olalaBytes(s string) []string {
	out := make([]string, 0, len(s)+1)
	for i := range len(s) {
		out = append(out, s[i:i+1])
	}
	return append(out, "")
}

func TestOlalaParser(t *testing.T) {
	cases := []struct {
		name         string
		think        *api.ThinkValue
		last         *api.Message
		output       string
		wantContent  string
		wantThinking string
		wantCalls    []string // name:args JSON
		wantIDs      []string
	}{
		{
			name:         "thinking and answer",
			output:       "\nLet me think.\n<|channel_end|><|channel_start|>final<|content|>Hello there!<|channel_end|><|im_end|>",
			wantThinking: "Let me think.",
			wantContent:  "Hello there!",
		},
		{
			name:        "disabled",
			think:       &api.ThinkValue{Value: false},
			output:      "<|channel_start|>final<|content|>Bonjour.<|channel_end|><|im_end|>",
			wantContent: "Bonjour.",
		},
		{
			name:         "tool call",
			think:        &api.ThinkValue{Value: "low"},
			output:       "Need weather.<|channel_end|><|channel_start|>final<|content|><|channel_end|><|channel_start|>tools<|content|><toolcalls><call id=\"call_0\"><name>get_weather</name><arguments>{\"city\": \"Lyon\"}</arguments></call><call id=\"call_1\"><name>get_weather</name><arguments>{\"city\": \"Paris\", \"unit\": \"c\"}</arguments></call></toolcalls><|channel_end|><|im_end|>",
			wantThinking: "Need weather.",
			wantCalls:    []string{`get_weather:{"city":"Lyon"}`, `get_weather:{"city":"Paris","unit":"c"}`},
			wantIDs:      []string{"call_0", "call_1"},
		},
		{
			name:         "missing final closer before tools",
			output:       "t<|channel_end|><|channel_start|>final<|content|>Checking.<|channel_start|>tools<|content|>\n<toolcalls>\n  <call id=\"a\">\n    <name>f</name>\n    <arguments>{}</arguments>\n  </call>\n</toolcalls>",
			wantThinking: "t",
			wantContent:  "Checking.",
			wantCalls:    []string{`f:{}`},
			wantIDs:      []string{"a"},
		},
		{
			name:         "unknown channel dropped",
			output:       "x<|channel_end|><|channel_start|>final<|content|>Hi<|channel_end|><|channel_start|>notes<|content|>secret<|channel_end|><|im_end|>",
			wantThinking: "x",
			wantContent:  "Hi",
		},
		{
			name:         "truncated thinking",
			output:       "still thinking about <|chan",
			wantThinking: "still thinking about <|chan",
		},
		{
			name:        "prefill content continues final",
			last:        &api.Message{Role: "assistant", Thinking: "t", Content: "Hel"},
			output:      "lo!<|channel_end|><|im_end|>",
			wantContent: "lo!",
		},
	}

	for _, tc := range cases {
		for _, mode := range []string{"whole", "bytes"} {
			t.Run(tc.name+"/"+mode, func(t *testing.T) {
				chunks := []string{tc.output}
				if mode == "bytes" {
					chunks = olalaBytes(tc.output)
				} else {
					chunks = append(chunks, "")
				}
				r := olalaRun(t, tc.think, tc.last, chunks)
				if r.content != tc.wantContent {
					t.Errorf("content = %q, want %q", r.content, tc.wantContent)
				}
				if r.thinking != tc.wantThinking {
					t.Errorf("thinking = %q, want %q", r.thinking, tc.wantThinking)
				}
				if len(r.calls) != len(tc.wantCalls) {
					t.Fatalf("calls = %+v, want %v", r.calls, tc.wantCalls)
				}
				for i, c := range r.calls {
					got := c.Function.Name + ":" + c.Function.Arguments.String()
					if got != tc.wantCalls[i] {
						t.Errorf("call %d = %s, want %s", i, got, tc.wantCalls[i])
					}
					if c.ID != tc.wantIDs[i] || c.Function.Index != i {
						t.Errorf("call %d id/index = %s/%d", i, c.ID, c.Function.Index)
					}
				}
			})
		}
	}
}

func TestOlalaParserThinkingClose(t *testing.T) {
	p := ParserForName("olala")
	p.Init(nil, nil, nil)
	if got := p.ThinkingClose(); len(got) != 1 || got[0] != "<|channel_end|>" {
		t.Fatalf("ThinkingClose = %v", got)
	}
	p.Init(nil, nil, &api.ThinkValue{Value: false})
	if got := p.ThinkingClose(); got != nil {
		t.Fatalf("ThinkingClose with thinking off = %v", got)
	}
}
