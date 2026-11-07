package parsers

import (
	"reflect"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func rippleParseChunks(t *testing.T, chunks []string) (string, string, []api.ToolCall, error) {
	t.Helper()
	p := ParserForName("ripple")
	if p == nil {
		t.Fatal("missing ripple parser")
	}
	p.Init(nil, nil, nil)
	var content, thinking strings.Builder
	var calls []api.ToolCall
	for _, chunk := range append(chunks, "") {
		c, r, tc, err := p.Add(chunk, false)
		content.WriteString(c)
		thinking.WriteString(r)
		calls = append(calls, tc...)
		if err != nil {
			return content.String(), thinking.String(), calls, err
		}
	}
	c, r, tc, err := p.Add("", true)
	content.WriteString(c)
	thinking.WriteString(r)
	calls = append(calls, tc...)
	return content.String(), thinking.String(), calls, err
}

func rippleChunkings(text string) [][]string {
	chunks := [][]string{{text}}
	for i := 0; i <= len(text); i++ {
		chunks = append(chunks, []string{text[:i], text[i:]})
	}
	bytes := make([]string, len(text))
	for i := range len(text) {
		bytes[i] = text[i : i+1]
	}
	return append(chunks, bytes)
}

func TestRippleParserThinking(t *testing.T) {
	for _, tt := range []struct{ name, input, content, thinking string }{
		{"plain", " \nhello\n", " \nhello\n", ""},
		{"thinking", "<think>\nreason\n</think>\nanswer", "answer", "reason"},
		{"no structural newlines", "<think>reason</think>answer", "answer", "reason"},
		{"one newline only", "<think>\n</think>\n", "", ""},
		{"meaningful whitespace", "<think>\n\n \treason \n\n</think>\n\n answer \n", "\n answer \n", "\n \treason \n"},
		{"CRLF is not a structural newline", "<think>\r\nreason\r\n</think>\r\nanswer", "\r\nanswer", "\r\nreason\r"},
		{"nonleading opener", " \n<think>reason</think>\nanswer", " \n<think>reason</think>\nanswer", ""},
		{"no opener", "reason\n</think>\nanswer", "reason\n</think>\nanswer", ""},
		{"later tags literal", "<think>r</think>c<think>x</think>", "c<think>x</think>", "r"},
		{"tool inside thinking stays literal", "<think>\n<tool_call>{}</tool_call>\n</think>\nanswer", "answer", "<tool_call>{}</tool_call>"},
		{"truncated opener", "<thin", "<thin", ""},
		{"truncated reasoning", "<think>\nreason\n</thi", "", "reason\n</thi"},
		{"trailing newline drained", "<think>\nreason\n", "", "reason\n"},
		{"opener only", "<think>", "", ""},
		{"opener and newline", "<think>\n", "", ""},
		{"close only", "<think>\nreason\n</think>", "", "reason"},
		{"partial tool text", "text\n<tool_", "text\n<tool_", ""},
		{"unicode", "<think>\n日本語🙂\n</think>\né\n", "é\n", "日本語🙂"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			for _, chunks := range rippleChunkings(tt.input) {
				c, r, calls, err := rippleParseChunks(t, chunks)
				if err != nil || c != tt.content || r != tt.thinking || len(calls) != 0 {
					t.Fatalf("chunks %q: got (%q, %q, %v, %v), want (%q, %q)", chunks, c, r, calls, err, tt.content, tt.thinking)
				}
			}
			p := ParserForName("ripple")
			p.Init(nil, nil, nil)
			c, r, calls, err := p.Add(tt.input, true)
			if err != nil || c != tt.content || r != tt.thinking || len(calls) != 0 {
				t.Fatalf("final chunk: (%q, %q, %v, %v)", c, r, calls, err)
			}
			c, r, calls, err = p.Add("", true)
			if err != nil || c != "" || r != "" || len(calls) != 0 {
				t.Fatal("final drain repeated output")
			}
		})
	}
}

func TestRippleParserThinkingTransitions(t *testing.T) {
	p := ParserForName("ripple")
	p.Init(nil, nil, nil)
	for _, tt := range []struct {
		input, content, thinking string
		close                    bool
	}{
		{"<thi", "", "", false},
		{"nk>", "", "", true},
		{"\nreason\n", "", "reason", true},
		{"</thi", "", "", true},
		{"nk>", "", "", false},
		{"\nanswer", "answer", "", false},
	} {
		c, r, _, err := p.Add(tt.input, false)
		if c != tt.content || r != tt.thinking || err != nil {
			t.Fatalf("%q: (%q, %q, %v)", tt.input, c, r, err)
		}
		if close := p.ThinkingClose(); (len(close) != 0) != tt.close || tt.close && close[0] != "</think>" {
			t.Fatalf("%q: close %v", tt.input, close)
		}
	}
	p.Init(nil, nil, nil)
	c, r, _, err := p.Add("<think>\nfresh\n</think>\nnew", true)
	if c != "new" || r != "fresh" || err != nil {
		t.Fatalf("reset: (%q, %q, %v)", c, r, err)
	}
}

func TestRippleParserTools(t *testing.T) {
	one := "<tool_call>\n" + `{"name": "one", "arguments": {"z": "a\n\"b\"\\<&é", "n": 3, "nested": {"x": true}}}` + "\n</tool_call>"
	two := "<tool_call>\n" + `{"name": "two", "arguments": {}}` + "\n</tool_call>"
	for _, tt := range []struct {
		name, input, content, thinking string
		calls                          []api.ToolCall
	}{
		{"one", one, "", "", []api.ToolCall{{Function: api.ToolCallFunction{Name: "one", Arguments: args(`{"z":"a\n\"b\"\\<&é","n":3,"nested":{"x":true}}`)}}}},
		{"multiple", " \n before \t\n\n" + one + "\n" + two, " \n before \t\n", "", []api.ToolCall{{Function: api.ToolCallFunction{Name: "one", Arguments: args(`{"z":"a\n\"b\"\\<&é","n":3,"nested":{"x":true}}`)}}, {Function: api.ToolCallFunction{Index: 1, Name: "two", Arguments: args(`{}`)}}}},
		{"reasoning and tools", "<think>\n R \n</think>\n\n" + two, "", " R ", []api.ToolCall{{Function: api.ToolCallFunction{Name: "two", Arguments: args(`{}`)}}}},
		{"preserve spaces before call", "  before \t" + two, "  before \t", "", []api.ToolCall{{Function: api.ToolCallFunction{Name: "two", Arguments: args(`{}`)}}}},
		{"drop post call text", two + "ignored \n<think>also text</think>\n" + two + "more ignored", "", "", []api.ToolCall{{Function: api.ToolCallFunction{Name: "two", Arguments: args(`{}`)}}, {Function: api.ToolCallFunction{Index: 1, Name: "two", Arguments: args(`{}`)}}}},
		{"delimiter inside string", `<tool_call>{"name":"two","arguments":{"text":"</tool_call> <tool_call> \"quoted\""}}</tool_call>`, "", "", []api.ToolCall{{Function: api.ToolCallFunction{Name: "two", Arguments: args(`{"text":"</tool_call> <tool_call> \"quoted\""}`)}}}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			for _, chunks := range rippleChunkings(tt.input) {
				c, r, calls, err := rippleParseChunks(t, chunks)
				if err != nil || c != tt.content || r != tt.thinking || len(calls) != len(tt.calls) {
					t.Fatalf("chunks %q: (%q, %q, %v, %v)", chunks, c, r, calls, err)
				}
				for i := range calls {
					if !toolCallEqual(calls[i], tt.calls[i]) {
						t.Fatalf("call %d = %+v, want %+v", i, calls[i], tt.calls[i])
					}
				}
			}
		})
	}
}

func TestRippleParserMalformedCalls(t *testing.T) {
	for _, tt := range []struct{ name, input, errorText string }{
		{"invalid JSON", "<tool_call>{bad}</tool_call>", "JSON"},
		{"missing name", `<tool_call>{"arguments":{}}</tool_call>`, "name"},
		{"empty name", `<tool_call>{"name":"","arguments":{}}</tool_call>`, "name"},
		{"missing arguments", `<tool_call>{"name":"f"}</tool_call>`, "object"},
		{"string arguments", `<tool_call>{"name":"f","arguments":"{}"}</tool_call>`, "object"},
		{"array arguments", `<tool_call>{"name":"f","arguments":[]}</tool_call>`, "object"},
		{"null arguments", `<tool_call>{"name":"f","arguments":null}</tool_call>`, "object"},
		{"JSON array", `<tool_call>[{"name":"f","arguments":{}}]</tool_call>`, "JSON"},
		{"multiple objects", `<tool_call>{"name":"f","arguments":{}}{}</tool_call>`, "JSON"},
		{"truncated call", `<tool_call>{"name":"f","arguments":{}`, "unterminated"},
		{"truncated closer", `<tool_call>{"name":"f","arguments":{}}</tool_`, "unterminated"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			for _, chunks := range rippleChunkings(tt.input) {
				_, _, calls, err := rippleParseChunks(t, chunks)
				if err == nil || !strings.Contains(err.Error(), tt.errorText) || len(calls) != 0 {
					t.Fatalf("chunks %q: calls %v, error %v; want %s", chunks, calls, err, tt.errorText)
				}
			}
		})
	}
}

func TestRippleParserToolNewlineHoldback(t *testing.T) {
	p := ParserForName("ripple")
	p.Init(nil, nil, nil)
	c, _, _, err := p.Add("before\n\n", false)
	if c != "before\n" || err != nil {
		t.Fatalf("first chunk: %q, %v", c, err)
	}
	c, _, _, err = p.Add("<tool_", false)
	if c != "" || err != nil {
		t.Fatalf("partial delimiter: %q, %v", c, err)
	}
	c, _, calls, err := p.Add(`call>{"name":"f","arguments":{}}</tool_call>`, true)
	if c != "" || err != nil || len(calls) != 1 {
		t.Fatalf("call: %q, %v, %v", c, calls, err)
	}
	p.Init(nil, nil, nil)
	_, _, _, _ = p.Add("before\n", false)
	c, _, _, err = p.Add("<tool_", true)
	if c != "\n<tool_" || err != nil {
		t.Fatalf("drain: %q, %v", c, err)
	}
}

func TestRippleParserPlain(t *testing.T) {
	for _, text := range []string{"", "Hello", " \nanswer\n\n", "Reason without opener</think>\nAnswer", " <think>literal</think>", "Text <think>literal</think>", "<thinking>literal", "é 日本語 🙂"} {
		t.Run(text, func(t *testing.T) {
			content, thinking, calls, err := rippleParseChunks(t, []string{text})
			if err != nil || content != text || thinking != "" || len(calls) != 0 {
				t.Fatalf("got (%q, %q, %v, %v), want %q", content, thinking, calls, err, text)
			}
		})
	}
}

func TestRippleParserCapabilities(t *testing.T) {
	p := ParserForName("ripple")
	if p == nil {
		t.Fatal("missing ripple parser")
	}
	if !p.HasToolSupport() || !p.HasThinkingSupport() {
		t.Fatal("missing capabilities")
	}
	if got := p.PreservedTokens(); !reflect.DeepEqual(got, []string{"<think>", "</think>", "<tool_call>", "</tool_call>"}) {
		t.Fatalf("preserved = %v", got)
	}
	tools := []api.Tool{{Type: "function", Function: api.ToolFunction{Name: "f"}}}
	for _, think := range []*api.ThinkValue{nil, {Value: true}, {Value: false}, {Value: "high"}} {
		if got := p.Init(tools, &api.Message{Role: "assistant", Thinking: "old", Content: "old"}, think); !reflect.DeepEqual(got, tools) {
			t.Fatalf("changed tools: %v", got)
		}
		if got := p.ThinkingClose(); len(got) != 0 {
			t.Fatalf("generation prompt does not open reasoning: %v", got)
		}
	}
}
