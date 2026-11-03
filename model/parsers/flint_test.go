package parsers

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestFlintParserStreaming(t *testing.T) {
	tools := []api.Tool{{Type: "function", Function: api.ToolFunction{
		Name: "weather", Parameters: api.ToolFunctionParameters{Type: "object", Required: []string{"city"}},
	}}}
	for _, tc := range []struct {
		name, raw, content, thinking string
		calls                        int
	}{
		{"direct", "Hello, Flint!", "Hello, Flint!", "", 0},
		{"reasoned", "<think>consider</think>Now", "Now", "consider", 0},
		{"tool", "<think>consider</think>Now<tool_call>weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>", "Now", "consider", 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for split := range len(tc.raw) + 1 {
				p := ParserForName("flint")
				p.Init(tools, nil, nil)
				content, thinking := "", ""
				var calls []api.ToolCall
				for _, chunk := range []string{tc.raw[:split], tc.raw[split:]} {
					part, thought, newCalls, err := p.Add(chunk, false)
					if err != nil {
						t.Fatalf("split %d: %v", split, err)
					}
					content += part
					thinking += thought
					calls = append(calls, newCalls...)
				}
				part, thought, newCalls, err := p.Add("", true)
				if err != nil {
					t.Fatalf("split %d final: %v", split, err)
				}
				content += part
				thinking += thought
				calls = append(calls, newCalls...)
				if content != tc.content || thinking != tc.thinking || len(calls) != tc.calls {
					t.Fatalf("split %d: content=%q thinking=%q calls=%v", split, content, thinking, calls)
				}
				if len(calls) > 0 {
					if calls[0].Function.Name != "weather" {
						t.Fatalf("tool = %q", calls[0].Function.Name)
					}
					if got, _ := calls[0].Function.Arguments.Get("city"); got != "Paris" {
						t.Fatalf("city = %#v", got)
					}
				}
			}
		})
	}
}

func TestFlintParserExplicitReasoningPrefill(t *testing.T) {
	for _, tc := range []struct {
		name                           string
		think                          bool
		raw, wantContent, wantThinking string
	}{
		{"thinking", true, "consider</think>Tokyo", "Tokyo", "consider"},
		{"direct", false, "Tokyo", "Tokyo", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := ParserForName("flint")
			p.Init(nil, nil, &api.ThinkValue{Value: tc.think})
			content, thinking, _, err := p.Add(tc.raw, true)
			if err != nil {
				t.Fatal(err)
			}
			if content != tc.wantContent || thinking != tc.wantThinking {
				t.Errorf("think=%t: content=%q thinking=%q", tc.think, content, thinking)
			}
		})
	}
}

func TestFlintParserDisabledThinkingBarePrompt(t *testing.T) {
	tools := []api.Tool{{Type: "function", Function: api.ToolFunction{
		Name: "weather", Parameters: api.ToolFunctionParameters{Type: "object", Required: []string{"city"}},
	}}}
	for _, tc := range []struct {
		name, raw, want string
		calls           int
	}{
		{"direct", "Tokyo", "Tokyo", 0},
		{"publisher reasoning", "<think>2 plus 2 is 4</think>4", "4", 0},
		{"publisher reasoning and tool", "<think>get the weather</think><tool_call>weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>", "", 1},
		{"direct tool", "<tool_call>weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>", "", 1},
		{"opened but unfinished", "<think>unfinished reasoning", "", 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for split := range len(tc.raw) + 1 {
				p := ParserForName("flint")
				p.Init(tools, nil, &api.ThinkValue{Value: false})
				var content, thinking string
				var calls []api.ToolCall
				for _, chunk := range []string{tc.raw[:split], tc.raw[split:]} {
					part, thought, newCalls, err := p.Add(chunk, false)
					if err != nil {
						t.Fatalf("split %d: %v", split, err)
					}
					content += part
					thinking += thought
					calls = append(calls, newCalls...)
				}
				part, thought, newCalls, err := p.Add("", true)
				if err != nil {
					t.Fatalf("split %d final: %v", split, err)
				}
				content += part
				thinking += thought
				calls = append(calls, newCalls...)
				if content != tc.want || thinking != "" || len(calls) != tc.calls {
					t.Fatalf("split %d: content=%q thinking=%q calls=%v", split, content, thinking, calls)
				}
				if len(calls) > 0 {
					if calls[0].Function.Name != "weather" {
						t.Fatalf("split %d: tool=%q", split, calls[0].Function.Name)
					}
					if got, _ := calls[0].Function.Arguments.Get("city"); got != "Paris" {
						t.Fatalf("split %d: city=%#v", split, got)
					}
				}
			}
		})
	}
}

func TestFlintParserDisabledThinkingStreamsDirectContent(t *testing.T) {
	p := ParserForName("flint")
	p.Init(nil, nil, &api.ThinkValue{Value: false})
	content, thinking, calls, err := p.Add("Tok", false)
	if err != nil || content != "Tok" || thinking != "" || len(calls) != 0 {
		t.Fatalf("first chunk: content=%q thinking=%q calls=%v err=%v", content, thinking, calls, err)
	}
	content, thinking, calls, err = p.Add("yo", true)
	if err != nil || content != "yo" || thinking != "" || len(calls) != 0 {
		t.Fatalf("final chunk: content=%q thinking=%q calls=%v err=%v", content, thinking, calls, err)
	}
}

func TestFlintParserDisabledThinkingByteChunks(t *testing.T) {
	tools := []api.Tool{{Type: "function", Function: api.ToolFunction{
		Name: "weather", Parameters: api.ToolFunctionParameters{Type: "object", Required: []string{"city"}},
	}}}
	const raw = "<think>Check Paris conditions</think>Clear skies<tool_call>weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>"
	p := ParserForName("flint")
	p.Init(tools, nil, &api.ThinkValue{Value: false})
	var content, thinking string
	var calls []api.ToolCall
	for i := range len(raw) {
		part, thought, newCalls, err := p.Add(raw[i:i+1], false)
		if err != nil {
			t.Fatalf("byte %d: %v", i, err)
		}
		content += part
		thinking += thought
		calls = append(calls, newCalls...)
	}
	part, thought, newCalls, err := p.Add("", true)
	if err != nil {
		t.Fatal(err)
	}
	content += part
	thinking += thought
	calls = append(calls, newCalls...)
	if content != "Clear skies" || thinking != "" || len(calls) != 1 || calls[0].Function.Name != "weather" {
		t.Fatalf("content=%q thinking=%q calls=%v", content, thinking, calls)
	}
}

func FuzzFlintParser(f *testing.F) {
	for _, seed := range []struct {
		raw      string
		disabled bool
	}{
		{"Hello, Flint!", false},
		{"<think>consider</think>Now", false},
		{"<think>reasoning split by the parser</think>Tokyo", true},
		{"<tool_call>weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>", true},
	} {
		f.Add(seed.raw, seed.disabled)
	}
	f.Fuzz(func(t *testing.T, raw string, disabled bool) {
		if len(raw) > 4096 {
			t.Skip()
		}
		var think *api.ThinkValue
		if disabled {
			think = &api.ThinkValue{Value: false}
		}
		parser := ParserForName("flint")
		parser.Init(nil, nil, think)
		split := len(raw) / 2
		_, _, _, _ = parser.Add(raw[:split], false)
		_, _, _, _ = parser.Add(raw[split:], true)
	})
}
