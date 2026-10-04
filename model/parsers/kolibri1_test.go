package parsers

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestKolibri1Streaming(t *testing.T) {
	for _, size := range []int{1, 3, 13, 1000} {
		p := ParserForName("kolibri1")
		p.Init(nil, nil, nil)
		input := "<think>\nCompute.\n</think>\n\n42."
		var content, thinking strings.Builder
		for offset := 0; offset < len(input); offset += size {
			end := min(offset+size, len(input))
			c, th, _, err := p.Add(input[offset:end], end == len(input))
			if err != nil {
				t.Fatal(err)
			}
			content.WriteString(c)
			thinking.WriteString(th)
		}
		if strings.TrimSpace(content.String()) != "42." || strings.TrimSpace(thinking.String()) != "Compute." {
			t.Fatalf("chunk %d: content %q thinking %q", size, content.String(), thinking.String())
		}
	}
	p := ParserForName("kolibri1")
	p.Init(nil, nil, &api.ThinkValue{Value: false})
	c, th, _, err := p.Add("42.", true)
	if err != nil || c != "42." || th != "" {
		t.Fatalf("thinking off: %q %q %v", c, th, err)
	}
}

func TestKolibri1ToolsAndContinuation(t *testing.T) {
	for _, size := range []int{1, 7, 1000} {
		p := ParserForName("kolibri1")
		p.Init(nil, nil, &api.ThinkValue{Value: false})
		text := `<tool_call>{"name":"weather","arguments":{"city":"Berlin"}}</tool_call>`
		var calls []api.ToolCall
		for i := 0; i < len(text); i += size {
			end := min(i+size, len(text))
			_, _, chunk, err := p.Add(text[i:end], end == len(text))
			if err != nil {
				t.Fatal(err)
			}
			calls = append(calls, chunk...)
		}
		if len(calls) != 1 || calls[0].Function.Name != "weather" || calls[0].Function.Arguments.ToMap()["city"] != "Berlin" {
			t.Fatalf("tools at chunk %d: %+v", size, calls)
		}
	}
	p := ParserForName("kolibri1")
	p.Init(nil, &api.Message{Role: "assistant", Content: "The answer is"}, nil)
	c, th, _, err := p.Add(" 42.", true)
	if err != nil || c != " 42." || th != "" {
		t.Fatalf("continuation: %q %q %v", c, th, err)
	}
}
