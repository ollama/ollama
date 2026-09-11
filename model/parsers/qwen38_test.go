package parsers

import (
	"strconv"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/ollama/ollama/api"
)

func TestQwen38ParserReferenceContinuationAtEverySplit(t *testing.T) {
	tools := []api.Tool{
		tool("get_weather", map[string]api.ToolProperty{
			"city": {Type: api.PropertyType{"string"}},
		}),
		tool("set_uv_threshold", map[string]api.ToolProperty{
			"index": {Type: api.PropertyType{"integer"}},
		}),
	}
	continuation := `<think>
Need weather and UV for Montréal.
</think>

I'll check both.

<tool_call>
<function=get_weather>
<parameter=city>
Montréal
</parameter>
</function>
</tool_call>
<tool_call>
<function=set_uv_threshold>
<parameter=index>
5
</parameter>
</function>
</tool_call>`

	assertResult := func(t *testing.T, chunks []string) {
		t.Helper()
		var content, thinking strings.Builder
		var calls []api.ToolCall

		parser := ParserForName("qwen3.5")
		parser.Init(tools, nil, &api.ThinkValue{Value: true})
		for i, chunk := range chunks {
			chunkContent, chunkThinking, chunkCalls, err := parser.Add(chunk, i == len(chunks)-1)
			if err != nil {
				t.Fatalf("chunk %d parse failed: %v", i, err)
			}
			content.WriteString(chunkContent)
			thinking.WriteString(chunkThinking)
			calls = append(calls, chunkCalls...)
		}

		if got, want := thinking.String(), "Need weather and UV for Montréal."; got != want {
			t.Fatalf("thinking = %q, want %q", got, want)
		}
		if got, want := content.String(), "I'll check both."; got != want {
			t.Fatalf("content = %q, want %q", got, want)
		}
		if len(calls) != 2 {
			t.Fatalf("tool calls = %d, want 2", len(calls))
		}
		if got, want := calls[0].Function.Name, "get_weather"; got != want {
			t.Fatalf("first tool name = %q, want %q", got, want)
		}
		if got, ok := calls[0].Function.Arguments.Get("city"); !ok || got != "Montréal" {
			t.Fatalf("city argument = %#v, %v; want %q, true", got, ok, "Montréal")
		}
		if got, want := calls[1].Function.Name, "set_uv_threshold"; got != want {
			t.Fatalf("second tool name = %q, want %q", got, want)
		}
		if got, ok := calls[1].Function.Arguments.Get("index"); !ok || got != 5 {
			t.Fatalf("index argument = %#v, %v; want 5, true", got, ok)
		}
	}

	for split := range len(continuation) + 1 {
		if !utf8.ValidString(continuation[:split]) || !utf8.ValidString(continuation[split:]) {
			continue
		}
		t.Run("split_"+strconv.Itoa(split), func(t *testing.T) {
			assertResult(t, []string{continuation[:split], continuation[split:]})
		})
	}

	var runeChunks []string
	for _, r := range continuation {
		runeChunks = append(runeChunks, string(r))
	}
	t.Run("one_rune_per_chunk", func(t *testing.T) {
		assertResult(t, runeChunks)
	})
}

func TestQwen38ParserMalformedAndControlLikeContent(t *testing.T) {
	tests := []struct {
		name         string
		continuation string
		wantContent  string
		wantThinking string
	}{
		{
			name:         "ordinary control-like text",
			continuation: "Plan</think>ordinary <tool_calling> marker.",
			wantContent:  "ordinary <tool_calling> marker.",
			wantThinking: "Plan",
		},
		{
			name:         "truncated tool call",
			continuation: "Plan</think>Before<tool_call><function=get_weather>",
			wantContent:  "Before",
			wantThinking: "Plan",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := ParserForName("qwen3.5")
			parser.Init(nil, nil, &api.ThinkValue{Value: true})
			content, thinking, calls, err := parser.Add(tt.continuation, true)
			if err != nil {
				t.Fatal(err)
			}
			if content != tt.wantContent || thinking != tt.wantThinking || len(calls) != 0 {
				t.Fatalf("parsed content=%q thinking=%q calls=%d; want content=%q thinking=%q calls=0",
					content, thinking, len(calls), tt.wantContent, tt.wantThinking)
			}
		})
	}
}
