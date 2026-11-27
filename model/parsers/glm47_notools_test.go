package parsers

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestGLM47ParserKeepsToolMarkupWithoutTools(t *testing.T) {
	markup := `<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>`
	cases := []struct {
		name   string
		think  *api.ThinkValue
		chunks []string
	}{
		{
			name:   "thinking enabled",
			think:  nil,
			chunks: []string{"plan</think>Answer" + markup},
		},
		{
			name:   "thinking disabled",
			think:  &api.ThinkValue{Value: false},
			chunks: []string{"Answer" + markup},
		},
		{
			name:   "tag split across chunks",
			think:  &api.ThinkValue{Value: false},
			chunks: []string{"Answer<tool_", "call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			parser := GLM47Parser{}
			parser.Init(nil, nil, tc.think)
			var content strings.Builder
			var calls []api.ToolCall
			for i, chunk := range tc.chunks {
				part, _, parsedCalls, err := parser.Add(chunk, i == len(tc.chunks)-1)
				if err != nil {
					t.Fatalf("parse failed: %v", err)
				}
				content.WriteString(part)
				calls = append(calls, parsedCalls...)
			}
			if got, want := content.String(), "Answer"+markup; got != want {
				t.Fatalf("tool markup was not preserved verbatim: got %q, want %q", got, want)
			}
			if len(calls) != 0 {
				t.Fatalf("expected no calls without declared tools, got %d: %#v", len(calls), calls)
			}
		})
	}
}
