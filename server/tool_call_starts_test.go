package server

import (
	"slices"
	"testing"

	"github.com/ollama/ollama/model/parsers"
	"github.com/ollama/ollama/tools"
)

func TestToolCallStartsForCompletion(t *testing.T) {
	qwen35 := parsers.ParserForName("qwen3.5")
	tagParser := tools.NewParserWithTag(nil, "<tool>")
	jsonParser := tools.NewParserWithTag(nil, "{")

	tests := []struct {
		name          string
		builtinParser parsers.Parser
		toolParser    *tools.Parser
		hasTools      bool
		want          []string
	}{
		{name: "no tools", builtinParser: qwen35},
		{name: "builtin parser", builtinParser: qwen35, hasTools: true, want: []string{"<tool_call>"}},
		{name: "generic parser", toolParser: tagParser, hasTools: true, want: []string{"<tool>"}},
		{name: "ambiguous json tag", toolParser: jsonParser, hasTools: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := toolCallStartsForCompletion(tt.builtinParser, tt.toolParser, tt.hasTools)
			if !slices.Equal(got, tt.want) {
				t.Fatalf("toolCallStartsForCompletion() = %q, want %q", got, tt.want)
			}
		})
	}
}
