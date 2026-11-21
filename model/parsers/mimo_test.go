package parsers

import (
	"fmt"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

// MiMo V2.6 uses compact Qwen-style XML tool calls after a think block.
func TestMiMoCompactToolCallParser(t *testing.T) {
	tools := []api.Tool{{
		Type: "function",
		Function: api.ToolFunction{
			Name: "search",
			Parameters: api.ToolFunctionParameters{
				Properties: func() *api.ToolPropertiesMap {
					props := api.NewToolPropertiesMap()
					props.Set("query", api.ToolProperty{Type: api.PropertyType{"string"}})
					return props
				}(),
			},
		},
	}}
	const output = "<think>Need a source</think><tool_call><function=search><parameter=query>weather</parameter></function></tool_call>"
	for _, chunkSize := range []int{1, 3, len(output)} {
		t.Run(fmt.Sprintf("chunk_%d", chunkSize), func(t *testing.T) {
			parser := ParserForName("qwen3.5")
			parser.Init(tools, nil, &api.ThinkValue{Value: true})
			var content, thinking strings.Builder
			var calls []api.ToolCall
			for i := 0; i < len(output); i += chunkSize {
				end := min(i+chunkSize, len(output))
				c, th, nextCalls, err := parser.Add(output[i:end], end == len(output))
				if err != nil {
					t.Fatal(err)
				}
				content.WriteString(c)
				thinking.WriteString(th)
				calls = append(calls, nextCalls...)
			}
			if content.String() != "" || thinking.String() != "Need a source" || len(calls) != 1 {
				t.Fatalf("content=%q thinking=%q calls=%d", content.String(), thinking.String(), len(calls))
			}
			if calls[0].Function.Name != "search" {
				t.Fatalf("tool=%q", calls[0].Function.Name)
			}
			if value, ok := calls[0].Function.Arguments.Get("query"); !ok || value != "weather" {
				t.Fatalf("query=%v present=%v", value, ok)
			}
		})
	}
}
