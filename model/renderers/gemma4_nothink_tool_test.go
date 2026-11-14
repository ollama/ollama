package renderers

import (
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/stretchr/testify/assert"
)

// With thinking explicitly off, the generation prompt after a tool response
// closes the empty thought, for every gemma4 variant. Without it gemma4:26b
// sometimes closes the thought itself with <tool_call|> and stops, which
// reaches the client as an empty message with no tool call.
func TestGemma4ClosesEmptyThoughtAfterToolResponseWhenThinkingOff(t *testing.T) {
	q := `<|"|>`
	messages := []api.Message{
		{Role: "user", Content: "Weather in Lyon?"},
		{Role: "assistant", ToolCalls: []api.ToolCall{{
			Function: api.ToolCallFunction{Name: "get_weather", Arguments: testArgs(map[string]any{"city": "Lyon"})},
		}}},
		{Role: "tool", ToolName: "get_weather", Content: "Sunny"},
	}
	tail := "<|tool_response>response:get_weather{value:" + q + "Sunny" + q + "}<tool_response|>"

	for _, name := range []string{"gemma4", "gemma4-small", "gemma4-large"} {
		t.Run(name+"/think-off", func(t *testing.T) {
			got, err := RenderWithRenderer(name, messages, weatherTool(), &api.ThinkValue{Value: false})
			assert.NoError(t, err)
			assert.True(t, len(got) > 0)
			assert.Contains(t, got, tail+"<|channel>thought\n<channel|>")
		})
		t.Run(name+"/think-on", func(t *testing.T) {
			got, err := RenderWithRenderer(name, messages, weatherTool(), &api.ThinkValue{Value: true})
			assert.NoError(t, err)
			assert.Contains(t, got, tail+"<|channel>thought\n")
			assert.NotContains(t, got, tail+"<|channel>thought\n<channel|>")
		})
		t.Run(name+"/think-unset", func(t *testing.T) {
			got, err := RenderWithRenderer(name, messages, weatherTool(), nil)
			assert.NoError(t, err)
			assert.NotContains(t, got, tail+"<|channel>thought\n<channel|>", "unset keeps the reference rendering")
		})
	}
}
