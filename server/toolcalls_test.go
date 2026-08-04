package server

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestCompleteToolCalls(t *testing.T) {
	// the editor tool from a coding agent: the shape that produced this
	editor := api.Tool{
		Type: "function",
		Function: api.ToolFunction{
			Name: "editor",
			Parameters: api.ToolFunctionParameters{
				Type:     "object",
				Required: []string{"path", "new_text"},
			},
		},
	}
	shell := api.Tool{
		Type: "function",
		Function: api.ToolFunction{
			Name: "shell",
			Parameters: api.ToolFunctionParameters{
				Type:     "object",
				Required: []string{"command"},
			},
		},
	}

	call := func(name string, args ...string) api.ToolCall {
		a := api.NewToolCallFunctionArguments()
		for i := 0; i < len(args); i += 2 {
			a.Set(args[i], args[i+1])
		}
		return api.ToolCall{Function: api.ToolCallFunction{Name: name, Arguments: a}}
	}

	tests := []struct {
		name      string
		toolCalls []api.ToolCall
		truncated bool
		want      int
	}{
		{
			// a 62k-character new_text ran out of tokens before path was
			// written, so the call arrives with two of its four arguments
			name:      "an unfinished call is dropped when the response was cut short",
			toolCalls: []api.ToolCall{call("editor", "insert_line", "48", "new_text", "const SOUND_DATA = {")},
			truncated: true,
			want:      0,
		},
		{
			name:      "a complete call survives a truncated response",
			toolCalls: []api.ToolCall{call("editor", "path", "index.html", "new_text", "hello")},
			truncated: true,
			want:      1,
		},
		{
			// the model's own mistake to report, not something to hide
			name:      "an incomplete call is kept when the response ended normally",
			toolCalls: []api.ToolCall{call("editor", "new_text", "hello")},
			want:      1,
		},
		{
			name: "only the unfinished call in a batch is dropped",
			toolCalls: []api.ToolCall{
				call("shell", "command", "ls"),
				call("editor", "new_text", "hello"),
			},
			truncated: true,
			want:      1,
		},
		{
			// a tool the request never declared has no schema to check against
			name:      "an unknown tool is left alone",
			toolCalls: []api.ToolCall{call("mystery", "anything", "here")},
			truncated: true,
			want:      1,
		},
		{
			name:      "no tool calls is not a special case",
			truncated: true,
			want:      0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := completeToolCalls(tt.toolCalls, []api.Tool{editor, shell}, tt.truncated)
			if len(got) != tt.want {
				t.Fatalf("kept %d tool calls, want %d", len(got), tt.want)
			}
		})
	}
}
