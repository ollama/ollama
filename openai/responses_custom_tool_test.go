package openai

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
)

func TestResponsesCustomToolHistory(t *testing.T) {
	body := []byte(`{"model":"test","input":[
		{"role":"user","content":"Read both notes."},
		{"type":"function_call","call_id":"ordinary","name":"read_file","arguments":"{\"path\":\"first.txt\"}"},
		{"type":"custom_tool_call","call_id":"freeform","namespace":"workspace","name":"read_note","input":"  Read \"note\" at C:\\notes\n\t雪 <marker>\r\n  "},
		{"type":"custom_tool_call_output","call_id":"freeform","output":[{"type":"input_text","text":"marker\n"},{"type":"input_text","text":"雪  "}]},
		{"type":"function_call_output","call_id":"ordinary","output":"first note"},
		{"role":"assistant","content":"Read both notes."},
		{"role":"user","content":"Continue."}
	]}`)
	want := []api.Message{
		{Role: "user", Content: "Read both notes."},
		{Role: "assistant", ToolCalls: []api.ToolCall{
			{ID: "ordinary", Function: api.ToolCallFunction{Name: "read_file", Arguments: testArgs(map[string]any{"path": "first.txt"})}},
			{ID: "freeform", Function: api.ToolCallFunction{Name: "workspace.read_note", Arguments: testArgs(map[string]any{"input": "  Read \"note\" at C:\\notes\n\t雪 <marker>\r\n  "})}},
		}},
		{Role: "tool", ToolCallID: "freeform", Content: "marker\n雪  "},
		{Role: "tool", ToolCallID: "ordinary", Content: "first note"},
		{Role: "assistant", Content: "Read both notes."},
		{Role: "user", Content: "Continue."},
	}
	var request ResponsesRequest
	if err := json.Unmarshal(body, &request); err != nil {
		t.Fatal(err)
	}
	chat, err := FromResponsesRequest(request)
	if err != nil {
		t.Fatal(err)
	}
	if diff := cmp.Diff(want, chat.Messages, argsComparer); diff != "" {
		t.Fatalf("tool history changed (-want +got):\n%s", diff)
	}

	if _, err := PrepareStandaloneCompaction(body); err != nil {
		t.Fatalf("compaction rejected custom tool history: %v", err)
	}
}

func TestResponsesCustomToolHistoryValidation(t *testing.T) {
	for _, tt := range []struct{ name, item, want string }{
		{"missing call ID", `{"type":"custom_tool_call","name":"read_note","input":"text"}`, "call_id"},
		{"blank name", `{"type":"custom_tool_call","call_id":"c","name":"  ","input":"text"}`, "name"},
		{"missing input", `{"type":"custom_tool_call","call_id":"c","name":"read_note"}`, "requires input"},
		{"object input", `{"type":"custom_tool_call","call_id":"c","name":"read_note","input":{}}`, "input"},
		{"missing output call ID", `{"type":"custom_tool_call_output","name":"read_note","output":"text"}`, "call_id"},
		{"null output", `{"type":"custom_tool_call_output","call_id":"c","output":null}`, "requires output"},
		{"object output", `{"type":"custom_tool_call_output","call_id":"c","output":{}}`, "output must be a string or array"},
		{"unknown output content", `{"type":"custom_tool_call_output","call_id":"c","output":[{"type":"unknown"}]}`, "unknown content type"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var request ResponsesRequest
			err := json.Unmarshal([]byte(`{"model":"test","input":[`+tt.item+`]}`), &request)
			if err == nil || !strings.Contains(err.Error(), "input[0]:") || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error = %v, want input[0] error containing %q", err, tt.want)
			}
		})
	}
}
