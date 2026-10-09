package openai

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
)

func TestResponsesCustomToolHistory(t *testing.T) {
	for _, tt := range []struct{ name, input string }{
		{"freeform", "  Read \"note\" at C:\\notes\n\t雪 <marker>\r\n  "},
		{"JSON text", `{"path":"notes.txt"}`},
		{"empty", ""},
	} {
		t.Run(tt.name, func(t *testing.T) {
			body, err := json.Marshal(map[string]any{"model": "test", "input": []any{
				map[string]any{"role": "user", "content": "Read both notes."},
				map[string]any{"type": "function_call", "call_id": "ordinary", "name": "read_file", "arguments": `{"path":"first.txt"}`},
				map[string]any{"type": "custom_tool_call", "id": "ctc_note", "call_id": "freeform", "namespace": "workspace", "name": "read_note", "input": tt.input},
				map[string]any{"role": "assistant", "content": "Reading both notes."},
				map[string]any{"type": "custom_tool_call_output", "id": "ctco_note", "call_id": "freeform", "output": []any{
					map[string]any{"type": "input_text", "text": "marker\n"},
					map[string]any{"type": "input_text", "text": "雪  "},
				}},
				map[string]any{"type": "function_call_output", "call_id": "ordinary", "output": "first note"},
				map[string]any{"role": "user", "content": "Continue."},
			}})
			if err != nil {
				t.Fatal(err)
			}
			var request ResponsesRequest
			if err := json.Unmarshal(body, &request); err != nil {
				t.Fatal(err)
			}
			if call, ok := request.Input.Items[2].(ResponsesFunctionCall); !ok || call.ID != "ctc_note" {
				t.Fatalf("custom call item ID changed: %+v", request.Input.Items[2])
			}
			if output, ok := request.Input.Items[4].(ResponsesFunctionCallOutput); !ok || output.ID != "ctco_note" {
				t.Fatalf("custom output item ID changed: %+v", request.Input.Items[4])
			}
			chat, err := FromResponsesRequest(request)
			if err != nil {
				t.Fatal(err)
			}
			want := []api.Message{
				{Role: "user", Content: "Read both notes."},
				{Role: "assistant", Content: "Reading both notes.", ToolCalls: []api.ToolCall{
					{ID: "ordinary", Function: api.ToolCallFunction{Name: "read_file", Arguments: testArgs(map[string]any{"path": "first.txt"})}},
					{ID: "freeform", Function: api.ToolCallFunction{Name: "workspace.read_note", Arguments: testArgs(map[string]any{"input": tt.input})}},
				}},
				{Role: "tool", ToolCallID: "freeform", Content: "marker\n雪  "},
				{Role: "tool", ToolCallID: "ordinary", Content: "first note"},
				{Role: "user", Content: "Continue."},
			}
			if diff := cmp.Diff(want, chat.Messages, argsComparer); diff != "" {
				t.Fatalf("tool history changed (-want +got):\n%s", diff)
			}
		})
	}
}

func TestResponsesCustomToolHistoryValidation(t *testing.T) {
	for _, tt := range []struct {
		name string
		item string
		want string
	}{
		{"missing call ID", `{"type":"custom_tool_call","name":"read_note","input":"text"}`, "call_id"},
		{"blank call ID", `{"type":"custom_tool_call","call_id":" \t","name":"read_note","input":"text"}`, "call_id"},
		{"missing name", `{"type":"custom_tool_call","call_id":"c","input":"text"}`, "name"},
		{"blank name", `{"type":"custom_tool_call","call_id":"c","name":"  ","input":"text"}`, "name"},
		{"missing input", `{"type":"custom_tool_call","call_id":"c","name":"read_note"}`, "input"},
		{"null input", `{"type":"custom_tool_call","call_id":"c","name":"read_note","input":null}`, "input"},
		{"object input", `{"type":"custom_tool_call","call_id":"c","name":"read_note","input":{}}`, "input"},
		{"missing output call ID", `{"type":"custom_tool_call_output","name":"read_note","output":"text"}`, "call_id"},
		{"null output call ID", `{"type":"custom_tool_call_output","call_id":null,"name":"read_note","output":"text"}`, "call_id"},
		{"blank output call ID", `{"type":"custom_tool_call_output","call_id":"  ","output":"text"}`, "call_id"},
		{"missing output", `{"type":"custom_tool_call_output","call_id":"c"}`, "output"},
		{"null output", `{"type":"custom_tool_call_output","call_id":"c","output":null}`, "output"},
		{"object output", `{"type":"custom_tool_call_output","call_id":"c","output":{}}`, "output must be a string or array"},
		{"unknown output content", `{"type":"custom_tool_call_output","call_id":"c","output":[{"type":"unknown"}]}`, "unknown content type"},
		{"unknown history", `{"type":"unknown_tool_call"}`, `unknown input item type: "unknown_tool_call"`},
	} {
		t.Run(tt.name, func(t *testing.T) {
			body := []byte(`{"model":"test","input":[{"role":"user","content":"Continue."},` + tt.item + `]}`)
			var request ResponsesRequest
			err := json.Unmarshal(body, &request)
			if err == nil || !strings.Contains(err.Error(), "input[1]:") || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error = %v, want input[1] error containing %q", err, tt.want)
			}
			if _, err := PrepareStandaloneCompaction(body); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("compaction error = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestCompactionCustomToolHistory(t *testing.T) {
	input := `[
		{"role":"user","content":"Read the notes."},
		{"type":"function_call","call_id":"ordinary","name":"read_file","arguments":"{}"},
		{"type":"custom_tool_call","call_id":"freeform","namespace":"workspace","name":"read_note","input":" Read \"marker\"\n雪 "},
		{"type":"custom_tool_call_output","call_id":"freeform","output":[{"type":"input_text","text":"marker"},{"type":"input_image","image_url":"` + compactionTestPNG + `"}]},
		{"type":"function_call_output","call_id":"ordinary","output":"first note"},
		{"role":"assistant","content":"Read both notes."}
	]`
	for _, triggered := range []bool{false, true} {
		t.Run(map[bool]string{false: "standalone", true: "triggered"}[triggered], func(t *testing.T) {
			var plan *ResponsesCompactionPlan
			var err error
			if triggered {
				var requested bool
				plan, requested, err = PrepareTriggeredCompaction([]byte(`{"model":"test","stream":true,"input":` + strings.TrimSuffix(input, "]") + `,{"type":"compaction_trigger"}]}`))
				if !requested {
					t.Fatal("compaction trigger was not recognized")
				}
			} else {
				plan, err = PrepareStandaloneCompaction([]byte(`{"model":"test","input":` + input + `}`))
			}
			if err != nil {
				t.Fatal(err)
			}
			result, err := plan.Complete(compactionResponseBody(t, map[string]any{
				"summary": "Keep the tool results.", "retain_item_ids": []string{"item_000004", "item_000005"},
			}))
			if err != nil {
				t.Fatal(err)
			}
			retained := decodeResultPayload(t, result).Retained
			want := make([]api.Message, 0, 4)
			for _, item := range plan.items[1:5] {
				want = append(want, item.Message)
			}
			if diff := cmp.Diff(want, retained, argsComparer); diff != "" {
				t.Fatalf("compaction lost or reordered a tool pair (-want +got):\n%s", diff)
			}
			call := retained[1].ToolCalls[0]
			if got, _ := call.Function.Arguments.Get("input"); got != " Read \"marker\"\n雪 " || call.ID != "freeform" || call.Function.Name != "workspace.read_note" {
				t.Fatalf("custom call changed: %+v", call)
			}
			if retained[2].Content != "marker" || retained[2].ToolCallID != "freeform" || len(retained[2].Images) != 1 {
				t.Fatalf("custom output changed: %+v", retained[2])
			}
			replay, err := json.Marshal(map[string]any{"model": "test", "input": []any{result.Item}})
			if err != nil {
				t.Fatal(err)
			}
			expanded, changed, err := ExpandResponsesCompactionInput(replay)
			if err != nil || !changed {
				t.Fatalf("compaction expansion: changed=%v, err=%v", changed, err)
			}
			replayed, err := PrepareStandaloneCompaction(expanded)
			if err != nil {
				t.Fatal(err)
			}
			var got []api.Message
			for _, item := range replayed.items[2:] { // Skip the summary call and output.
				got = append(got, item.Message)
			}
			if diff := cmp.Diff(retained, got, argsComparer); diff != "" {
				t.Fatalf("compaction replay changed history (-want +got):\n%s", diff)
			}
		})
	}
}
