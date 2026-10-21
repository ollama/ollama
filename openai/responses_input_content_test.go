package openai

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/api"
)

func TestResponsesReadableHistory(t *testing.T) {
	body := []byte(`{"model":"test","input":[
		{"role":"user","content":"Read the note without deleting it."},
		{"type":"reasoning","id":"rs_foreign","summary":[{"type":"summary_text","text":"Read safely."}],"content":[{"type":"reasoning_text","text":"Keep <note> intact."},{"type":"text","text":"雪"}],"encrypted_content":"opaque-state"},
		{"type":"function_call","id":"fc_read","call_id":"call_read","name":"read_note","arguments":"{}"},
		{"role":"assistant","content":[{"type":"output_text","text":"Checking. "},{"type":"refusal","refusal":"I cannot delete the note."}]},
		{"type":"function_call_output","call_id":"call_read","output":"note contents"},
		{"type":"reasoning","content":[{"type":"reasoning_text","text":"The note is intact."}]},
		{"role":"user","content":"Continue."}
	]}`)
	var req ResponsesRequest
	if err := json.Unmarshal(body, &req); err != nil {
		t.Fatal(err)
	}
	if call := req.Input.Items[2].(ResponsesFunctionCall); call.ID != "fc_read" || call.CallID != "call_read" {
		t.Fatalf("tool identity changed: %+v", call)
	}
	chat, err := FromResponsesRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	want := []api.Message{
		{Role: "user", Content: "Read the note without deleting it."},
		{Role: "assistant", Content: "Checking. I cannot delete the note.", Thinking: "Read safely.\nKeep <note> intact.\n雪", ToolCalls: []api.ToolCall{{ID: "call_read", Function: api.ToolCallFunction{Name: "read_note", Arguments: testArgs(map[string]any{})}}}},
		{Role: "tool", ToolCallID: "call_read", Content: "note contents"},
		{Role: "assistant", Thinking: "The note is intact."},
		{Role: "user", Content: "Continue."},
	}
	if diff := cmp.Diff(want, chat.Messages, argsComparer); diff != "" {
		t.Fatalf("history changed (-want +got):\n%s", diff)
	}

	plan, err := PrepareStandaloneCompaction(body)
	if err != nil {
		t.Fatal(err)
	}
	result, err := plan.Complete(compactionResponseBody(t, map[string]any{
		"summary":         "Read the note safely.",
		"retain_item_ids": []string{"item_000001", "item_000002", "item_000003", "item_000004", "item_000005", "item_000006", "item_000007"},
	}))
	if err != nil {
		t.Fatal(err)
	}
	continuation, err := json.Marshal(map[string]any{"model": "test", "input": []any{result.Item, map[string]string{"role": "user", "content": "Next."}}})
	if err != nil {
		t.Fatal(err)
	}
	expanded, _, err := ExpandResponsesCompactionInput(continuation)
	if err != nil {
		t.Fatal(err)
	}
	var resumed ResponsesRequest
	if err := json.Unmarshal(expanded, &resumed); err != nil {
		t.Fatal(err)
	}
	chat, err = FromResponsesRequest(resumed)
	if err != nil {
		t.Fatal(err)
	}
	// The first two messages carry the compaction summary and its acknowledgement.
	want = append(want, api.Message{Role: "user", Content: "Next."})
	if len(chat.Messages) != len(want)+2 {
		t.Fatalf("compaction changed history length: %+v", chat.Messages)
	}
	if diff := cmp.Diff(want, chat.Messages[2:], argsComparer); diff != "" {
		t.Fatalf("compaction continuation changed history (-want +got):\n%s", diff)
	}
}

func TestResponsesAgentImages(t *testing.T) {
	body := []byte(`{"model":"test","input":[{"type":"agent_message","author":"worker","recipient":"root","content":[
		{"type":"input_text","text":"Inspect this pixel."},
		{"type":"input_image","image_url":"` + compactionTestPNG + `"}
	]}]}`)
	var req ResponsesRequest
	if err := json.Unmarshal(body, &req); err != nil {
		t.Fatal(err)
	}
	chat, err := FromResponsesRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	image, err := decodeImageURL(compactionTestPNG)
	if err != nil {
		t.Fatal(err)
	}
	want := api.Message{Role: "user", Content: "Agent message from \"worker\" to \"root\":\nInspect this pixel.", Images: []api.ImageData{image}}
	if diff := cmp.Diff([]api.Message{want}, chat.Messages); diff != "" {
		t.Fatalf("agent content changed (-want +got):\n%s", diff)
	}
	plan, err := PrepareStandaloneCompaction(body)
	if err != nil {
		t.Fatal(err)
	}
	result, err := plan.Complete(compactionResponseBody(t, map[string]any{
		"summary": "Inspect the pixel.", "retain_item_ids": []string{"item_000001"},
	}))
	if err != nil {
		t.Fatal(err)
	}
	if diff := cmp.Diff([]api.Message{want}, decodeResultPayload(t, result).Retained); diff != "" {
		t.Fatalf("compaction lost agent content (-want +got):\n%s", diff)
	}
}

func TestResponsesRejectsUnrepresentableContent(t *testing.T) {
	for _, tt := range []struct{ name, item, want string }{
		{"file image", `{"role":"user","content":[{"type":"input_text","text":"preserve me"},{"type":"input_image","file_id":"file_private"}]}`, "file_id"},
		{"missing image", `{"role":"user","content":[{"type":"input_image"}]}`, "image_url"},
		{"tool image", `{"type":"function_call_output","call_id":"call_1","output":[{"type":"input_text","text":"preserve me"},{"type":"input_image","file_id":"file_private"}]}`, "file_id"},
		{"agent image", `{"type":"agent_message","author":"worker","recipient":"root","content":[{"type":"input_image","file_id":"file_private"}]}`, "file_id"},
		{"unknown reasoning", `{"type":"reasoning","content":[{"type":"future_reasoning","text":"private-content"}]}`, "unsupported reasoning content type"},
		{"missing reasoning text", `{"type":"reasoning","content":[{"type":"reasoning_text"}]}`, "requires text"},
		{"null reasoning text", `{"type":"reasoning","content":[{"type":"text","text":null}]}`, "requires text"},
		{"missing refusal", `{"role":"assistant","content":[{"type":"refusal"}]}`, "requires refusal"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			body := []byte(`{"model":"test","input":[{"role":"user","content":"Before."},` + tt.item + `,{"role":"user","content":"After."}]}`)
			var req ResponsesRequest
			err := json.Unmarshal(body, &req)
			if err == nil {
				_, err = FromResponsesRequest(req)
			}
			if err == nil || !strings.Contains(err.Error(), tt.want) || strings.Contains(err.Error(), "private-content") || strings.Contains(err.Error(), "file_private") {
				t.Fatalf("error = %v, want safe error containing %q", err, tt.want)
			}
			if _, err := PrepareStandaloneCompaction(body); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("compaction error = %v, want %q", err, tt.want)
			}
		})
	}
}
