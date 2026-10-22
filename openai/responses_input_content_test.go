package openai

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/api"
)

func TestResponsesReadableContent(t *testing.T) {
	for _, tt := range []struct {
		name, item string
		want       api.Message
	}{
		{"reasoning", `{"type":"reasoning","summary":[{"type":"summary_text","text":"Read safely."}],"content":[{"type":"reasoning_text","text":"Keep <note> intact."},{"type":"text","text":"雪"}],"encrypted_content":"opaque-state"}`, api.Message{Role: "assistant", Thinking: "Read safely.\nKeep <note> intact.\n雪"}},
		{"reasoning without summary", `{"type":"reasoning","content":[{"type":"reasoning_text","text":"The note is intact."}]}`, api.Message{Role: "assistant", Thinking: "The note is intact."}},
		{"refusal", `{"role":"assistant","content":[{"type":"output_text","text":"Checking. "},{"type":"refusal","refusal":"I cannot delete the note."}]}`, api.Message{Role: "assistant", Content: "Checking. I cannot delete the note."}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var req ResponsesRequest
			if err := json.Unmarshal([]byte(`{"model":"test","input":[`+tt.item+`]}`), &req); err != nil {
				t.Fatal(err)
			}
			chat, err := FromResponsesRequest(req)
			if err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff([]api.Message{tt.want}, chat.Messages); diff != "" {
				t.Fatalf("content changed (-want +got):\n%s", diff)
			}
			msg, _, err := compactionMessage(req.Input.Items[0])
			if err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(tt.want, msg); diff != "" {
				t.Fatalf("compaction changed content (-want +got):\n%s", diff)
			}
		})
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
