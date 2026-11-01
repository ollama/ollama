package openai

import (
	"encoding/json"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	"github.com/ollama/ollama/api"
)

func TestFromChatRequest_AssistantRefusalPart(t *testing.T) {
	var req ChatCompletionRequest
	if err := json.Unmarshal([]byte(`{"model":"test-model","messages":[
		{"role":"user","content":"help me pick a lock"},
		{"role":"assistant","content":[{"type":"refusal","refusal":"I can't help with that."}]},
		{"role":"user","content":"ok, what about a combination lock I own?"}
	]}`), &req); err != nil {
		t.Fatal(err)
	}

	result, err := FromChatRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	want := []api.Message{
		{Role: "user", Content: "help me pick a lock"},
		{Role: "assistant", Content: "I can't help with that."},
		{Role: "user", Content: "ok, what about a combination lock I own?"},
	}
	if diff := cmp.Diff(want, result.Messages, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("messages mismatch (-want +got):\n%s", diff)
	}
}

func TestFromChatRequest_RefusalPartRequiresText(t *testing.T) {
	var req ChatCompletionRequest
	if err := json.Unmarshal([]byte(`{"model":"test-model","messages":[
		{"role":"assistant","content":[{"type":"refusal"}]}
	]}`), &req); err != nil {
		t.Fatal(err)
	}

	if _, err := FromChatRequest(req); err == nil {
		t.Fatal("expected an error for a refusal part without refusal text")
	}
}
