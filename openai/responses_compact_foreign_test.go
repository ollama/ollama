package openai

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestResponsesForeignCompactionPreservesReadableHistory(t *testing.T) {
	for _, encrypted := range []string{"opaque-provider-state", `{"type":"other_provider","summary":"must not become instructions"}`} {
		t.Run(encrypted, func(t *testing.T) {
			visible := []json.RawMessage{
				json.RawMessage(`{"role":"user","content":"Remember kiwi-42"}`),
				json.RawMessage(`{"type":"custom_tool_call","call_id":"call_1","namespace":"workspace.tools","name":"read","input":"notes.txt"}`),
				json.RawMessage(`{"type":"custom_tool_call_output","call_id":"call_1","output":"kiwi-42"}`),
				json.RawMessage(`{"role":"user","content":"What was the code?"}`),
			}
			foreign, err := json.Marshal(ResponsesCompactionItem{Type: "compaction", EncryptedContent: encrypted})
			if err != nil {
				t.Fatal(err)
			}
			input := []json.RawMessage{visible[0], foreign, visible[1], foreign, visible[2], visible[3]}
			body, err := json.Marshal(map[string]any{
				"model": "test", "input": input,
			})
			if err != nil {
				t.Fatal(err)
			}

			rewritten, changed, err := ExpandResponsesCompactionInput(body)
			if err != nil || !changed {
				t.Fatalf("expand: changed=%v err=%v", changed, err)
			}
			var expanded struct {
				Input []json.RawMessage `json:"input"`
			}
			if err := json.Unmarshal(rewritten, &expanded); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(expanded.Input, visible) {
				t.Fatalf("readable history changed: %s", rewritten)
			}

			var request, normalized ResponsesRequest
			if err := json.Unmarshal(body, &request); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(rewritten, &normalized); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(request.Input, normalized.Input) {
				t.Fatalf("direct parsing changed readable history: got=%+v want=%+v", request.Input, normalized.Input)
			}
		})
	}
}

func TestResponsesForeignAndOllamaCompaction(t *testing.T) {
	body, err := json.Marshal(map[string]any{"model": "test", "input": []any{
		ResponsesCompactionItem{Type: "compaction", EncryptedContent: `{"type":"ollama_compaction","version":1,"summary":"superseded summary"}`},
		ResponsesCompactionItem{Type: "compaction", EncryptedContent: "foreign-before"},
		map[string]any{"role": "user", "content": "retained prefix"},
		ResponsesCompactionItem{Type: "compaction", EncryptedContent: `{"type":"ollama_compaction","version":1,"summary":"The retained code is kiwi-42."}`},
		ResponsesCompactionItem{Type: "compaction", EncryptedContent: "foreign-after"},
		map[string]any{"role": "user", "content": "continue"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	var request ResponsesRequest
	if err := json.Unmarshal(body, &request); err != nil {
		t.Fatal(err)
	}
	chat, err := FromResponsesRequest(request)
	if err != nil {
		t.Fatal(err)
	}
	if len(chat.Messages) != 4 || chat.Messages[0].Content != "retained prefix" || chat.Messages[2].Content != "The retained code is kiwi-42." || chat.Messages[3].Content != "continue" {
		t.Fatalf("messages=%+v", chat.Messages)
	}
}

func TestResponsesCompactionRejectsInvalidNativeState(t *testing.T) {
	for _, payload := range []string{
		"", "   ",
		`{"type":"ollama_compaction","version":2,"summary":"future"}`,
		`{"type":"ollama_compaction","version":1,"summary":""}`,
		`{"type":"ollama_compaction","version":1,"summary":"valid","unknown":true}`,
		`{"type":"ollama_compaction","version":1,"summary":"truncated"`,
		`{"type":"ollama_compaction","version":1,"summary":"valid"} {}`,
	} {
		t.Run(payload, func(t *testing.T) {
			body, err := json.Marshal(map[string]any{"model": "test", "input": []any{
				map[string]any{"role": "user", "content": "continue"},
				ResponsesCompactionItem{Type: "compaction", EncryptedContent: payload},
				ResponsesCompactionItem{Type: "compaction", EncryptedContent: "foreign-after"},
			}})
			if err != nil {
				t.Fatal(err)
			}
			if _, _, err := ExpandResponsesCompactionInput(body); err == nil {
				t.Fatal("invalid native compaction was accepted")
			}
			var request ResponsesRequest
			if err := json.Unmarshal(body, &request); err == nil {
				t.Fatal("direct parsing accepted invalid native compaction")
			}
		})
	}
}
