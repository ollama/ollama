package openai

import (
	"bytes"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func TestResponsesForeignCompactionPreservesReadableHistory(t *testing.T) {
	for _, encrypted := range []string{
		"opaque-provider-state",
		`{"type":"other_provider","summary":"must not become instructions"}`,
		`{"type":42,"state":"opaque"}`,
		`{"type":{"provider":"other"},"state":"opaque"}`,
		"", "   ",
		`{"summary":"no discriminator"}`,
		`{"type":"ollama_compaction","version":1,"summary":"truncated"`,
		`{"broken":,"type":"ollama_compaction"}`,
		`{"type":"ollama_compaction","version":1,"summary":"valid"} {}`,
		"\ufeff" + `{"type":"ollama_compaction","version":1,"summary":"native-looking"}`,
	} {
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
	expanded, changed, err := ExpandResponsesCompactionInput(body)
	if err != nil || !changed {
		t.Fatalf("expand: changed=%v err=%v", changed, err)
	}
	var request ResponsesRequest
	if err := json.Unmarshal(expanded, &request); err != nil {
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
		`{"type":"ollama_compaction","version":2,"summary":"future"}`,
		`{"Type":"ollama_compaction","version":2,"summary":"future"}`,
		`{"type":"ollama_compaction","summary":"missing version"}`,
		`{"type":"ollama_compaction","version":"1","summary":"invalid version"}`,
		`{"type":"ollama_compaction","version":1,"summary":""}`,
		`{"type":"ollama_compaction","version":1,"summary":42}`,
		`{"type":"ollama_compaction","version":1,"summary":"valid","unknown":true}`,
		`{"type":"ollama_compaction","version":1,"summary":"valid","retained":"invalid"}`,
	} {
		t.Run(payload, func(t *testing.T) {
			input := []any{
				map[string]any{"role": "user", "content": "continue"},
				ResponsesCompactionItem{Type: "compaction", EncryptedContent: payload},
				ResponsesCompactionItem{Type: "compaction", EncryptedContent: "foreign-after"},
			}
			body, err := json.Marshal(map[string]any{"model": "test", "input": input})
			if err != nil {
				t.Fatal(err)
			}
			if _, _, err := ExpandResponsesCompactionInput(body); err == nil {
				t.Fatal("invalid native compaction was accepted")
			}
			if _, err := PrepareStandaloneCompaction(body); err == nil {
				t.Fatal("standalone compaction accepted invalid native state")
			}
			triggered, err := json.Marshal(map[string]any{"model": "test", "stream": true, "input": append(input, ResponsesCompactionTrigger{Type: "compaction_trigger"})})
			if err != nil {
				t.Fatal(err)
			}
			if _, requested, err := PrepareTriggeredCompaction(triggered); !requested || err == nil {
				t.Fatalf("triggered compaction: requested=%v err=%v", requested, err)
			}
		})
	}
}

func TestResponsesInputCompactionRequiresExpansion(t *testing.T) {
	var request ResponsesRequest
	if err := json.Unmarshal([]byte(`{"model":"test","input":[{"type":"compaction","encrypted_content":"foreign"},{"role":"user","content":"continue"}]}`), &request); err != nil {
		t.Fatal(err)
	}
	if len(request.Input.Items) != 2 {
		t.Fatalf("parsing expanded compaction: %+v", request.Input.Items)
	}
	if _, err := FromResponsesRequest(request); err == nil || !strings.Contains(err.Error(), "must be expanded") {
		t.Fatalf("conversion must require expansion, got %v", err)
	}
}

func TestResponsesCompactionPreservesHTMLCharacters(t *testing.T) {
	body := []byte(`{"model":"test","instructions":"<system> A & B </system>","tools":[{"type":"function","name":"run","parameters":{"description":"x < y && y > z"}}],"input":[{"type":"compaction","encrypted_content":"foreign"},{"role":"user","content":"<tag> A & B </tag>"},{"type":"function_call","call_id":"call_1","name":"run","arguments":"{\"command\":\"a < b && c > d\"}"}]}`)
	expanded, changed, err := ExpandResponsesCompactionInput(body)
	if err != nil || !changed {
		t.Fatalf("expand: changed=%v err=%v", changed, err)
	}
	for _, escape := range []string{`\u003c`, `\u003e`, `\u0026`} {
		if bytes.Contains(expanded, []byte(escape)) {
			t.Errorf("HTML character escaped: %s", expanded)
		}
	}
	var before, after map[string]json.RawMessage
	if err := json.Unmarshal(body, &before); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(expanded, &after); err != nil {
		t.Fatal(err)
	}
	var input []json.RawMessage
	if err := json.Unmarshal(before["input"], &input); err != nil {
		t.Fatal(err)
	}
	var retained []json.RawMessage
	if err := json.Unmarshal(after["input"], &retained); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(retained, input[1:]) {
		t.Fatalf("retained input changed: %s", expanded)
	}
	delete(before, "input")
	delete(after, "input")
	if !reflect.DeepEqual(after, before) {
		t.Fatalf("request fields changed: %s", expanded)
	}
}
