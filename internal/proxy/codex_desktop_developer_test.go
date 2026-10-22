package proxy

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestNormalizeOllamaDeveloperInstructions(t *testing.T) {
	const input = `[
        {"type":"message","role":"system","content":"Base policy"},
        {"role":"developer","content":"Use the scratch project."},
        {"role":"user","content":"Read the fixture"},
        {"type":"function_call","call_id":"call_1","name":"read_file","arguments":"{}"},
        {"type":"function_call_output","call_id":"call_1","output":"fixture contents"},
        {"type":"message","id":"dev_2","role":"developer","content":[{"type":"input_text","text":"Preserve the fixture."}],"metadata":{"turn":"later"}},
        {"role":"user","content":"Continue"}
    ]`
	var want []map[string]any
	if err := json.Unmarshal([]byte(input), &want); err != nil {
		t.Fatal(err)
	}
	for _, message := range want {
		if message["role"] == "developer" {
			message["role"] = "system"
		}
	}
	body, err := normalizeOllamaRequestBody([]byte(`{"model":"test","input":`+input+`}`), routingModel{})
	if err != nil {
		t.Fatal(err)
	}
	var got struct {
		Input []map[string]any `json:"input"`
	}
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.Input, want) {
		t.Fatalf("gateway must only rewrite developer roles and preserve contents, metadata, and tool history: %s", body)
	}
}
