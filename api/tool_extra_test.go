package api

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

// Function-level and tool-level keys that the structs do not model must
// survive a decode/encode round trip, the same way unmodeled JSON Schema
// keywords already survive on ToolFunctionParameters. See #19852.
func TestToolExtraRoundTrip(t *testing.T) {
	cases := []string{
		`{"type":"function","function":{"name":"get_weather","description":"Get the weather","strict":true,"parameters":{"type":"object","properties":{"city":{"type":"string","description":"City"}},"required":["city"],"additionalProperties":false}}}`,
		`{"type":"function","cache_control":{"type":"ephemeral"},"function":{"name":"lookup","strict":false,"parameters":{"type":"object","properties":{}}}}`,
		`{"type":"function","function":{"name":"plain","parameters":{"type":"object","properties":{}}}}`,
	}
	for _, in := range cases {
		var tool Tool
		if err := json.Unmarshal([]byte(in), &tool); err != nil {
			t.Fatalf("unmarshal %s: %v", in, err)
		}
		out, err := json.Marshal(tool)
		if err != nil {
			t.Fatalf("marshal %s: %v", in, err)
		}
		var want, got any
		if err := json.Unmarshal([]byte(in), &want); err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(out, &got); err != nil {
			t.Fatalf("unmarshal round trip %s: %v", out, err)
		}
		if !reflect.DeepEqual(want, got) {
			t.Errorf("round trip changed the tool:\n in: %s\nout: %s", in, out)
		}
	}
}

// The modeled fields keep their meaning and none of them is emitted twice
// when unmodeled keys are appended.
func TestToolExtraDoesNotDuplicateModeledKeys(t *testing.T) {
	in := `{"type":"function","function":{"name":"f","description":"d","strict":true,"parameters":{"type":"object","properties":{}}}}`
	var tool Tool
	if err := json.Unmarshal([]byte(in), &tool); err != nil {
		t.Fatal(err)
	}
	if tool.Function.Name != "f" || tool.Function.Description != "d" {
		t.Fatalf("modeled fields not decoded: %+v", tool.Function)
	}
	out, err := json.Marshal(tool)
	if err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{`"name"`, `"description"`, `"parameters"`, `"strict"`} {
		if n := strings.Count(string(out), key); n != 1 {
			t.Errorf("key %s appears %d times in %s, want once", key, n, out)
		}
	}
}

func TestToolWithoutExtraMarshalsAsBefore(t *testing.T) {
	tool := Tool{Type: "function", Function: ToolFunction{Name: "f", Parameters: ToolFunctionParameters{Type: "object"}}}
	out, err := json.Marshal(tool)
	if err != nil {
		t.Fatal(err)
	}
	const want = `{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":null}}}`
	if string(out) != want {
		t.Errorf("marshal without extras: got %s, want %s", out, want)
	}
}
