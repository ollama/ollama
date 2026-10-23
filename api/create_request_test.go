package api

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestCreateRequestSystemJSON(t *testing.T) {
	for _, tc := range []struct {
		name, input, want string
		present           bool
	}{
		{"omitted", `{}`, "", false},
		{"nonempty", `{"system":"replacement"}`, "replacement", true},
		{"empty", `{"system":""}`, "", true},
		{"null", `{"system":null}`, "", false},
		{"case insensitive", `{"SYSTEM":""}`, "", true},
		{"last duplicate", `{"system":"first","system":""}`, "", true},
		{"null leaves string", `{"system":"first","system":null}`, "first", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var req CreateRequest
			if err := json.Unmarshal([]byte(tc.input), &req); err != nil {
				t.Fatal(err)
			}
			if req.System != tc.want {
				t.Fatalf("System = %q, want %q", req.System, tc.want)
			}
			for _, value := range []any{req, &req} {
				data, err := json.Marshal(value)
				if err != nil {
					t.Fatal(err)
				}
				var wire map[string]json.RawMessage
				if err := json.Unmarshal(data, &wire); err != nil {
					t.Fatal(err)
				}
				_, present := wire["system"]
				if want := tc.present || tc.want != ""; present != want {
					t.Fatalf("system presence in %s = %v, want %v", data, present, want)
				}
			}
		})
	}
	for _, input := range []string{`{"system":42}`, `{"system":true}`, `{"system":[]}`, `{"system":{}}`, `{"system":`} {
		var req CreateRequest
		if err := json.Unmarshal([]byte(input), &req); err == nil {
			t.Errorf("Unmarshal(%s) succeeded, want error", input)
		}
	}
}

func TestCreateRequestSystemLegacyGoValues(t *testing.T) {
	for _, tc := range []struct {
		name    string
		req     CreateRequest
		want    string
		present bool
	}{
		{"zero value", CreateRequest{}, "", false},
		{"legacy empty string", CreateRequest{System: ""}, "", false},
		{"legacy nonempty string", CreateRequest{System: "replacement"}, "replacement", true},
		{"explicit empty string", CreateRequest{SystemSet: true}, "", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var legacyString string = tc.req.System
			if legacyString != tc.want {
				t.Fatalf("System = %q, want %q", legacyString, tc.want)
			}
			data, err := json.Marshal(tc.req)
			if err != nil {
				t.Fatal(err)
			}
			var wire map[string]json.RawMessage
			if err := json.Unmarshal(data, &wire); err != nil {
				t.Fatal(err)
			}
			_, present := wire["system"]
			if present != tc.present {
				t.Fatalf("system presence in %s = %v, want %v", data, present, tc.present)
			}
			if _, present := wire["SystemSet"]; present {
				t.Fatalf("presence metadata leaked onto wire: %s", data)
			}
		})
	}
}

func TestCreateRequestSystemPreservesOtherFields(t *testing.T) {
	const input = `{"model":"child","from":"base","stream":false,"system":"","license":["MIT"],"parameters":{"temperature":0.5},"messages":[{"role":"user","content":"hello"}],"unknown":true}`
	var req CreateRequest
	if err := json.Unmarshal([]byte(input), &req); err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	want := map[string]any{"model": "child", "from": "base", "name": "", "stream": false, "system": "", "license": []any{"MIT"}, "parameters": map[string]any{"temperature": 0.5}, "messages": []any{map[string]any{"role": "user", "content": "hello"}}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("request round trip = %s, want %v", data, want)
	}
}
