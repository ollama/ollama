package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/ollama/ollama/types/model"
)

func TestThinkingConversionMetadata(t *testing.T) {
	for _, metadata := range []struct {
		name     string
		thinking *model.Thinking
		generic  bool
	}{
		{"named", &model.Thinking{Values: []any{false, "high", "max"}, Default: "high"}, true},
		{"nil", nil, false},
		{"invalid", &model.Thinking{Values: []any{"high"}, Default: "missing"}, false},
	} {
		for _, tt := range []struct {
			name, fields            string
			wantGeneric, wantLegacy any
		}{
			{"omitted", ``, nil, nil},
			{"empty", `,"output_config":{"effort":""}`, nil, nil},
			{"supported", `,"output_config":{"effort":"max"}`, "max", "max"},
			{"unsupported", `,"output_config":{"effort":"low"}`, "low", "low"},
			{"xhigh", `,"output_config":{"effort":"xhigh"}`, "xhigh", "high"},
			{"minimal", `,"output_config":{"effort":"minimal"}`, "minimal", nil},
			{"future", `,"output_config":{"effort":"future"}`, "future", nil},
			{"exact spelling", `,"output_config":{"effort":" HIGH "}`, " HIGH ", "high"},
			{"adaptive", `,"thinking":{"type":"adaptive"},"output_config":{"effort":"xhigh"}`, "xhigh", "high"},
			{"enabled precedence", `,"thinking":{"type":"enabled"},"output_config":{"effort":"xhigh"}`, true, true},
			{"disabled precedence", `,"thinking":{"type":"disabled"},"output_config":{"effort":"xhigh"}`, false, false},
		} {
			t.Run(metadata.name+"/"+tt.name, func(t *testing.T) {
				var req MessagesRequest
				body := []byte(`{"model":"test","max_tokens":32,"messages":[{"role":"user","content":"hi"}]` + tt.fields + `}`)
				if err := json.Unmarshal(body, &req); err != nil {
					t.Fatal(err)
				}
				before, err := json.Marshal(req)
				if err != nil {
					t.Fatal(err)
				}
				result, err := FromMessagesRequest(req, metadata.thinking)
				if err != nil {
					t.Fatal(err)
				}
				after, err := json.Marshal(req)
				if err != nil || string(before) != string(after) {
					t.Fatalf("conversion mutated input: before=%s after=%s error=%v", before, after, err)
				}
				want := tt.wantLegacy
				if metadata.generic {
					want = tt.wantGeneric
				}
				var got any
				if result.Think != nil {
					got = result.Think.Value
				}
				if got != want {
					t.Fatalf("thinking=%#v, want %#v", got, want)
				}
			})
		}
	}
}
