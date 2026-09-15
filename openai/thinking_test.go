package openai

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

func TestThinkingConversionMetadata(t *testing.T) {
	for _, metadata := range []struct {
		name     string
		thinking *model.Thinking
		generic  bool
	}{
		{"named", &model.Thinking{Values: []any{false, "high", "max"}, Default: "high"}, true},
		{"nonthinking", &model.Thinking{Values: []any{false}, Default: false}, true},
		{"nil", nil, false},
		{"invalid", &model.Thinking{Values: []any{"high"}, Default: "missing"}, false},
	} {
		for _, protocol := range []string{"chat", "responses"} {
			for _, tt := range []struct {
				name, fields            string
				wantGeneric, wantLegacy any
				legacyError             bool
			}{
				{"omitted", ``, nil, nil, false},
				{"empty", `,"reasoning":{"effort":""}`, nil, nil, false},
				{"none", `,"reasoning":{"effort":"none"}`, false, false, false},
				{"supported", `,"reasoning":{"effort":"max"}`, "max", "max", false},
				{"unsupported", `,"reasoning":{"effort":"low"}`, "low", "low", false},
				{"xhigh", `,"reasoning":{"effort":"xhigh"}`, "xhigh", "max", false},
				{"minimal", `,"reasoning":{"effort":"minimal"}`, "minimal", "low", false},
				{"future", `,"reasoning":{"effort":"future"}`, "future", nil, true},
				{"exact spelling", `,"reasoning":{"effort":" HIGH "}`, " HIGH ", nil, true},
				{"nested precedence", `,"reasoning_effort":"low","reasoning":{"effort":"xhigh"}`, "xhigh", "max", false},
				{"empty nested precedence", `,"reasoning_effort":"low","reasoning":{"effort":""}`, nil, nil, false},
			} {
				t.Run(metadata.name+"/"+protocol+"/"+tt.name, func(t *testing.T) {
					body := []byte(`{"model":"test","messages":[{"role":"user","content":"hi"}],"input":"hi"` + tt.fields + `}`)
					var request any
					if protocol == "chat" {
						request = &ChatCompletionRequest{}
					} else {
						request = &ResponsesRequest{}
					}
					if err := json.Unmarshal(body, request); err != nil {
						t.Fatal(err)
					}
					before, err := json.Marshal(request)
					if err != nil {
						t.Fatal(err)
					}
					var result *api.ChatRequest
					switch req := request.(type) {
					case *ChatCompletionRequest:
						result, err = FromChatRequest(*req, metadata.thinking)
					case *ResponsesRequest:
						result, err = FromResponsesRequest(*req, metadata.thinking)
					}
					after, marshalErr := json.Marshal(request)
					if marshalErr != nil || string(before) != string(after) {
						t.Fatalf("conversion mutated input: before=%s after=%s error=%v", before, after, marshalErr)
					}
					if !metadata.generic && tt.legacyError {
						if err == nil {
							t.Fatal("expected legacy effort error")
						}
						return
					}
					if err != nil {
						t.Fatal(err)
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
}

func TestResponsesThinkingOverrideWithMetadata(t *testing.T) {
	thinking := &model.Thinking{Values: []any{false, "high", "max"}, Default: "high"}
	for _, value := range []any{false, true, "minimal", "future", 75} {
		t.Run(fmt.Sprint(value), func(t *testing.T) {
			req := ResponsesRequest{Input: ResponsesInput{Text: "hi"}, Think: &api.ThinkValue{Value: value}}
			req.Reasoning.Effort = "xhigh"
			got, err := FromResponsesRequest(req, thinking)
			if value == 75 {
				if err == nil {
					t.Fatal("numeric thinking override must fail")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got.Think == nil || got.Think.Value != value {
				t.Fatalf("thinking=%v, want explicit override %#v", got.Think, value)
			}
		})
	}
}
