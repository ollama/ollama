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
	for _, metadata := range []struct {
		name     string
		thinking *model.Thinking
	}{
		{"named", &model.Thinking{Values: []any{false, "high", "max"}, Default: "high"}},
		{"nil", nil},
		{"invalid", &model.Thinking{Values: []any{"high"}, Default: "missing"}},
	} {
		for _, value := range []any{false, true, "low", "high", "max", "", "xhigh", "minimal", "future", 75} {
			t.Run(metadata.name+"/"+fmt.Sprint(value), func(t *testing.T) {
				req := ResponsesRequest{Input: ResponsesInput{Text: "hi"}, Think: &api.ThinkValue{Value: value}}
				req.Reasoning.Effort = "xhigh"
				got, err := FromResponsesRequest(req, metadata.thinking)
				wantErr := value == 75 || (!metadata.thinking.Valid() && (value == "" || value == "xhigh" || value == "minimal" || value == "future"))
				if wantErr {
					if err == nil {
						t.Fatal("invalid override must fail")
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
}

func TestThinkingBooleanOpenAIControls(t *testing.T) {
	for _, metadata := range []struct {
		name     string
		thinking *model.Thinking
		boolean  bool
	}{
		{"default off", &model.Thinking{Values: []any{false, true}, Default: false}, true},
		{"default on", &model.Thinking{Values: []any{false, true}, Default: true}, true},
		{"always on", &model.Thinking{Values: []any{true}, Default: true}, true},
		{"off only", &model.Thinking{Values: []any{false}, Default: false}, false},
		{"mixed", &model.Thinking{Values: []any{false, true, "medium"}, Default: true}, false},
	} {
		for _, protocol := range []string{"chat", "responses"} {
			for _, effort := range []string{"", "none", "low", "medium", "high", "max", "minimal", "xhigh", "ultra", "future", " HIGH "} {
				t.Run(metadata.name+"/"+protocol+"/"+effort, func(t *testing.T) {
					var got *api.ChatRequest
					var err error
					if protocol == "chat" {
						got, err = FromChatRequest(ChatCompletionRequest{Model: "test", ReasoningEffort: &effort}, metadata.thinking)
					} else {
						req := ResponsesRequest{Model: "test", Input: ResponsesInput{Text: "hi"}}
						req.Reasoning.Effort = effort
						got, err = FromResponsesRequest(req, metadata.thinking)
					}
					if err != nil {
						t.Fatal(err)
					}
					var want any = effort
					switch effort {
					case "":
						want = nil
					case "none":
						want = false
					case "future", " HIGH ":
					default:
						if metadata.boolean {
							want = true
						}
					}
					var actual any
					if got.Think != nil {
						actual = got.Think.Value
					}
					if actual != want {
						t.Fatalf("thinking=%#v, want %#v", actual, want)
					}
				})
			}
		}
	}
}
