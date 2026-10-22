package openai

import (
	"encoding/json"
	"fmt"
	"strings"
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
				name, fields                                string
				wantGeneric, wantLegacy, wantNamedResponses any
				legacyError, namedResponsesError            bool
			}{
				{"omitted", ``, nil, nil, nil, false, false},
				{"empty", `,"reasoning":{"effort":""}`, nil, nil, nil, false, false},
				{"none", `,"reasoning":{"effort":"none"}`, false, false, false, false, false},
				{"supported", `,"reasoning":{"effort":"max"}`, "max", "max", "max", false, false},
				{"unsupported", `,"reasoning":{"effort":"low"}`, "low", "low", "high", false, false},
				{"xhigh", `,"reasoning":{"effort":"xhigh"}`, "xhigh", "max", "max", false, false},
				{"minimal", `,"reasoning":{"effort":"minimal"}`, "minimal", "low", "high", false, false},
				{"future", `,"reasoning":{"effort":"future"}`, "future", nil, nil, true, true},
				{"exact spelling", `,"reasoning":{"effort":" HIGH "}`, " HIGH ", nil, nil, true, true},
				{"nested precedence", `,"reasoning_effort":"low","reasoning":{"effort":"xhigh"}`, "xhigh", "max", "max", false, false},
				{"empty nested precedence", `,"reasoning_effort":"low","reasoning":{"effort":""}`, nil, nil, nil, false, false},
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
					want := tt.wantLegacy
					if metadata.generic {
						want = tt.wantGeneric
					}
					responsesError := metadata.generic && tt.wantGeneric != nil && !metadata.thinking.Supports(tt.wantGeneric)
					if protocol == "responses" && metadata.name == "named" {
						want, responsesError = tt.wantNamedResponses, tt.namedResponsesError
					}
					if protocol == "responses" && responsesError {
						if err == nil || !strings.Contains(err.Error(), "supported values") {
							t.Fatalf("unsupported Responses control needs supported choices: %v", err)
						}
						return
					}
					if err != nil {
						t.Fatal(err)
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
				want := value
				wantErr := value == 75 || (!metadata.thinking.Valid() && (value == "" || value == "xhigh" || value == "minimal" || value == "future"))
				if metadata.thinking.Valid() {
					switch value {
					case "low", "minimal":
						want = "high"
					case "xhigh":
						want = "max"
					case "", "future":
						wantErr = true
					}
				}
				if wantErr {
					if err == nil {
						t.Fatal("invalid override must fail")
					}
					return
				}
				if err != nil {
					t.Fatal(err)
				}
				if got.Think == nil || got.Think.Value != want || req.Think.Value != value {
					t.Fatalf("thinking=%v, want resolved override %#v without changing %#v", got.Think, want, value)
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
					if protocol == "responses" && metadata.name == "mixed" && effort != "" && effort != "none" && effort != "future" && effort != " HIGH " {
						want = "medium"
					}
					if protocol == "responses" && want != nil && !metadata.thinking.Supports(want) {
						if err == nil || !strings.Contains(err.Error(), "supported values") {
							t.Fatalf("unsupported Responses control needs supported choices: %v", err)
						}
						return
					}
					if err != nil {
						t.Fatal(err)
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

func TestResponsesThinkingBestEffort(t *testing.T) {
	qwen := &model.Thinking{Values: []any{false, "low", "medium", "xhigh"}, Default: "medium"}
	glm := &model.Thinking{Values: []any{"low", "high", "max"}, Default: "max"}
	for _, tt := range []struct {
		name     string
		thinking *model.Thinking
		effort   string
		override any
		want     any
		wantErr  bool
	}{
		{name: "Qwen high rounds up", thinking: qwen, effort: "high", want: "xhigh"},
		{name: "Qwen minimal clamps to low", thinking: qwen, effort: "minimal", want: "low"},
		{name: "Qwen max uses xhigh", thinking: qwen, effort: "max", want: "xhigh"},
		{name: "Qwen ultra uses xhigh", thinking: qwen, effort: "ultra", want: "xhigh"},
		{name: "Qwen supported medium stays exact", thinking: qwen, effort: "medium", want: "medium"},
		{name: "GLM medium rounds up", thinking: glm, effort: "medium", want: "high"},
		{name: "GLM xhigh uses max", thinking: glm, effort: "xhigh", want: "max"},
		{name: "upper boundary clamps to high", thinking: &model.Thinking{Values: []any{"low", "medium", "high"}, Default: "medium"}, effort: "xhigh", want: "high"},
		{name: "metadata order does not affect ties", thinking: &model.Thinking{Values: []any{"xhigh", "medium", "low"}, Default: "low"}, effort: "high", want: "xhigh"},
		{name: "custom name stays exact", thinking: &model.Thinking{Values: []any{"balanced", "fast"}, Default: "fast"}, effort: "balanced", want: "balanced"},
		{name: "custom names cannot be ranked", thinking: &model.Thinking{Values: []any{"balanced", "fast"}, Default: "fast"}, effort: "high", wantErr: true},
		{name: "unknown name is rejected", thinking: qwen, effort: "future", wantErr: true},
		{name: "off stays off", thinking: qwen, effort: "none", want: false},
		{name: "off cannot become on", thinking: glm, effort: "none", wantErr: true},
		{name: "omission stays omitted", thinking: qwen},
		{name: "native level maps too", thinking: qwen, override: "high", want: "xhigh"},
		{name: "native override retains precedence", thinking: qwen, effort: "high", override: false, want: false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			req := ResponsesRequest{Input: ResponsesInput{Text: "hi"}}
			req.Reasoning.Effort = tt.effort
			if tt.override != nil {
				req.Think = &api.ThinkValue{Value: tt.override}
			}
			got, err := FromResponsesRequest(req, tt.thinking)
			if req.Reasoning.Effort != tt.effort || (req.Think != nil && req.Think.Value != tt.override) {
				t.Fatal("conversion changed the caller's thinking setting")
			}
			if tt.wantErr {
				if err == nil || !strings.Contains(err.Error(), "supported values") {
					t.Fatalf("unsupported thinking needs supported choices: %v", err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			var actual any
			if got.Think != nil {
				actual = got.Think.Value
			}
			if actual != tt.want {
				t.Fatalf("thinking=%#v, want %#v", actual, tt.want)
			}
		})
	}
}

func TestResponsesThinkingAliasOrder(t *testing.T) {
	for _, values := range [][]any{
		{"max", "ultra"},
		{"ultra", "max"},
	} {
		t.Run(fmt.Sprint(values), func(t *testing.T) {
			req := ResponsesRequest{Input: ResponsesInput{Text: "hi"}}
			req.Reasoning.Effort = "xhigh"
			got, err := FromResponsesRequest(req, &model.Thinking{Values: values, Default: "max"})
			if err != nil {
				t.Fatal(err)
			}
			if got.Think == nil || got.Think.Value != "max" {
				t.Fatalf("metadata order changed the maximum alias: think=%v", got.Think)
			}
		})
	}
}
