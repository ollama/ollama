package renderers

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

// The fixtures are the publishers' chat_template.jinja files, verbatim and
// pinned by SHA-256; the text and VL checkpoints share one template.
// Each variant renders with its own pinned Transformers version.
// Keep the fixtures and their digests together when updating the source pins.
var lfm3TemplateFixtures = []struct {
	name      string
	path      string
	pythonEnv string
	sha256    string
	renderer  LFM3Renderer
	// media maps each attachment kind the checkpoint accepts to the template's
	// placeholder for it; the runner expands an [img-N] tag the same way.
	media map[string]string
	// knownDifferences names API cases whose output intentionally differs (see lfm3AudioToolMedia).
	knownDifferences map[string]bool
}{
	{name: "text", path: "testdata/lfm3_chat_template.jinja2", pythonEnv: "LFM3_TEXT_JINJA_PYTHON", sha256: "f45005c50cd2a97329722bf3a2dee9f16c04cdf825a589b280fb7e99e6827818", media: map[string]string{"image": lfm3Image}},
	{name: "vl", path: "testdata/lfm3_chat_template.jinja2", pythonEnv: "LFM3_VL_JINJA_PYTHON", sha256: "f45005c50cd2a97329722bf3a2dee9f16c04cdf825a589b280fb7e99e6827818", media: map[string]string{"image": lfm3Image}},
	{name: "omni", path: "testdata/lfm3_omni_chat_template.jinja2", pythonEnv: "LFM3_VL_JINJA_PYTHON", sha256: "ea74cc1d14319ed436dc0f37632893004d436913113e7605267f2b01694d65e4", renderer: LFM3Renderer{noThinking: true}, media: map[string]string{"image": lfm3Image, "audio": "<|vision_start|><|audio|><|vision_end|>"}},
	{name: "audio", path: "testdata/lfm3_audio_chat_template.jinja2", pythonEnv: "LFM3_VL_JINJA_PYTHON", sha256: "8769fc2ac03684665a009b1908fee359e418329019dd83dbc57152167e448099", renderer: LFM3Renderer{audio: true}, media: map[string]string{"audio": "<|audio|>"}, knownDifferences: lfm3AudioToolMedia},
}

// The Audio template JSON-encodes each item of a list-valued tool result, so a
// tool result carrying audio renders as {"type": "audio"} and {"text": ...,
// "type": "text"} lines and the clip never reaches the model. Ollama keeps the
// audio and the plain text, as the newer LFM3 templates do.
var lfm3AudioToolMedia = map[string]bool{
	"tool result audio":                   true,
	"tool result audio only":              true,
	"same call results with audio merged": true,
}

// lfm3AudioClip is an attachment the server classifies as audio (a WAV header).
var lfm3AudioClip = api.ImageData("RIFF\x00\x00\x00\x00WAVEfmt ")

func lfm3MediaKinds(messages []api.Message) []string {
	var kinds []string
	for _, message := range messages {
		for _, data := range message.Images {
			if len(data) >= 12 && string(data[:4]) == "RIFF" && string(data[8:12]) == "WAVE" {
				kinds = append(kinds, "audio")
			} else {
				kinds = append(kinds, "image")
			}
		}
	}
	return kinds
}

func TestLFM3PublisherTemplateFixtures(t *testing.T) {
	for _, fixture := range lfm3TemplateFixtures {
		t.Run(fixture.name, func(t *testing.T) {
			verifyLFM3TemplateFixture(t, fixture.path, fixture.sha256)
		})
	}
}

func verifyLFM3TemplateFixture(t *testing.T, path, wantSHA256 string) {
	t.Helper()
	template, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(template)); got != wantSHA256 {
		t.Fatalf("%s SHA256 = %s, want %s", path, got, wantSHA256)
	}
}

// Set VERIFY_JINJA2=1 and the two LFM3_*_JINJA_PYTHON variables to render
// the checked-in fixtures with the publisher's pinned Transformers versions.
func TestLFM3RendererMatchesJinja2Reference(t *testing.T) {
	if os.Getenv("VERIFY_JINJA2") == "" {
		t.Skip("set VERIFY_JINJA2=1 and the LFM3 Jinja Python interpreters to run publisher template parity")
	}
	args := api.NewToolCallFunctionArguments()
	args.Set("city", "Paris")
	args.Set("days", 2)
	mixedArgs := api.NewToolCallFunctionArguments()
	mixedArgs.Set("enabled", true)
	mixedArgs.Set("disabled", false)
	mixedArgs.Set("optional", nil)
	mixedArgs.Set("cities", []string{"Paris", "Lyon"})
	mixedArgs.Set("limits", map[string]any{"max": 3})
	mixedArgs.Set("temperature", 19.5)
	mixedArgs.Set("label", "café <city>")
	weather := api.Tool{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "get_weather",
			Description: "Get weather",
			Parameters:  api.ToolFunctionParameters{Type: "object"},
		},
	}
	lookup := api.Tool{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "lookup_city",
			Description: "Find <city> & café",
			Parameters: api.ToolFunctionParameters{
				Type: "object",
			},
		},
	}
	call := api.ToolCall{Function: api.ToolCallFunction{Name: "get_weather", Arguments: args}}
	mixedCall := api.ToolCall{Function: api.ToolCallFunction{Name: "get_weather", Arguments: mixedArgs}}
	emptyCall := api.ToolCall{Function: api.ToolCallFunction{Name: "get_weather", Arguments: api.NewToolCallFunctionArguments()}}
	lookupArgs := api.NewToolCallFunctionArguments()
	lookupArgs.Set("name", "Lyon")
	lookupCall := api.ToolCall{Function: api.ToolCallFunction{Name: "lookup_city", Arguments: lookupArgs}}
	properties := api.NewToolPropertiesMap()
	properties.Set("city", api.ToolProperty{Type: api.PropertyType{"string"}, Description: "City name"})
	properties.Set("unit", api.ToolProperty{Type: api.PropertyType{"string"}, Enum: []any{"celsius", "fahrenheit"}})
	properties.Set("days", api.ToolProperty{Type: api.PropertyType{"integer", "null"}, Items: map[string]any{"type": "integer"}})
	forecast := api.Tool{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "get_forecast",
			Description: "Forecast for \"city\" — 3 days",
			Parameters:  api.ToolFunctionParameters{Type: "object", Required: []string{"city"}, Properties: properties},
		},
	}
	image := api.ImageData{1}
	think := func(enabled bool) *api.ThinkValue { return &api.ThinkValue{Value: enabled} }
	tests := []struct {
		name     string
		messages []api.Message
		tools    []api.Tool
		think    *api.ThinkValue
	}{
		{name: "empty transcript"},
		{name: "user only", messages: []api.Message{{Role: "user", Content: "Hello"}}},
		{name: "first system", messages: []api.Message{{Role: "system", Content: "Be concise."}, {Role: "user", Content: "Hello"}}},
		{name: "empty first system", messages: []api.Message{{Role: "system"}, {Role: "user", Content: "Hello"}}},
		{name: "system image", messages: []api.Message{{Role: "system", Content: "Describe the diagram.", Images: []api.ImageData{[]byte{1}}}, {Role: "user", Content: "Go"}}},
		{name: "system image only", messages: []api.Message{{Role: "system", Images: []api.ImageData{[]byte{1}}}, {Role: "user", Content: "Go"}}},
		{name: "system two images", messages: []api.Message{{Role: "system", Content: "Compare.", Images: []api.ImageData{[]byte{1}, []byte{2}}}, {Role: "user", Content: "Go"}}},
		{name: "system image with tools", messages: []api.Message{{Role: "system", Content: "Inspect.", Images: []api.ImageData{[]byte{1}}}, {Role: "user", Content: "Go"}}, tools: []api.Tool{weather}},
		{name: "system only", messages: []api.Message{{Role: "system", Content: "Be concise."}}},
		{name: "additional system", messages: []api.Message{{Role: "system", Content: "First."}, {Role: "user", Content: "Hi"}, {Role: "system", Content: "Second."}}},
		{name: "unknown role passthrough", messages: []api.Message{{Role: "developer", Content: "Use concise prose."}, {Role: "user", Content: "Hi"}}},
		{name: "multi turn reasoning removed", messages: []api.Message{{Role: "user", Content: "Q1"}, {Role: "assistant", Content: "A1", Thinking: "Past thought."}, {Role: "user", Content: "Q2"}}},
		{name: "old inline reasoning removed", messages: []api.Message{{Role: "user", Content: "Q1"}, {Role: "assistant", Content: "<think>Past thought.</think>A1"}, {Role: "user", Content: "Q2"}}},
		{name: "last inline reasoning removed", messages: []api.Message{{Role: "user", Content: "Q1"}, {Role: "assistant", Content: "<think>First.</think><think>Last.</think>A1"}, {Role: "user", Content: "Q2"}}},
		{name: "last assistant reasoning retained", messages: []api.Message{{Role: "user", Content: "Q"}, {Role: "assistant", Content: "A", Thinking: "Current thought."}}},
		{name: "last assistant inline reasoning retained", messages: []api.Message{{Role: "user", Content: "Q"}, {Role: "assistant", Content: "<think>Current thought.</think>A"}}},
		{name: "assistant without user", messages: []api.Message{{Role: "assistant", Content: "A", Thinking: "Current thought."}}},
		{name: "assistant before user", messages: []api.Message{{Role: "assistant", Content: "Prefill", Thinking: "Keep this."}, {Role: "user", Content: "Q"}}},
		{name: "empty assistant after user", messages: []api.Message{{Role: "user", Content: "Q"}, {Role: "assistant"}}},
		{name: "thinking enabled", messages: []api.Message{{Role: "user", Content: "Solve"}}, think: think(true)},
		{name: "thinking disabled", messages: []api.Message{{Role: "user", Content: "Answer"}}, think: think(false)},
		{name: "adversarial schema prompt", messages: []api.Message{{Role: "user", Content: "Say hi"}}, think: think(false)},
		{name: "tools ending user", messages: []api.Message{{Role: "user", Content: "Weather?"}}, tools: []api.Tool{weather}},
		{name: "tools without messages", tools: []api.Tool{weather, lookup}},
		{name: "multiple tools in registry", messages: []api.Message{{Role: "user", Content: "Weather and city?"}}, tools: []api.Tool{weather, lookup}},
		{name: "system with tools", messages: []api.Message{{Role: "system", Content: "Be concise."}, {Role: "user", Content: "Weather?"}}, tools: []api.Tool{weather}},
		{name: "system with tools no user", messages: []api.Message{{Role: "system", Content: "Be concise."}}, tools: []api.Tool{weather}},
		{name: "tools ending assistant", messages: []api.Message{{Role: "user", Content: "Weather?"}, {Role: "assistant", ToolCalls: []api.ToolCall{call}}}, tools: []api.Tool{weather}},
		{name: "empty tool call arguments", messages: []api.Message{{Role: "assistant", ToolCalls: []api.ToolCall{emptyCall}}}},
		{name: "multiple tool calls", messages: []api.Message{{Role: "user", Content: "Two cities"}, {Role: "assistant", ToolCalls: []api.ToolCall{call, mixedCall}}}, tools: []api.Tool{weather}},
		{name: "tools ending result", messages: []api.Message{{Role: "user", Content: "Weather?"}, {Role: "assistant", ToolCalls: []api.ToolCall{call}}, {Role: "tool", Content: `{"temp":19}`}}, tools: []api.Tool{weather}},
		{name: "preformatted result", messages: []api.Message{{Role: "tool", Content: "<tool_response>\nOK\n</tool_response>"}}},
		{name: "embedded preformatted result", messages: []api.Message{{Role: "tool", Content: "prefix <tool_response>\nOK\n</tool_response>"}}},
		{name: "empty tool result", messages: []api.Message{{Role: "tool"}}},
		{name: "tool result image", messages: []api.Message{{Role: "tool", Content: "Found image.", Images: []api.ImageData{[]byte{1}}}}},
		{name: "tool result image only", messages: []api.Message{{Role: "tool", Images: []api.ImageData{[]byte{1}}}}},
		{name: "tool result two images", messages: []api.Message{{Role: "tool", Content: "Found both.", Images: []api.ImageData{[]byte{1}, []byte{2}}}}},
		{name: "tool result image with preformatted text", messages: []api.Message{{Role: "tool", Content: "<tool_response>\nOK\n</tool_response>", Images: []api.ImageData{[]byte{1}}}}},
		{name: "images across turns and tool result", messages: []api.Message{{Role: "system", Images: []api.ImageData{[]byte{1}}}, {Role: "user", Content: "Compare.", Images: []api.ImageData{[]byte{2}, []byte{3}}}, {Role: "assistant", Content: "Checking.", ToolCalls: []api.ToolCall{call}}, {Role: "tool", Content: "Found.", Images: []api.ImageData{[]byte{4}}}, {Role: "user", Content: "Summarize."}}},
		{name: "continuation marker", messages: []api.Message{{Role: "user", Content: "Continue"}, {Role: "assistant", Content: "Part CONTINUE_FINAL_MESSAGE_TAG "}}},
		{name: "continuation with tool call", messages: []api.Message{{Role: "user", Content: "Weather?"}, {Role: "assistant", Content: "Part CONTINUE_FINAL_MESSAGE_TAG ", ToolCalls: []api.ToolCall{call}}}},
		{name: "still image", messages: []api.Message{{Role: "user", Content: "Describe it.", Images: []api.ImageData{[]byte{1}}}}},
		{name: "two still images", messages: []api.Message{{Role: "user", Content: "Compare them.", Images: []api.ImageData{[]byte{1}, []byte{2}}}}},
		{name: "still image only", messages: []api.Message{{Role: "user", Images: []api.ImageData{[]byte{1}}}}},
		{name: "assistant still image", messages: []api.Message{{Role: "assistant", Content: "Here.", Images: []api.ImageData{[]byte{1}}}}},
		{name: "assistant image with inline reasoning", messages: []api.Message{{Role: "user", Content: "Q"}, {Role: "assistant", Content: "<think>Seen.</think>Here.", Images: []api.ImageData{[]byte{1}}}}},
		{name: "agentic loop after user", messages: []api.Message{{Role: "user", Content: "Weather?"}, {Role: "assistant", ToolCalls: []api.ToolCall{call}}, {Role: "tool", Content: `{"temp":19}`, ToolCallID: "call_1"}, {Role: "assistant", Content: "It is 19."}}, tools: []api.Tool{weather}},
		{name: "same call results merged", messages: []api.Message{{Role: "user", Content: "Weather?"}, {Role: "assistant", ToolCalls: []api.ToolCall{call}}, {Role: "tool", Content: "part one", ToolCallID: "call_1"}, {Role: "tool", Content: "part two", ToolCallID: "call_1"}}, tools: []api.Tool{weather}},
		{name: "three same call results merged", messages: []api.Message{{Role: "tool", Content: "one", ToolCallID: "call_1"}, {Role: "tool", Content: "two", ToolCallID: "call_1"}, {Role: "tool", Content: "three", ToolCallID: "call_1"}}},
		{name: "same call results with images merged", messages: []api.Message{{Role: "tool", Content: "first", ToolCallID: "call_1", Images: []api.ImageData{[]byte{1}}}, {Role: "tool", ToolCallID: "call_1", Images: []api.ImageData{[]byte{2}}}}},
		{name: "same call empty first result", messages: []api.Message{{Role: "tool", ToolCallID: "call_1"}, {Role: "tool", Content: "only", ToolCallID: "call_1"}}},
		{name: "same call empty last result", messages: []api.Message{{Role: "tool", Content: "only", ToolCallID: "call_1"}, {Role: "tool", ToolCallID: "call_1"}}},
		{name: "same call preformatted results merged", messages: []api.Message{{Role: "tool", Content: "<tool_response>\nA\n</tool_response>", ToolCallID: "call_1"}, {Role: "tool", Content: "<tool_response>\nB\n</tool_response>", ToolCallID: "call_1"}}},
		{name: "different call results separate", messages: []api.Message{{Role: "tool", Content: "first", ToolCallID: "call_1"}, {Role: "tool", Content: "second", ToolCallID: "call_2"}}},
		{name: "results without call ids separate", messages: []api.Message{{Role: "tool", Content: "first"}, {Role: "tool", Content: "second"}}},
		{name: "same call results split by assistant", messages: []api.Message{{Role: "tool", Content: "first", ToolCallID: "call_1"}, {Role: "assistant", Content: "Next."}, {Role: "tool", Content: "second", ToolCallID: "call_1"}}},
		{name: "same call empty middle result", messages: []api.Message{{Role: "tool", Content: "one", ToolCallID: "call_1"}, {Role: "tool", ToolCallID: "call_1"}, {Role: "tool", Content: "three", ToolCallID: "call_1"}}},
		{name: "same call all results empty", messages: []api.Message{{Role: "tool", ToolCallID: "call_1"}, {Role: "tool", ToolCallID: "call_1"}}},

		// Empty content in every role.
		{name: "empty user message", messages: []api.Message{{Role: "user"}}},
		{name: "empty later system message", messages: []api.Message{{Role: "system", Content: "First."}, {Role: "user", Content: "Hi"}, {Role: "system"}}},
		{name: "empty assistant in history", messages: []api.Message{{Role: "user", Content: "Q1"}, {Role: "assistant"}, {Role: "user", Content: "Q2"}}},
		{name: "reasoning only assistant in history", messages: []api.Message{{Role: "user", Content: "Q1"}, {Role: "assistant", Thinking: "Thought."}, {Role: "user", Content: "Q2"}}},

		// Thinking switched across turns.
		{name: "thinking disabled after reasoning turn", messages: []api.Message{{Role: "user", Content: "Q1"}, {Role: "assistant", Content: "A1", Thinking: "T1"}, {Role: "user", Content: "Q2"}}, think: think(false)},
		{name: "thinking enabled after direct turn", messages: []api.Message{{Role: "user", Content: "Q1"}, {Role: "assistant", Content: "A1"}, {Role: "user", Content: "Q2"}}, think: think(true)},
		{name: "thinking disabled mid tool loop", messages: []api.Message{{Role: "user", Content: "Weather?"}, {Role: "assistant", Thinking: "Need the tool.", ToolCalls: []api.ToolCall{call}}, {Role: "tool", Content: "19C", ToolCallID: "call_1"}}, tools: []api.Tool{weather}, think: think(false)},
		{name: "thinking disabled with tools", messages: []api.Message{{Role: "user", Content: "Weather?"}}, tools: []api.Tool{weather, forecast}, think: think(false)},

		// Reasoning tags inside assistant content.
		{name: "unclosed inline reasoning in history", messages: []api.Message{{Role: "user", Content: "Q1"}, {Role: "assistant", Content: "<think>Partial"}, {Role: "user", Content: "Q2"}}},
		{name: "close tag only in history", messages: []api.Message{{Role: "user", Content: "Q1"}, {Role: "assistant", Content: "Reasoning.</think>A1"}, {Role: "user", Content: "Q2"}}},
		{name: "inline reasoning whitespace trimmed in history", messages: []api.Message{{Role: "user", Content: "Q1"}, {Role: "assistant", Content: "<think>T</think>\n\n A1 　 \n"}, {Role: "user", Content: "Q2"}}},
		{name: "inline reasoning with tool call in history", messages: []api.Message{{Role: "user", Content: "Q1"}, {Role: "assistant", Content: "<think>Use the tool.</think>Checking.", ToolCalls: []api.ToolCall{call}}, {Role: "tool", Content: "19C", ToolCallID: "call_1"}, {Role: "user", Content: "Q2"}}, tools: []api.Tool{weather}},
		{name: "reasoning field and inline reasoning in tool loop", messages: []api.Message{{Role: "user", Content: "Q"}, {Role: "assistant", Thinking: "Field.", Content: "<think>Inline.</think>Calling.", ToolCalls: []api.ToolCall{call}}, {Role: "tool", Content: "19C", ToolCallID: "call_1"}}, tools: []api.Tool{weather}},
		{name: "unclosed inline reasoning in tool loop", messages: []api.Message{{Role: "user", Content: "Q"}, {Role: "assistant", Content: "<think>Partial", ToolCalls: []api.ToolCall{call}}, {Role: "tool", Content: "19C", ToolCallID: "call_1"}}, tools: []api.Tool{weather}},

		// Tool-call chains and parallel calls.
		{name: "tool call chain with reasoning", messages: []api.Message{{Role: "user", Content: "Weather in Lyon?"}, {Role: "assistant", Thinking: "Find the city.", ToolCalls: []api.ToolCall{lookupCall}}, {Role: "tool", Content: `{"id":7}`, ToolCallID: "call_1"}, {Role: "assistant", Thinking: "Now the weather.", ToolCalls: []api.ToolCall{call}}, {Role: "tool", Content: "19C", ToolCallID: "call_2"}}, tools: []api.Tool{weather, lookup}},
		{name: "tool call chain answered then new question", messages: []api.Message{{Role: "user", Content: "Weather?"}, {Role: "assistant", Thinking: "Use the tool.", ToolCalls: []api.ToolCall{call}}, {Role: "tool", Content: "19C", ToolCallID: "call_1"}, {Role: "assistant", Thinking: "Answer now.", Content: "It is 19C."}, {Role: "user", Content: "And tomorrow?"}}, tools: []api.Tool{weather}},
		{name: "parallel tool calls and results", messages: []api.Message{{Role: "user", Content: "Weather and city?"}, {Role: "assistant", Content: "Both.", ToolCalls: []api.ToolCall{call, lookupCall}}, {Role: "tool", Content: "19C", ToolCallID: "call_1"}, {Role: "tool", Content: `{"id":7}`, ToolCallID: "call_2"}}, tools: []api.Tool{weather, lookup}},
		{name: "full tool schema", messages: []api.Message{{Role: "user", Content: "Forecast?"}}, tools: []api.Tool{forecast}},

		// A trailing assistant message is continued (Transformers continue_final_message).
		{name: "prefill continues answer", messages: []api.Message{{Role: "user", Content: "Q"}, {Role: "assistant", Content: "The answer is"}}},
		{name: "prefill with thinking disabled", messages: []api.Message{{Role: "user", Content: "Q"}, {Role: "assistant", Content: "Sure, "}}, think: think(false)},
		{name: "prefill with thinking enabled", messages: []api.Message{{Role: "user", Content: "Q"}, {Role: "assistant", Content: "Sure, "}}, think: think(true)},
		{name: "prefill with reasoning", messages: []api.Message{{Role: "user", Content: "Q"}, {Role: "assistant", Thinking: "Short.", Content: "So"}}},
		{name: "prefill with tool call", messages: []api.Message{{Role: "user", Content: "Weather?"}, {Role: "assistant", Content: "Checking.", ToolCalls: []api.ToolCall{call}}}, tools: []api.Tool{weather}},
		{name: "prefill after tool result", messages: []api.Message{{Role: "user", Content: "Weather?"}, {Role: "assistant", ToolCalls: []api.ToolCall{call}}, {Role: "tool", Content: "19C", ToolCallID: "call_1"}, {Role: "assistant", Content: "Based on"}}, tools: []api.Tool{weather}},
		{name: "prefill with image", messages: []api.Message{{Role: "user", Content: "Q"}, {Role: "assistant", Content: "See", Images: []api.ImageData{image}}}},
		{name: "continuation marker before tool result", messages: []api.Message{{Role: "user", Content: "Weather?"}, {Role: "assistant", Content: "Part CONTINUE_FINAL_MESSAGE_TAG ", ToolCalls: []api.ToolCall{call}}, {Role: "tool", Content: "19C"}}},

		// Media placement and kinds.
		{name: "image markers interleaved", messages: []api.Message{{Role: "user", Content: "Left [img] right [img] end", Images: []api.ImageData{image, image}}}},
		{name: "more images than markers", messages: []api.Message{{Role: "user", Content: "Only [img] here", Images: []api.ImageData{image, image}}}},
		{name: "more markers than images", messages: []api.Message{{Role: "user", Content: "[img] and [img]", Images: []api.ImageData{image}}}},
		{name: "tool result image markers", messages: []api.Message{{Role: "tool", Content: "Before [img] after", Images: []api.ImageData{image}}}},
		{name: "tool result image marker first", messages: []api.Message{{Role: "tool", Content: "[img] caption", Images: []api.ImageData{image}}}},
		{name: "user audio", messages: []api.Message{{Role: "user", Content: "What is said?", Images: []api.ImageData{lfm3AudioClip}}}},
		{name: "transcription prompt", messages: []api.Message{{Role: "system", Content: "Perform ASR."}, {Role: "user", Images: []api.ImageData{lfm3AudioClip}}}, think: think(false)},
		{name: "system audio", messages: []api.Message{{Role: "system", Content: "Reference voice.", Images: []api.ImageData{lfm3AudioClip}}, {Role: "user", Content: "Match it."}}},
		{name: "audio marker", messages: []api.Message{{Role: "user", Content: "Clip [img] then question.", Images: []api.ImageData{lfm3AudioClip}}}},
		{name: "tool result audio", messages: []api.Message{{Role: "tool", Content: "Recorded.", Images: []api.ImageData{lfm3AudioClip}}}},
		{name: "tool result audio only", messages: []api.Message{{Role: "tool", Images: []api.ImageData{lfm3AudioClip}}}},
		{name: "same call results with audio merged", messages: []api.Message{{Role: "tool", Content: "first", ToolCallID: "call_1", Images: []api.ImageData{lfm3AudioClip}}, {Role: "tool", Content: "second", ToolCallID: "call_1"}}},
		{name: "image and audio across turns", messages: []api.Message{{Role: "system", Images: []api.ImageData{image}}, {Role: "user", Content: "Compare.", Images: []api.ImageData{lfm3AudioClip, image}}, {Role: "assistant", Content: "Checking.", ToolCalls: []api.ToolCall{call}}, {Role: "tool", Content: "Found.", Images: []api.ImageData{lfm3AudioClip}}, {Role: "user", Content: "Summarize."}}},
	}

	for _, variant := range lfm3TemplateFixtures {
		verifyLFM3TemplateFixture(t, variant.path, variant.sha256)
		python := os.Getenv(variant.pythonEnv)
		if python == "" {
			t.Fatalf("%s must name the pinned Transformers Python", variant.pythonEnv)
		}
		for _, tc := range tests {
			t.Run(variant.name+"/"+tc.name, func(t *testing.T) {
				kinds := lfm3MediaKinds(tc.messages)
				for _, kind := range kinds {
					if _, ok := variant.media[kind]; !ok {
						t.Skipf("the %s checkpoint rejects %s input", variant.name, kind)
					}
				}
				want := lfm3RenderJinja2(t, python, variant.path, tc.messages, tc.tools, tc.think)
				// The Audio template has no thinking switch; its README adds no
				// prefill for ASR, "<think></think>" to answer directly and
				// "<think>" to reason.
				if variant.renderer.audio && !lfm3Continues(tc.messages) && tc.think != nil {
					if tc.think.Bool() {
						want += "<think>"
					} else {
						want += "<think></think>"
					}
				}
				if variant.knownDifferences[tc.name] {
					// See lfm3AudioToolMedia: only the JSON-encoded parts may differ.
					fixed := strings.ReplaceAll(want, `{"type": "audio"}`, variant.media["audio"])
					fixed = lfm3JSONTextPart.ReplaceAllStringFunc(fixed, func(part string) string {
						var text string
						if err := json.Unmarshal([]byte(lfm3JSONTextPart.FindStringSubmatch(part)[1]), &text); err != nil {
							t.Fatal(err)
						}
						return text
					})
					if fixed == want {
						t.Fatalf("known media difference no longer present: %q", want)
					}
					want = fixed
				}
				// The server renders [img-N] tags, which the runner expands per kind;
				// the plain renderer writes the image placeholder itself.
				for _, imgTags := range []bool{true, false} {
					if !imgTags && slices.Contains(kinds, "audio") {
						continue
					}
					renderer := variant.renderer
					renderer.useImgTags = imgTags
					got, err := renderer.Render(tc.messages, tc.tools, tc.think)
					if err != nil {
						t.Fatal(err)
					}
					got = lfm3ImgTag.ReplaceAllStringFunc(got, func(tag string) string {
						n, _ := strconv.Atoi(lfm3ImgTag.FindStringSubmatch(tag)[1])
						return variant.media[kinds[n]]
					})
					if got != want {
						t.Fatalf("renderer (img tags %v) differs from publisher Jinja2 template:\ngot:  %q\nwant: %q", imgTags, got, want)
					}
				}
			})
		}
	}
}

var (
	lfm3ImgTag       = regexp.MustCompile(`\[img-(\d+)\]`)
	lfm3JSONTextPart = regexp.MustCompile(`\{"text": ("(?:[^"\\]|\\.)*"), "type": "text"\}`)
)

// lfm3Continues reports whether Render continues the final message.
func lfm3Continues(messages []api.Message) bool {
	return len(messages) > 0 && messages[len(messages)-1].Role == "assistant" && messages[len(messages)-1].Content != ""
}

// These inputs are supported by the publisher templates, but do not have an
// equivalent in api.Message or api.Tool. They establish the pinned reference
// behavior without claiming Ollama renderer parity for these branches.
func lfm3RenderJinja2(t *testing.T, python, templatePath string, messages []api.Message, tools []api.Tool, think *api.ThinkValue) string {
	t.Helper()
	// This maps API fields to template fields without changing renderer policy.
	// Attachments become content items of their kind: at an [img] marker, or
	// preceding the text. A trailing assistant message with content is a
	// prefill, which Transformers renders with continue_final_message.
	kinds := lfm3MediaKinds(messages)
	inputMessages := make([]map[string]any, 0, len(messages))
	for _, message := range messages {
		item := map[string]any{"role": message.Role, "content": message.Content}
		if message.Thinking != "" {
			item["thinking"] = message.Thinking
		}
		if len(message.Images) > 0 {
			media := make([]map[string]any, len(message.Images))
			for i := range media {
				media[i] = map[string]any{"type": kinds[0]}
				kinds = kinds[1:]
			}
			parts := strings.Split(message.Content, "[img]")
			marked := min(len(parts)-1, len(media))
			content := media[marked:]
			text := parts[0]
			for i, part := range parts[1:] {
				if i >= marked {
					text += "[img]" + part
					continue
				}
				if text != "" {
					content = append(content, map[string]any{"type": "text", "text": text})
				}
				content = append(content, media[i])
				text = part
			}
			if text != "" {
				content = append(content, map[string]any{"type": "text", "text": text})
			}
			item["content"] = content
		}
		if len(message.ToolCalls) > 0 {
			item["tool_calls"] = message.ToolCalls
		}
		if message.ToolCallID != "" {
			item["tool_call_id"] = message.ToolCallID
		}
		inputMessages = append(inputMessages, item)
	}
	input := map[string]any{
		"messages":              inputMessages,
		"tools":                 tools,
		"bos_token":             lfm3BOS,
		"add_generation_prompt": true,
	}
	if lfm3Continues(messages) {
		input["add_generation_prompt"] = false
		input["continue_final_message"] = true
	}
	if think != nil {
		input["enable_thinking"] = think.Bool()
	}
	output, err := lfm3RunJinja2(t, python, templatePath, input)
	if err != nil {
		t.Fatalf("render publisher template: %v\n%s", err, output)
	}
	return output
}

// lfm3RunJinja2 renders through Transformers' render_jinja_template, which
// also implements continue_final_message.
func lfm3RunJinja2(t *testing.T, python, templatePath string, input map[string]any) (string, error) {
	t.Helper()
	data, err := json.Marshal(input)
	if err != nil {
		t.Fatal(err)
	}
	const script = `
import json
import sys
from pathlib import Path
from transformers.utils.chat_template_utils import render_jinja_template

kwargs = json.load(sys.stdin)
messages = kwargs.pop("messages", [])
tools = kwargs.pop("tools", None)
rendered, _ = render_jinja_template([messages], tools=tools, chat_template=Path(sys.argv[1]).read_text(), **kwargs)
print(rendered[0], end="")
`
	cmd := exec.Command(python, "-c", script, templatePath)
	cmd.Stdin = bytes.NewReader(data)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	output, err := cmd.Output()
	if err != nil {
		return string(output) + stderr.String(), err
	}
	return string(output), nil
}
