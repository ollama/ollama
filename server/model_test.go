package server

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/template"
)

func readFile(t *testing.T, base, name string) *bytes.Buffer {
	t.Helper()

	bts, err := os.ReadFile(filepath.Join(base, name))
	if err != nil {
		t.Fatal(err)
	}

	return bytes.NewBuffer(bts)
}

func TestExecuteWithTools(t *testing.T) {
	p := filepath.Join("testdata", "tools")
	cases := []struct {
		model  string
		output string
		ok     bool
	}{
		{"mistral", `[TOOL_CALLS]  [{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`, true},
		{"mistral", `[TOOL_CALLS]  [{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]

The temperature in San Francisco, CA is 70°F and in Toronto, Canada is 20°C.`, true},
		{"mistral", `I'm not aware of that information. However, I can suggest searching for the weather using the "get_current_weather" function:

		[{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`, true},
		{"mistral", " The weather in San Francisco, CA is 70°F and in Toronto, Canada is 20°C.", false},
		{"command-r-plus", "Action: ```json" + `
[
    {
        "tool_name": "get_current_weather",
        "parameters": {
            "format": "fahrenheit",
            "location": "San Francisco, CA"
        }
    },
    {
        "tool_name": "get_current_weather",
        "parameters": {
            "format": "celsius",
            "location": "Toronto, Canada"
        }
    }
]
` + "```", true},
		{"command-r-plus", " The weather in San Francisco, CA is 70°F and in Toronto, Canada is 20°C.", false},
		{"firefunction", ` functools[{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`, true},
		{"firefunction", " The weather in San Francisco, CA is 70°F and in Toronto, Canada is 20°C.", false},
		{"llama3-groq-tool-use", `<tool_call>
{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}}
{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}
</tool_call>`, true},
		{"xlam", `{"tool_calls": [{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]}`, true},
		{"nemotron", `<toolcall>{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]} </toolcall>`, true},
	}

	var tools []api.Tool
	if err := json.Unmarshal(readFile(t, p, "tools.json").Bytes(), &tools); err != nil {
		t.Fatal(err)
	}

	var messages []api.Message
	if err := json.Unmarshal(readFile(t, p, "messages.json").Bytes(), &messages); err != nil {
		t.Fatal(err)
	}

	calls := []api.ToolCall{
		{
			Function: api.ToolCallFunction{
				Name: "get_current_weather",
				Arguments: api.ToolCallFunctionArguments{
					"format":   "fahrenheit",
					"location": "San Francisco, CA",
				},
			},
		},
		{
			Function: api.ToolCallFunction{
				Name: "get_current_weather",
				Arguments: api.ToolCallFunctionArguments{
					"format":   "celsius",
					"location": "Toronto, Canada",
				},
			},
		},
	}

	for _, tt := range cases {
		t.Run(tt.model, func(t *testing.T) {
			tmpl, err := template.Parse(readFile(t, p, fmt.Sprintf("%s.gotmpl", tt.model)).String())
			if err != nil {
				t.Fatal(err)
			}

			t.Run("template", func(t *testing.T) {
				var actual bytes.Buffer
				if err := tmpl.Execute(&actual, template.Values{Tools: tools, Messages: messages}); err != nil {
					t.Fatal(err)
				}

				if diff := cmp.Diff(actual.String(), readFile(t, p, fmt.Sprintf("%s.out", tt.model)).String()); diff != "" {
					t.Errorf("mismatch (-got +want):\n%s", diff)
				}
			})

			t.Run("parse", func(t *testing.T) {
				m := &Model{Template: tmpl}
				actual, ok := m.parseToolCalls(tt.output)
				if ok != tt.ok {
					t.Fatalf("expected %t, got %t", tt.ok, ok)
				}

				if tt.ok {
					if diff := cmp.Diff(actual, calls); diff != "" {
						t.Errorf("mismatch (-got +want):\n%s", diff)
					}
				}
			})
		})
	}
}

func TestParseFromFileFromLayer(t *testing.T) {
	tempModels := t.TempDir()
	t.Setenv("OLLAMA_MODELS", tempModels)

	file, err := os.CreateTemp(tempModels, "")
	if err != nil {
		t.Fatalf("failed to open file: %v", err)
	}
	defer file.Close()
	if err := llm.WriteGGUF(file, llm.KV{"general.architecture": "gemma"}, []llm.Tensor{}); err != nil {
		t.Fatalf("failed to write gguf: %v", err)
	}

	if _, err := file.Seek(0, io.SeekStart); err != nil {
		t.Fatalf("failed to seek to start: %v", err)
	}

	layers, err := parseFromFile(context.Background(), "model", []*layerGGML{}, file, "", func(api.ProgressResponse) {})
	if err != nil {
		t.Fatalf("failed to parse from file: %v", err)
	}

	if len(layers) != 1 {
		t.Fatalf("got %d != want 1", len(layers))
	}

	if _, err := file.Seek(0, io.SeekStart); err != nil {
		t.Fatalf("failed to seek to start: %v", err)
	}

	layers2, err := parseFromFile(context.Background(), "model", []*layerGGML{}, file, layers[0].Digest, func(api.ProgressResponse) {})
	if err != nil {
		t.Fatalf("failed to parse from file: %v", err)
	}
	if len(layers2) != 1 {
		t.Fatalf("got %d != want 1", len(layers2))
	}

	if layers[0].Digest != layers2[0].Digest {
		t.Fatalf("got %s != want %s", layers[0].Digest, layers2[0].Digest)
	}

	if layers[0].Size != layers2[0].Size {
		t.Fatalf("got %d != want %d", layers[0].Size, layers2[0].Size)
	}

	if layers[0].MediaType != layers2[0].MediaType {
		t.Fatalf("got %v != want %v", layers[0].MediaType, layers2[0].MediaType)
	}
}

func TestParseLayerFromCopy(t *testing.T) {
	tempModels := t.TempDir()
	t.Setenv("OLLAMA_MODELS", tempModels)

	file2, err := os.CreateTemp(tempModels, "")
	if err != nil {
		t.Fatalf("failed to open file: %v", err)
	}
	defer file2.Close()

	for range 5 {
		if err := llm.WriteGGUF(file2, llm.KV{"general.architecture": "gemma"}, []llm.Tensor{}); err != nil {
			t.Fatalf("failed to write gguf: %v", err)
		}
	}

	if _, err := file2.Seek(0, io.SeekStart); err != nil {
		t.Fatalf("failed to seek to start: %v", err)
	}

	layers, err := parseFromFile(context.Background(), "model", []*layerGGML{}, file2, "", func(api.ProgressResponse) {})
	if err != nil {
		t.Fatalf("failed to parse from file: %v", err)
	}

	if len(layers) != 5 {
		t.Fatalf("got %d != want 5", len(layers))
	}
}

func TestParseObjects(t *testing.T) {
	tests := []struct {
		input string
		want  []map[string]any
	}{
		{
			input: `[{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`,
			want: []map[string]any{
				{"name": "get_current_weather", "arguments": map[string]any{"format": "fahrenheit", "location": "San Francisco, CA"}},
				{"name": "get_current_weather", "arguments": map[string]any{"format": "celsius", "location": "Toronto, Canada"}},
			},
		},
		{
			input: `<toolcall>{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}} </toolcall>`,
			want: []map[string]any{
				{"name": "get_current_weather", "arguments": map[string]any{"format": "fahrenheit", "location": "San Francisco, CA"}},
			},
		},
		{
			input: `<toolcall>{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}} </toolcall> <toolcall>{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, ON"}} </toolcall>`,
			want: []map[string]any{
				{"name": "get_current_weather", "arguments": map[string]any{"format": "fahrenheit", "location": "San Francisco, CA"}},
				{"name": "get_current_weather", "arguments": map[string]any{"format": "celsius", "location": "Toronto, ON"}},
			},
		},
		{
			input: `{"name": "get_current_weather", "arguments": `,
			want:  nil,
		},
	}

	for _, tc := range tests {
		t.Run(tc.input, func(t *testing.T) {
			got := parseObjects(tc.input)

			if diff := cmp.Diff(got, tc.want); diff != "" {
				t.Errorf("mismatch (-got +want):\n%s", diff)
			}
		})
	}
}
