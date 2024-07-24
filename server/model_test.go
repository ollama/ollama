package server

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
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
	t.Setenv("OLLAMA_MODELS",tempModels)
	envconfig.LoadConfig()
	digest := "sha256-fb9d435dc2c4fe681ce63917c062c91022524e9ce57474c9b10ef5169495d902"

	_, err := GetBlobsPath(digest)
	if err != nil {
		t.Fatalf("failed to get blobs path: %v", err)
	}

	file, err := os.CreateTemp(tempModels+"/blobs", digest)
	if err != nil {
		t.Fatalf("failed to open file: %v", err)
	}
	defer file.Close()

	sGGUF := llm.NewGGUFV3(binary.LittleEndian)
	kv := make(llm.KV)
	tensors := []llm.Tensor{}

	if err := sGGUF.Encode(file, kv, tensors); err != nil {
		t.Fatalf("failed to encode gguf: %v", err)
	}

	layers, err := parseFromFile(context.Background(), file, digest, func(api.ProgressResponse) {})
	if err != nil {
		t.Fatalf("failed to parse from file: %v", err)
	}
	
	fmt.Println(layers)
	// assert something here i don't know yet

	t.Run("2x gguf", func(t *testing.T) {
		digest := "sha256-fb9d435dc2c4fe681ce63917c062c91022524e9ce57474c9b10ef5169495d903"

		_, err := GetBlobsPath(digest)
		if err != nil {
			t.Fatalf("failed to get blobs path: %v", err)
		}

		file2, err := os.CreateTemp(tempModels+"/blobs", digest)
		if err != nil {
			t.Fatalf("failed to open file: %v", err)
		}
		defer file2.Close()

		num1GGUF := llm.NewGGUFV3(binary.LittleEndian)
		num2GGUF := llm.NewGGUFV3(binary.LittleEndian)
		kv := make(llm.KV)
		tensors := []llm.Tensor{}

		if err := num1GGUF.Encode(file2, kv, tensors); err != nil {
			t.Fatalf("failed to encode gguf1: %v", err)
		}

		if err := num2GGUF.Encode(file2, kv, tensors); err != nil {
			t.Fatalf("failed to encode gguf2: %v", err)
		}

		layers, err := parseFromFile(context.Background(), file2, digest, func(api.ProgressResponse) {})
		if err != nil {
			t.Fatalf("failed to parse from file: %v", err)
		}

		// assert on layers
		fmt.Println(layers)
	})
}
