package renderers

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/x/tokenizer"
)

const (
	qwen38Template = "testdata/qwen38_chat_template.jinja"

	qwen38RefXHigh = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer."
	qwen38RefLow   = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration."
)

func TestQwen38RendererMatchesReferenceFlows(t *testing.T) {
	think := func(value any) *api.ThinkValue { return &api.ThinkValue{Value: value} }
	verifyJinja := os.Getenv("VERIFY_JINJA2") != ""
	if verifyJinja {
		requireQwen38Jinja(t)
		t.Log("VERIFY_JINJA2=1: verifying expected values against the Qwen3.8 Jinja template")
	}

	weatherArgs := api.NewToolCallFunctionArguments()
	weatherArgs.Set("city", "Montréal")
	weather := []api.Tool{
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "get_weather",
				Description: "Get weather",
				Parameters: api.ToolFunctionParameters{
					Type:     "object",
					Required: []string{"city"},
					Properties: testPropsOrdered([]orderedProp{{
						Key:   "city",
						Value: api.ToolProperty{Type: api.PropertyType{"string"}},
					}}),
				},
			},
		},
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "get_uv",
				Description: "Get UV index",
				Parameters: api.ToolFunctionParameters{
					Type:     "object",
					Required: []string{"city"},
					Properties: testPropsOrdered([]orderedProp{{
						Key:   "city",
						Value: api.ToolProperty{Type: api.PropertyType{"string"}},
					}}),
				},
			},
		},
	}

	toolHeader := `<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "required": ["city"], "properties": {"city": {"type": "string"}}}}}
{"type": "function", "function": {"name": "get_uv", "description": "Get UV index", "parameters": {"type": "object", "required": ["city"], "properties": {"city": {"type": "string"}}}}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT><|im_end|>
`
	developerToolHeader := strings.TrimSuffix(toolHeader, imEndTag+"\n") + "\n\nUse tools when requested." + imEndTag + "\n"

	tests := []struct {
		name        string
		messages    []api.Message
		tools       []api.Tool
		think       *api.ThinkValue
		jinjaEffort string
		want        string
	}{
		{
			name:     "default xhigh injects system guidance",
			messages: []api.Message{{Role: "user", Content: "Hello"}},
			want: `<|im_start|>system
` + qwen38RefXHigh + `<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
<think>
`,
		},
		{
			name: "developer instruction maps to system",
			messages: []api.Message{
				{Role: "developer", Content: "Use Go."},
				{Role: "user", Content: "Hello"},
			},
			want: `<|im_start|>system
` + qwen38RefXHigh + `

Use Go.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
<think>
`,
		},
		{
			name: "system and developer instructions merge",
			messages: []api.Message{
				{Role: "system", Content: "Base policy."},
				{Role: "developer", Content: "Use Go."},
				{Role: "user", Content: "Hello"},
			},
			want: `<|im_start|>system
` + qwen38RefXHigh + `

Base policy.

Use Go.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
<think>
`,
		},
		{
			name: "multiple leading system instructions merge",
			messages: []api.Message{
				{Role: "system", Content: "Base policy."},
				{Role: "system", Content: "Request policy."},
				{Role: "user", Content: "Hello"},
			},
			want: `<|im_start|>system
` + qwen38RefXHigh + `

Base policy.

Request policy.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
<think>
`,
		},
		{
			name: "request system after base conversation merges with base policy",
			messages: []api.Message{
				{Role: "system", Content: "Base policy."},
				{Role: "user", Content: "Base question"},
				{Role: "assistant", Content: "Base answer"},
				{Role: "system", Content: "Request policy."},
				{Role: "user", Content: "Current question"},
			},
			want: `<|im_start|>system
` + qwen38RefXHigh + `

Base policy.

Request policy.<|im_end|>
<|im_start|>user
Base question<|im_end|>
<|im_start|>assistant
<think>

</think>

Base answer<|im_end|>
<|im_start|>user
Current question<|im_end|>
<|im_start|>assistant
<think>
`,
		},
		{
			name: "boolean true uses API medium effort",
			messages: []api.Message{
				{Role: "system", Content: " Be concise. \n"},
				{Role: "user", Content: "Hello"},
			},
			think:       think(true),
			jinjaEffort: "medium",
			want: `<|im_start|>system
Be concise.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
<think>
`,
		},
		{
			name:        "low effort injects concise guidance",
			messages:    []api.Message{{Role: "user", Content: "Hello"}},
			think:       think("low"),
			jinjaEffort: "low",
			want: `<|im_start|>system
` + qwen38RefLow + `<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
<think>
`,
		},
		{
			name:     "thinking disabled emits explicit empty block",
			messages: []api.Message{{Role: "user", Content: "Hello"}},
			think:    think(false),
			want: `<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
<think>

</think>

`,
		},
		{
			name: "preserves reasoning metadata without extracting content tags",
			messages: []api.Message{
				{Role: "user", Content: "First"},
				{Role: "assistant", Thinking: "Plan", Content: "<think>literal</think>\nAnswer"},
				{Role: "user", Content: "Next"},
			},
			want: `<|im_start|>system
` + qwen38RefXHigh + `<|im_end|>
<|im_start|>user
First<|im_end|>
<|im_start|>assistant
<think>
Plan
</think>

<think>literal</think>
Answer<|im_end|>
<|im_start|>user
Next<|im_end|>
<|im_start|>assistant
<think>
`,
		},
		{
			name: "developer instruction with tools",
			messages: []api.Message{
				{Role: "developer", Content: "Use tools when requested."},
				{Role: "user", Content: "Check the weather."},
			},
			tools:       weather,
			think:       think(true),
			jinjaEffort: "medium",
			want: developerToolHeader + `<|im_start|>user
Check the weather.<|im_end|>
<|im_start|>assistant
<think>
`,
		},
		{
			name: "tool call result and next generation prompt",
			messages: []api.Message{
				{Role: "user", Content: "Weather?"},
				{
					Role:     "assistant",
					Thinking: "Need current data.",
					Content:  "I'll check.",
					ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{
						Name:      "get_weather",
						Arguments: weatherArgs,
					}}},
				},
				{Role: "tool", Content: `{"temp": 18}`},
			},
			tools:       weather,
			think:       think(true),
			jinjaEffort: "medium",
			want: toolHeader + `<|im_start|>user
Weather?<|im_end|>
<|im_start|>assistant
<think>
Need current data.
</think>

I'll check.

<tool_call>
<function=get_weather>
<parameter=city>
Montréal
</parameter>
</function>
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"temp": 18}
</tool_response><|im_end|>
<|im_start|>assistant
<think>
`,
		},
		{
			name: "image content",
			messages: []api.Message{{
				Role:    "user",
				Content: "Describe this.",
				Images:  []api.ImageData{{1}},
			}},
			think: think(false),
			want: `<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Describe this.<|im_end|>
<|im_start|>assistant
<think>

</think>

`,
		},
	}

	renderer := newQwen38Renderer()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := renderer.Render(tt.messages, tt.tools, tt.think)
			if err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Fatalf("renderer output mismatch (-want +got):\n%s", diff)
			}
			if verifyJinja {
				normalized, err := normalizeQwen38Messages(tt.messages)
				if err != nil {
					t.Fatal(err)
				}
				jinja := renderQwen38Jinja(t, normalized, tt.tools, tt.think, tt.jinjaEffort)
				if diff := cmp.Diff(jinja, tt.want); diff != "" {
					t.Fatalf("hardcoded expected mismatch vs Jinja (-jinja +want):\n%s", diff)
				}
				if diff := cmp.Diff(jinja, got); diff != "" {
					t.Fatalf("renderer output mismatch vs Jinja (-jinja +got):\n%s", diff)
				}
			}
		})
	}
}

func TestQwen38RendererReasoningEffortMapping(t *testing.T) {
	tests := []struct {
		name  string
		think *api.ThinkValue
		want  string
	}{
		{name: "unset", want: qwen38RefXHigh},
		{name: "true", think: &api.ThinkValue{Value: true}, want: ""},
		{name: "false", think: &api.ThinkValue{Value: false}, want: ""},
		{name: "low", think: &api.ThinkValue{Value: "low"}, want: qwen38RefLow},
		{name: "medium", think: &api.ThinkValue{Value: "medium"}, want: ""},
		{name: "high", think: &api.ThinkValue{Value: "high"}, want: qwen38RefXHigh},
		{name: "max", think: &api.ThinkValue{Value: "max"}, want: qwen38RefXHigh},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := qwen38ReasoningInstructions(tt.think)
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Fatalf("reasoning instructions = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestQwen38RendererRejectsInvalidTranscripts(t *testing.T) {
	tests := []struct {
		name     string
		messages []api.Message
		wantErr  string
	}{
		{name: "empty", wantErr: "no messages provided"},
		{
			name:     "no user query",
			messages: []api.Message{{Role: "system", Content: "Hello"}},
			wantErr:  "no user query found in messages",
		},
		{
			name: "system image",
			messages: []api.Message{
				{Role: "system", Images: []api.ImageData{{1}}},
				{Role: "user", Content: "Hello"},
			},
			wantErr: "system message cannot contain images",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := newQwen38Renderer().Render(tt.messages, nil, nil)
			if err == nil || err.Error() != tt.wantErr {
				t.Fatalf("Render() error = %v, want %q", err, tt.wantErr)
			}
		})
	}
}

func TestQwen38RendererAssistantPrefillKnownJinjaDifference(t *testing.T) {
	messages := []api.Message{
		{Role: "user", Content: "Complete this"},
		{Role: "assistant", Thinking: "Draft", Content: "Partial"},
	}
	got, err := newQwen38Renderer().Render(messages, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	want := `<|im_start|>system
` + qwen38RefXHigh + `<|im_end|>
<|im_start|>user
Complete this<|im_end|>
<|im_start|>assistant
<think>
Draft
</think>

Partial`
	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatalf("assistant prefill mismatch (-want +got):\n%s", diff)
	}

	if os.Getenv("VERIFY_JINJA2") == "" {
		return
	}
	requireQwen38Jinja(t)
	jinja := renderQwen38Jinja(t, messages, nil, nil, "")
	wantJinja := got + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
	if diff := cmp.Diff(wantJinja, jinja); diff != "" {
		t.Fatalf("assistant prefill Jinja difference changed (-want +jinja):\n%s", diff)
	}
}

func TestQwen38TokenizerMatchesReference(t *testing.T) {
	modelDir := os.Getenv("QWEN38_MODEL_DIR")
	if modelDir == "" {
		t.Skip("set QWEN38_MODEL_DIR to run Qwen3.8 production/reference tokenizer parity")
	}
	requireQwen38Jinja(t)

	args := api.NewToolCallFunctionArguments()
	args.Set("path", "résumé.go")
	args.Set("options", map[string]any{"indent": 2, "strict": true})
	tools := []api.Tool{{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "read_file",
			Description: "Read source code",
			Parameters: api.ToolFunctionParameters{
				Type:     "object",
				Required: []string{"path"},
				Properties: testPropsOrdered([]orderedProp{
					{Key: "path", Value: api.ToolProperty{Type: api.PropertyType{"string"}}},
					{Key: "options", Value: api.ToolProperty{Type: api.PropertyType{"object"}}},
				}),
			},
		},
	}}

	renderer := newQwen38Renderer()
	thinkFalse := &api.ThinkValue{Value: false}
	cases := []struct {
		name     string
		messages []api.Message
		tools    []api.Tool
		think    *api.ThinkValue
	}{
		{
			name: "whitespace punctuation contractions and unicode",
			messages: []api.Message{{
				Role:    "user",
				Content: "Don't normalize naïve café — punctuation !\n\tline 2:  x += 1;\n世界 🌍",
			}},
			think: thinkFalse,
		},
		{
			name: "tool JSON and source result",
			messages: []api.Message{
				{Role: "user", Content: "Read the file."},
				{Role: "assistant", Thinking: "Need source.", ToolCalls: []api.ToolCall{{
					Function: api.ToolCallFunction{Name: "read_file", Arguments: args},
				}}},
				{Role: "tool", Content: "{\"ok\":true,\"source\":\"func main() {\\n\\tfmt.Println(\\\"héllo\\\")\\n}\"}"},
			},
			tools: tools,
		},
		{
			name: "special-token-looking strings",
			messages: []api.Message{{
				Role:    "user",
				Content: "Literal markers: <|im_start|> <|im_end|> <think>not control</think> <|audio_start|>.",
			}},
			think: thinkFalse,
		},
	}

	prompts := make([]string, 0, len(cases))
	for _, tc := range cases {
		prompt, err := renderer.Render(tc.messages, tc.tools, tc.think)
		if err != nil {
			t.Fatalf("render %q: %v", tc.name, err)
		}
		prompts = append(prompts, prompt)
	}

	read := func(name string) []byte {
		t.Helper()
		data, err := os.ReadFile(filepath.Join(modelDir, name))
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		return data
	}
	production, err := tokenizer.LoadFromBytesWithConfig(read("tokenizer.json"), &tokenizer.TokenizerConfig{
		ConfigJSON:           read("config.json"),
		TokenizerConfigJSON:  read("tokenizer_config.json"),
		GenerationConfigJSON: read("generation_config.json"),
	})
	if err != nil {
		t.Fatal(err)
	}

	promptsJSON, err := json.Marshal(prompts)
	if err != nil {
		t.Fatal(err)
	}
	script := `
import json
import sys
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
prompts = json.loads(sys.argv[2])
print(json.dumps([tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]))
`
	cmd := qwen38PythonCommand(t, "-c", script, modelDir, string(promptsJSON))
	output, err := cmd.Output()
	if err != nil {
		t.Fatalf("reference tokenization failed: %v", err)
	}
	var reference [][]int32
	if err := json.Unmarshal(output, &reference); err != nil {
		t.Fatalf("decode reference token IDs: %v", err)
	}
	if len(reference) != len(cases) {
		t.Fatalf("reference returned %d cases, want %d", len(reference), len(cases))
	}

	for i, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := production.Encode(prompts[i], false)
			if diff := cmp.Diff(reference[i], got); diff != "" {
				t.Fatalf("production token IDs differ from reference (-reference +production):\n%s", diff)
			}
		})
	}
}

type qwen38JinjaMessage struct {
	Role             string         `json:"role"`
	Content          any            `json:"content"`
	ReasoningContent string         `json:"reasoning_content,omitempty"`
	ToolCalls        []api.ToolCall `json:"tool_calls,omitempty"`
}

type qwen38JinjaContentPart struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

func renderQwen38Jinja(t *testing.T, messages []api.Message, tools []api.Tool, think *api.ThinkValue, effort string) string {
	t.Helper()

	jinjaMessages := make([]qwen38JinjaMessage, 0, len(messages))
	for _, message := range messages {
		content := any(message.Content)
		if len(message.Images) > 0 {
			parts := make([]qwen38JinjaContentPart, 0, len(message.Images)+1)
			for range message.Images {
				parts = append(parts, qwen38JinjaContentPart{Type: "image"})
			}
			if message.Content != "" {
				parts = append(parts, qwen38JinjaContentPart{Type: "text", Text: message.Content})
			}
			content = parts
		}
		jinjaMessages = append(jinjaMessages, qwen38JinjaMessage{
			Role:             message.Role,
			Content:          content,
			ReasoningContent: message.Thinking,
			ToolCalls:        message.ToolCalls,
		})
	}

	messagesJSON, err := json.Marshal(jinjaMessages)
	if err != nil {
		t.Fatal(err)
	}
	toolsJSON, err := json.Marshal(tools)
	if err != nil {
		t.Fatal(err)
	}

	templatePath, err := filepath.Abs(qwen38Template)
	if err != nil {
		t.Fatal(err)
	}
	thinking := "unset"
	if think != nil && think.Value != nil {
		if think.Bool() {
			thinking = "true"
		} else {
			thinking = "false"
		}
	}

	script := `
import json
import sys
from pathlib import Path
from transformers.utils.chat_template_utils import _compile_jinja_template

template_path, messages_json, tools_json, thinking, effort = sys.argv[1:6]
tmpl = _compile_jinja_template(Path(template_path).read_text())
kwargs = {
    "messages": json.loads(messages_json),
    "tools": json.loads(tools_json),
    "add_generation_prompt": True,
}
if thinking == "true":
    kwargs["enable_thinking"] = True
elif thinking == "false":
    kwargs["enable_thinking"] = False
if effort:
    kwargs["reasoning_effort"] = effort
print(tmpl.render(**kwargs), end="")
`
	cmd := qwen38PythonCommand(t, "-c", script, templatePath, string(messagesJSON), string(toolsJSON), thinking, effort)
	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("Jinja render failed: %v\nstderr: %s", err, stderr.String())
	}
	return stdout.String()
}

func requireQwen38Jinja(t *testing.T) {
	t.Helper()
	if err := qwen38PythonCommand(t, "-c", "import jinja2, transformers").Run(); err != nil {
		t.Fatal("VERIFY_JINJA2=1 requires .venv/bin/python with transformers 5.x and jinja2, or uv with downloadable dependencies")
	}
}

func qwen38PythonCommand(t *testing.T, args ...string) *exec.Cmd {
	t.Helper()
	if python, ok := findQwen38VenvPython(); ok {
		return exec.CommandContext(t.Context(), python, args...)
	}

	uvArgs := append([]string{"run", "--with", "transformers>=5,<6", "--with", "jinja2>=3.1,<4", "python"}, args...)
	return exec.CommandContext(t.Context(), "uv", uvArgs...)
}

func findQwen38VenvPython() (string, bool) {
	dir, err := os.Getwd()
	if err != nil {
		return "", false
	}
	for {
		python := filepath.Join(dir, ".venv", "bin", "python")
		if _, err := os.Stat(python); err == nil {
			return python, true
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", false
		}
		dir = parent
	}
}
