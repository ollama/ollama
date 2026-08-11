package renderers

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
)

const nemotron3NanoTemplate = "testdata/nemotron3nano_chat_template.jinja2"

// nemotronReferenceCase is one reference conversation. The same inputs drive
// every Nemotron flavor's parity test; only the template and the frozen
// expectations differ.
type nemotronReferenceCase struct {
	name     string
	messages []api.Message
	tools    []api.Tool
	think    *api.ThinkValue
	expected string
}

func nemotronReferenceCases() []nemotronReferenceCase {
	return []nemotronReferenceCase{
		{
			name: "no system default thinking on",
			messages: []api.Message{
				{Role: "user", Content: "Hello"},
			},
			expected: "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n",
		},
		{
			name: "no system explicit thinking off",
			messages: []api.Message{
				{Role: "user", Content: "Hello"},
			},
			think:    thinkFalse(),
			expected: "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think></think>",
		},
		{
			name: "literal endthink does not enable thinking",
			messages: []api.Message{
				{Role: "user", Content: "literal </think> only"},
			},
			think:    thinkFalse(),
			expected: "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nliteral </think> only<|im_end|>\n<|im_start|>assistant\n<think></think>",
		},
		{
			name: "user no think toggle",
			messages: []api.Message{
				{Role: "user", Content: "Hello /no_think"},
			},
			think:    thinkTrue(),
			expected: "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think></think>",
		},
		{
			name: "system think toggle overrides false",
			messages: []api.Message{
				{Role: "system", Content: "Policy /think"},
				{Role: "user", Content: "Hello"},
			},
			think:    thinkFalse(),
			expected: "<|im_start|>system\nPolicy <|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n",
		},
		{
			name: "later toggle wins",
			messages: []api.Message{
				{Role: "system", Content: "Policy /no_think"},
				{Role: "user", Content: "Actually /think"},
			},
			think:    thinkFalse(),
			expected: "<|im_start|>system\nPolicy <|im_end|>\n<|im_start|>user\nActually<|im_end|>\n<|im_start|>assistant\n<think>\n",
		},
		{
			name: "system sanitizes toggles but preserves closing tag",
			messages: []api.Message{
				{Role: "system", Content: "A /think B /no_think C </think>"},
				{Role: "user", Content: "Hello"},
			},
			think:    thinkFalse(),
			expected: "<|im_start|>system\nA  B  C </think><|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n",
		},
		{
			name: "assistant plain content adds empty think block",
			messages: []api.Message{
				{Role: "user", Content: "Hi"},
				{Role: "assistant", Content: "Hello there"},
			},
			think:    thinkTrue(),
			expected: "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think></think>Hello there<|im_end|>\n<|im_start|>assistant\n<think>\n",
		},
		{
			name: "assistant reasoning content",
			messages: []api.Message{
				{Role: "user", Content: "Hi"},
				{Role: "assistant", Content: "Answer", Thinking: "Need to think"},
			},
			think:    thinkTrue(),
			expected: "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\nNeed to think\n</think>\nAnswer<|im_end|>\n<|im_start|>assistant\n<think>\n",
		},
		{
			name: "assistant preserves existing think tags",
			messages: []api.Message{
				{Role: "user", Content: "Hi"},
				{Role: "assistant", Content: "<think>kept</think>Answer"},
			},
			think:    thinkTrue(),
			expected: "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>kept</think>Answer<|im_end|>\n<|im_start|>assistant\n<think>\n",
		},
		{
			name: "tools without system",
			messages: []api.Message{
				{Role: "user", Content: "Use a tool"},
			},
			tools:    nemotron3NanoReferenceTools(),
			think:    thinkTrue(),
			expected: "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n<function>\n<name>search_docs</name>\n<description>Search docs</description>\n<parameters>\n<parameter>\n<name>query</name>\n<type>string</type>\n<description>Search query</description>\n<enum>[\"api\", \"cli\"]</enum>\n</parameter>\n<parameter>\n<name>mode</name>\n<type>['string', 'null']</type>\n<description>Mode</description>\n<anyOf>[{\"type\": \"string\"}, {\"type\": \"number\"}]</anyOf>\n</parameter>\n<parameter>\n<name>payload</name>\n<type>object</type>\n<description>Payload</description>\n<properties>{\"enabled\": {\"type\": \"boolean\"}}</properties>\n<required>[\"enabled\"]</required>\n</parameter>\n<parameter>\n<name>tags</name>\n<type>array</type>\n<description>Tags</description>\n<items>{\"type\": \"string\"}</items>\n</parameter>\n<$defs>{\"shared\": {\"type\": \"string\"}}</$defs>\n<required>[\"query\"]</required>\n</parameters>\n</function>\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT><|im_end|>\n<|im_start|>user\nUse a tool<|im_end|>\n<|im_start|>assistant\n<think>\n",
		},
		{
			name: "system with tools",
			messages: []api.Message{
				{Role: "system", Content: "Follow policy."},
				{Role: "user", Content: "Use a tool"},
			},
			tools:    nemotron3NanoReferenceTools(),
			think:    thinkTrue(),
			expected: "<|im_start|>system\nFollow policy.\n\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n<function>\n<name>search_docs</name>\n<description>Search docs</description>\n<parameters>\n<parameter>\n<name>query</name>\n<type>string</type>\n<description>Search query</description>\n<enum>[\"api\", \"cli\"]</enum>\n</parameter>\n<parameter>\n<name>mode</name>\n<type>['string', 'null']</type>\n<description>Mode</description>\n<anyOf>[{\"type\": \"string\"}, {\"type\": \"number\"}]</anyOf>\n</parameter>\n<parameter>\n<name>payload</name>\n<type>object</type>\n<description>Payload</description>\n<properties>{\"enabled\": {\"type\": \"boolean\"}}</properties>\n<required>[\"enabled\"]</required>\n</parameter>\n<parameter>\n<name>tags</name>\n<type>array</type>\n<description>Tags</description>\n<items>{\"type\": \"string\"}</items>\n</parameter>\n<$defs>{\"shared\": {\"type\": \"string\"}}</$defs>\n<required>[\"query\"]</required>\n</parameters>\n</function>\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT><|im_end|>\n<|im_start|>user\nUse a tool<|im_end|>\n<|im_start|>assistant\n<think>\n",
		},
		{
			name: "assistant tool call with content",
			messages: []api.Message{
				{Role: "user", Content: "Weather?"},
				{
					Role:    "assistant",
					Content: "Checking now.",
					ToolCalls: []api.ToolCall{{
						Function: api.ToolCallFunction{
							Name:      "get_weather",
							Arguments: testArgs(map[string]any{"city": "Paris"}),
						},
					}},
				},
			},
			think:    thinkTrue(),
			expected: "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nWeather?<|im_end|>\n<|im_start|>assistant\n<think></think>Checking now.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>\n<|im_end|>\n<|im_start|>assistant\n<think>\n",
		},
		{
			name: "assistant tool call with structured arguments",
			messages: []api.Message{
				{Role: "user", Content: "Create data"},
				{
					Role: "assistant",
					ToolCalls: []api.ToolCall{{
						Function: api.ToolCallFunction{
							Name: "create",
							Arguments: testArgsOrdered([]orderedArg{
								{Key: "payload", Value: map[string]any{"count": 42, "nested": map[string]any{"value": "ok"}}},
								{Key: "tags", Value: []any{"a", "b"}},
							}),
						},
					}},
				},
			},
			think:    thinkTrue(),
			expected: "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nCreate data<|im_end|>\n<|im_start|>assistant\n<think></think>\n<tool_call>\n<function=create>\n<parameter=payload>\n{\"count\": 42, \"nested\": {\"value\": \"ok\"}}\n</parameter>\n<parameter=tags>\n[\"a\", \"b\"]\n</parameter>\n</function>\n</tool_call>\n<|im_end|>\n<|im_start|>assistant\n<think>\n",
		},
		{
			name: "assistant tool call truncated with reasoning",
			messages: []api.Message{
				{Role: "user", Content: "Weather?"},
				{
					Role:     "assistant",
					Content:  "Checking now.",
					Thinking: "Need weather",
					ToolCalls: []api.ToolCall{{
						Function: api.ToolCallFunction{
							Name:      "get_weather",
							Arguments: testArgs(map[string]any{"city": "Paris"}),
						},
					}},
				},
				{Role: "user", Content: "And tomorrow?"},
			},
			think:    thinkTrue(),
			expected: "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nWeather?<|im_end|>\n<|im_start|>assistant\n<think></think>Checking now.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>\n<|im_end|>\n<|im_start|>user\nAnd tomorrow?<|im_end|>\n<|im_start|>assistant\n<think>\n",
		},
		{
			name: "assistant tool call truncated open think only",
			messages: []api.Message{
				{Role: "user", Content: "Weather?"},
				{
					Role:    "assistant",
					Content: "<think>draft",
					ToolCalls: []api.ToolCall{{
						Function: api.ToolCallFunction{
							Name:      "get_weather",
							Arguments: testArgs(map[string]any{"city": "Paris"}),
						},
					}},
				},
				{Role: "user", Content: "And tomorrow?"},
			},
			think:    thinkTrue(),
			expected: "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nWeather?<|im_end|>\n<|im_start|>assistant\n<think></think>\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>\n<|im_end|>\n<|im_start|>user\nAnd tomorrow?<|im_end|>\n<|im_start|>assistant\n<think>\n",
		},
		{
			name: "assistant tool call empty content",
			messages: []api.Message{
				{Role: "user", Content: "Weather?"},
				{
					Role:    "assistant",
					Content: "",
					ToolCalls: []api.ToolCall{{
						Function: api.ToolCallFunction{
							Name:      "get_weather",
							Arguments: testArgs(map[string]any{"city": "Paris"}),
						},
					}},
				},
			},
			think:    thinkTrue(),
			expected: "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nWeather?<|im_end|>\n<|im_start|>assistant\n<think></think>\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>\n<|im_end|>\n<|im_start|>assistant\n<think>\n",
		},
		{
			name: "assistant truncated with think pair",
			messages: []api.Message{
				{Role: "user", Content: "Hi"},
				{Role: "assistant", Content: "<think>hidden</think>Visible"},
				{Role: "user", Content: "Next"},
			},
			think:    thinkTrue(),
			expected: "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think></think>Visible<|im_end|>\n<|im_start|>user\nNext<|im_end|>\n<|im_start|>assistant\n<think>\n",
		},
		{
			name: "assistant truncated reasoning content",
			messages: []api.Message{
				{Role: "user", Content: "Hi"},
				{Role: "assistant", Thinking: "hidden", Content: "Visible"},
				{Role: "user", Content: "Next"},
			},
			think:    thinkTrue(),
			expected: "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think></think>\nVisible<|im_end|>\n<|im_start|>user\nNext<|im_end|>\n<|im_start|>assistant\n<think>\n",
		},
		{
			name: "assistant truncated plain content",
			messages: []api.Message{
				{Role: "user", Content: "Hi"},
				{Role: "assistant", Content: "Visible"},
				{Role: "user", Content: "Next"},
			},
			think:    thinkTrue(),
			expected: "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think></think>Visible<|im_end|>\n<|im_start|>user\nNext<|im_end|>\n<|im_start|>assistant\n<think>\n",
		},
		{
			name: "assistant truncated empty content",
			messages: []api.Message{
				{Role: "user", Content: "Hi"},
				{Role: "assistant", Content: ""},
				{Role: "user", Content: "Next"},
			},
			think:    thinkTrue(),
			expected: "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think></think><|im_end|>\n<|im_start|>user\nNext<|im_end|>\n<|im_start|>assistant\n<think>\n",
		},
		{
			name: "consecutive tool messages grouped",
			messages: []api.Message{
				{Role: "user", Content: "Do work"},
				{
					Role: "assistant",
					ToolCalls: []api.ToolCall{{
						Function: api.ToolCallFunction{
							Name:      "step",
							Arguments: testArgs(map[string]any{"value": 1}),
						},
					}},
				},
				{Role: "tool", Content: "one"},
				{Role: "tool", Content: "two"},
			},
			think:    thinkTrue(),
			expected: "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nDo work<|im_end|>\n<|im_start|>assistant\n<think></think>\n<tool_call>\n<function=step>\n<parameter=value>\n1\n</parameter>\n</function>\n</tool_call>\n<|im_end|>\n<|im_start|>user\n<tool_response>\none\n</tool_response>\n<tool_response>\ntwo\n</tool_response>\n<|im_end|>\n<|im_start|>assistant\n<think>\n",
		},
		{
			name: "fallback role",
			messages: []api.Message{
				{Role: "developer", Content: "Custom role content"},
			},
			think:    thinkTrue(),
			expected: "<|im_start|>system\n<|im_end|>\n<|im_start|>developer\nCustom role content<|im_end|>\n<|im_start|>assistant\n<think>\n",
		},
	}
}

func TestNemotron3NanoRendererMatchesReference(t *testing.T) {
	tests := nemotronReferenceCases()

	verifyJinja2 := os.Getenv("VERIFY_JINJA2") != ""
	if verifyJinja2 {
		if _, err := os.Stat(filepath.Join(nemotron3NanoRepoRoot(t), ".venv", "bin", "python3")); err != nil {
			t.Fatal("VERIFY_JINJA2=1 requires .venv/bin/python3")
		}
	}

	renderer := &Nemotron3NanoRenderer{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := renderer.Render(tt.messages, tt.tools, tt.think)
			if err != nil {
				t.Fatalf("Render() error = %v", err)
			}

			if diff := cmp.Diff(tt.expected, got); diff != "" {
				t.Fatalf("renderer mismatch (-want +got):\n%s", diff)
			}

			if verifyJinja2 {
				jinja2Output := renderNemotron3NanoWithJinja2(t, tt.messages, tt.tools, tt.think)
				if diff := cmp.Diff(tt.expected, jinja2Output); diff != "" {
					t.Fatalf("reference template mismatch (-want +got):\n%s", diff)
				}
			}
		})
	}
}

func nemotron3NanoReferenceTools() []api.Tool {
	return []api.Tool{{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "search_docs",
			Description: "Search docs",
			Parameters: api.ToolFunctionParameters{
				Type:     "object",
				Defs:     map[string]any{"shared": map[string]any{"type": "string"}},
				Required: []string{"query"},
				Properties: testPropsOrdered([]orderedProp{
					{
						Key: "query",
						Value: api.ToolProperty{
							Type:        api.PropertyType{"string"},
							Description: "Search query",
							Enum:        []any{"api", "cli"},
						},
					},
					{
						Key: "mode",
						Value: api.ToolProperty{
							Type:        api.PropertyType{"string", "null"},
							Description: "Mode",
							AnyOf: []api.ToolProperty{
								{Type: api.PropertyType{"string"}},
								{Type: api.PropertyType{"number"}},
							},
						},
					},
					{
						Key: "payload",
						Value: api.ToolProperty{
							Type:        api.PropertyType{"object"},
							Description: "Payload",
							Properties:  testPropsOrdered([]orderedProp{{Key: "enabled", Value: api.ToolProperty{Type: api.PropertyType{"boolean"}}}}),
							Required:    []string{"enabled"},
						},
					},
					{
						Key: "tags",
						Value: api.ToolProperty{
							Type:        api.PropertyType{"array"},
							Description: "Tags",
							Items:       map[string]any{"type": "string"},
						},
					},
				}),
			},
		},
	}}
}

func renderNemotron3NanoWithJinja2(t *testing.T, messages []api.Message, tools []api.Tool, think *api.ThinkValue) string {
	return renderNemotronWithJinja2(t, nemotron3NanoTemplate, messages, tools, think)
}

// renderNemotronWithJinja2 renders through the real jinja engine using the
// named testdata template, so each Nemotron flavor is checked against the
// template its checkpoint ships.
func renderNemotronWithJinja2(t *testing.T, templateRel string, messages []api.Message, tools []api.Tool, think *api.ThinkValue) string {
	t.Helper()

	type jinja2ToolCall struct {
		ID       string `json:"id,omitempty"`
		Function struct {
			Name      string `json:"name"`
			Arguments any    `json:"arguments"`
		} `json:"function"`
	}
	type jinja2Message struct {
		Role             string           `json:"role"`
		Content          string           `json:"content"`
		ReasoningContent string           `json:"reasoning_content,omitempty"`
		ToolCalls        []jinja2ToolCall `json:"tool_calls,omitempty"`
		Name             string           `json:"name,omitempty"`
		ToolCallID       string           `json:"tool_call_id,omitempty"`
	}

	var jMsgs []jinja2Message
	for _, m := range messages {
		jm := jinja2Message{
			Role:             m.Role,
			Content:          m.Content,
			ReasoningContent: m.Thinking,
			Name:             m.ToolName,
			ToolCallID:       m.ToolCallID,
		}
		for _, tc := range m.ToolCalls {
			jtc := jinja2ToolCall{ID: tc.ID}
			jtc.Function.Name = tc.Function.Name
			// Pass the arguments through as raw JSON. Decoding into a Go map
			// would drop their order, and re-marshaling would re-emit them
			// sorted -- the template renders one <parameter=...> per key in
			// order, so that silently changes the reference.
			raw, err := tc.Function.Arguments.MarshalJSON()
			if err != nil {
				t.Fatalf("failed to marshal tool args: %v", err)
			}
			jtc.Function.Arguments = json.RawMessage(raw)
			jm.ToolCalls = append(jm.ToolCalls, jtc)
		}
		jMsgs = append(jMsgs, jm)
	}

	msgsJSON, err := json.Marshal(jMsgs)
	if err != nil {
		t.Fatalf("failed to marshal messages: %v", err)
	}

	toolsJSON := "None"
	if len(tools) > 0 {
		b, err := json.Marshal(tools)
		if err != nil {
			t.Fatalf("failed to marshal tools: %v", err)
		}
		toolsJSON = string(b)
	}

	thinking := "unset"
	if think != nil {
		if think.Bool() {
			thinking = "true"
		} else {
			thinking = "false"
		}
	}

	repoRoot := nemotron3NanoRepoRoot(t)
	templatePath := filepath.Join(repoRoot, "model", "renderers", templateRel)
	pythonPath := filepath.Join(repoRoot, ".venv", "bin", "python3")
	script := `
import json
import sys
from pathlib import Path
from transformers.utils.chat_template_utils import _compile_jinja_template

template_path, messages_json, tools_json, thinking = sys.argv[1:5]
tmpl = _compile_jinja_template(Path(template_path).read_text())
kwargs = {
    "messages": json.loads(messages_json),
    "add_generation_prompt": True,
}
if tools_json != "None":
    kwargs["tools"] = json.loads(tools_json)
if thinking == "true":
    kwargs["enable_thinking"] = True
elif thinking == "false":
    kwargs["enable_thinking"] = False
print(tmpl.render(**kwargs), end="")
`

	cmd := exec.CommandContext(t.Context(), pythonPath, "-c", script, templatePath, string(msgsJSON), toolsJSON, thinking)
	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("python render failed: %v\nstderr: %s", err, stderr.String())
	}
	return stdout.String()
}

func nemotron3NanoRepoRoot(t *testing.T) string {
	t.Helper()
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("failed to locate test file")
	}
	return filepath.Dir(filepath.Dir(filepath.Dir(filename)))
}
