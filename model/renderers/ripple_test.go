package renderers

import (
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/model/parsers"
)

const rippleTemplate = "testdata/ripple_chat_template.jinja"

func TestRipplePublisherTemplate(t *testing.T) {
	verifyLFM3TemplateFixture(t, rippleTemplate, "877c2dbf365243dda14738ca250b31398059b35ffd91d4ed0ec68be45ca6ebd3")
}

func TestRippleRendererTools(t *testing.T) {
	call := api.ToolCall{Function: api.ToolCallFunction{Name: "f\"<&é", Arguments: args(`{"z":"line\n\"quote\"\\<&é","a":[true,null,3]}`)}}
	tests := []struct {
		name     string
		messages []api.Message
		want     string
	}{
		{
			"call alone",
			[]api.Message{{Role: "assistant", ToolCalls: []api.ToolCall{call}}},
			"<|im_start|>assistant\n<tool_call>\n" + `{"name": "f\"<&é", "arguments": {"z": "line\n\"quote\"\\<&é", "a": [true, null, 3]}}` + "\n</tool_call><|im_end|>\n<|im_start|>assistant\n",
		},
		{
			"multiple calls with reasoning",
			[]api.Message{{Role: "assistant", Thinking: "R", Content: " C \n", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "one"}}, {Function: api.ToolCallFunction{Name: "two"}}}}},
			"<|im_start|>assistant\n<think>\nR\n</think>\n C \n\n<tool_call>\n{\"name\": \"one\", \"arguments\": {}}\n</tool_call>\n<tool_call>\n{\"name\": \"two\", \"arguments\": {}}\n</tool_call><|im_end|>\n<|im_start|>assistant\n",
		},
		{
			"grouped results",
			[]api.Message{{Role: "tool", Content: " A\n"}, {Role: "tool", Content: "B"}, {Role: "user", Content: "Next"}, {Role: "tool", Content: "C"}},
			"<|im_start|>user\n<tool_response>\n A\n\n</tool_response>\n<tool_response>\nB\n</tool_response><|im_end|>\n<|im_start|>user\nNext<|im_end|>\n<|im_start|>user\n<tool_response>\nC\n</tool_response><|im_end|>\n<|im_start|>assistant\n",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := RenderWithRenderer("ripple", tt.messages, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Fatalf("got %q\nwant %q", got, tt.want)
			}
			if os.Getenv("VERIFY_JINJA2") != "" {
				want := renderQwenJinja(t, rippleTemplate, tt.messages, nil, nil, "")
				if got != want {
					t.Fatalf("Go %q\nPython %q", got, want)
				}
			}
		})
	}
}

func TestRippleRendererToolInstructions(t *testing.T) {
	tool := api.Tool{Type: "function", Function: api.ToolFunction{Name: "execute_bash", Parameters: api.ToolFunctionParameters{Type: "object"}}}
	for _, system := range []string{"", "Be precise."} {
		messages := []api.Message{{Role: "system", Content: system}, {Role: "user", Content: "pwd"}}
		got, err := RenderWithRenderer("ripple", messages, []api.Tool{tool}, nil)
		if err != nil {
			t.Fatal(err)
		}
		want := "<|im_start|>system\n" + system + "\n\n" + `# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools> XML tags:
<tools>
{"type": "function", "function": {"name": "execute_bash", "parameters": {"type": "object", "properties": null}}}
</tools>

For each function call, return exactly one valid JSON object with function name and arguments inside XML tool-call tags. The arguments field must be a JSON object, not a string, and strings must use valid JSON escaping:
<tool_call>
{"name": "execute_bash", "arguments": {"command": "pwd"}}
</tool_call><|im_end|>
<|im_start|>user
pwd<|im_end|>
<|im_start|>assistant
`
		if got != want {
			t.Fatalf("got %q\nwant %q", got, want)
		}
		if os.Getenv("VERIFY_JINJA2") != "" {
			want := renderQwenJinja(t, rippleTemplate, messages, []api.Tool{tool}, nil, "")
			if got != want {
				t.Fatalf("Go %q\nPython %q", got, want)
			}
		}
	}
}

func TestRippleRendererInvalidInput(t *testing.T) {
	for _, tt := range []struct {
		name      string
		messages  []api.Message
		tools     []api.Tool
		errorText string
	}{
		{name: "empty transcript", errorText: "at least one message"},
		{name: "empty transcript with tools", tools: []api.Tool{{Type: "function"}}, errorText: "at least one message"},
		{name: "image", messages: []api.Message{{Role: "user", Content: "describe", Images: []api.ImageData{{1}}}}, errorText: "text-only"},
		{name: "invalid tool schema", messages: []api.Message{{Role: "user"}}, tools: []api.Tool{{Items: make(chan int)}}, errorText: "unsupported type"},
		{name: "invalid tool arguments", messages: []api.Message{{Role: "assistant", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "f", Arguments: testArgs(map[string]any{"x": make(chan int)})}}}}}, errorText: "unsupported type"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := RenderWithRenderer("ripple", tt.messages, tt.tools, nil)
			if got != "" || err == nil || !strings.Contains(err.Error(), tt.errorText) {
				t.Fatalf("got (%q, %v), want %s", got, err, tt.errorText)
			}
		})
	}
}

func TestRippleRendererToolResultContinuation(t *testing.T) {
	messages := []api.Message{
		{Role: "user", Content: "Run the program and report what it prints."},
		{Role: "assistant", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "execute_bash", Arguments: args(`{"command":"go run ./cmd/demo"}`)}}}},
		{Role: "tool", Content: `{"exit_code":0,"stdout":"fmt.Println(\"done\")","stderr":""}`},
		{Role: "assistant", Thinking: "The command succeeded; I can report its output.", Content: "The program prints `done`."},
	}
	tools := []api.Tool{{Type: "function", Function: api.ToolFunction{Name: "execute_bash"}}}
	want := "<|im_start|>system\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"execute_bash\", \"parameters\": {\"type\": \"\", \"properties\": null}}}\n</tools>\n\nFor each function call, return exactly one valid JSON object with function name and arguments inside XML tool-call tags. The arguments field must be a JSON object, not a string, and strings must use valid JSON escaping:\n<tool_call>\n{\"name\": \"execute_bash\", \"arguments\": {\"command\": \"pwd\"}}\n</tool_call><|im_end|>\n<|im_start|>user\nRun the program and report what it prints.<|im_end|>\n<|im_start|>assistant\n<tool_call>\n{\"name\": \"execute_bash\", \"arguments\": {\"command\": \"go run ./cmd/demo\"}}\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n{\"exit_code\":0,\"stdout\":\"fmt.Println(\\\"done\\\")\",\"stderr\":\"\"}\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\nThe command succeeded; I can report its output.\n</think>\nThe program prints `done`.<|im_end|>\n<|im_start|>assistant\n"
	got, err := (&RippleRenderer{}).Render(messages, tools, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("render mismatch:\ngot:\n%s\nwant:\n%s", got, want)
	}
}

func TestRippleRendererJinjaParity(t *testing.T) {
	if os.Getenv("VERIFY_JINJA2") == "" {
		t.Skip("set VERIFY_JINJA2=1 and JINJA_PYTHON for publisher template parity")
	}
	values := api.NewToolCallFunctionArguments()
	values.Set("z", json.Number("3.0"))
	values.Set("a", json.RawMessage(`{"z": 1e-7, "a": ["\u2028\u2029<&é", true, null]}`))
	call := api.ToolCall{Function: api.ToolCallFunction{Name: "f\"\\\n<&é", Arguments: values}}
	tool := api.Tool{Type: "function", Function: api.ToolFunction{Name: "f", Description: "Use \"quoted\" <&é\n", Parameters: api.ToolFunctionParameters{Type: "object", Properties: propsMap(`{"z":{"type":"integer"},"a":{"type":"string"}}`), Required: []string{"z"}}}}
	for _, tt := range []struct {
		name     string
		messages []api.Message
		tools    []api.Tool
		think    *api.ThinkValue
	}{
		{name: "plain", messages: []api.Message{{Role: "user", Content: " \nHi\t "}}},
		{name: "empty user", messages: []api.Message{{Role: "user"}}},
		{name: "initial and later systems", messages: []api.Message{{Role: "system", Content: "A"}, {Role: "system", Content: "B"}, {Role: "user", Content: "C"}, {Role: "system", Content: "D"}}},
		{name: "tools without system", messages: []api.Message{{Role: "user", Content: "Call"}}, tools: []api.Tool{tool, tool}},
		{name: "old reasoning retained", messages: []api.Message{{Role: "assistant", Thinking: "old", Content: "first"}, {Role: "user", Content: "next"}, {Role: "assistant", Thinking: "new", Content: "second"}, {Role: "user", Content: "again"}}},
		{name: "no duplicate wrapper", messages: []api.Message{{Role: "assistant", Thinking: "ignored", Content: "\x1c\t\u2003<think>old</think>answer"}}},
		{name: "nonleading tag", messages: []api.Message{{Role: "assistant", Thinking: "new", Content: "words<think>literal</think>"}}},
		{name: "thinking false has no invented prefix", messages: []api.Message{{Role: "assistant", Thinking: "retain", Content: "answer"}}, think: &api.ThinkValue{Value: false}},
		{name: "thinking true no forced opening", messages: []api.Message{{Role: "user", Content: "Hi"}}, think: &api.ThinkValue{Value: true}},
		{name: "tools JSON numbers escaping and order", messages: []api.Message{{Role: "assistant", Thinking: "Reason", Content: " \n ", ToolCalls: []api.ToolCall{call}}, {Role: "tool", Content: " result \n"}, {Role: "tool", Content: "other"}}, tools: []api.Tool{tool}},
		{name: "tools reasoning only", messages: []api.Message{{Role: "assistant", Thinking: "Reason", ToolCalls: []api.ToolCall{call}}}},
		{name: "group boundary ignores unknown role", messages: []api.Message{{Role: "tool", Content: "one"}, {Role: "ignored"}, {Role: "tool", Content: "two"}}},
		{
			name:  "tool result followed by assistant decision",
			tools: []api.Tool{{Type: "function", Function: api.ToolFunction{Name: "execute_bash"}}},
			messages: []api.Message{
				{Role: "user", Content: "Run the program and report what it prints."},
				{Role: "assistant", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "execute_bash", Arguments: args(`{"command":"go run ./cmd/demo"}`)}}}},
				{Role: "tool", Content: `{"exit_code":0,"stdout":"fmt.Println(\"done\")","stderr":""}`},
				{Role: "assistant", Thinking: "The command succeeded; I can report its output.", Content: "The program prints `done`."},
			},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := RenderWithRenderer("ripple", tt.messages, tt.tools, tt.think)
			if err != nil {
				t.Fatal(err)
			}
			want := renderQwenJinja(t, rippleTemplate, tt.messages, tt.tools, tt.think, "")
			if got != want {
				t.Fatalf("Go %q\nPython %q", got, want)
			}
		})
	}
	python := os.Getenv("JINJA_PYTHON")
	if python == "" {
		t.Fatal("set JINJA_PYTHON for negative publisher-template checks")
	}
	for _, messages := range []any{[]any{}, []any{map[string]any{"role": "user", "content": []any{map[string]any{"type": "image"}}}}} {
		output, err := lfm3RunJinja2(t, python, rippleTemplate, map[string]any{"messages": messages, "add_generation_prompt": true})
		if err == nil {
			t.Fatalf("publisher accepted unsupported transcript: %q", output)
		}
	}
}

func TestRippleRendererParserRoundTrip(t *testing.T) {
	for _, content := range []string{"", " ", "\n", "\n\n", " \t\n ", "answer\n", "日本語\n"} {
		for _, thinking := range []string{"", "R", " \n", "\n\nR\t\n"} {
			for _, calls := range [][]api.ToolCall{nil, {{Function: api.ToolCallFunction{Name: "f", Arguments: args(`{"text":"<&é\n\"quoted\""}`)}}}, {{Function: api.ToolCallFunction{Name: "f"}}, {Function: api.ToolCallFunction{Name: "g"}}}} {
				message := api.Message{Role: "assistant", Content: content, Thinking: thinking, ToolCalls: calls}
				original, err := RenderWithRenderer("ripple", []api.Message{message}, nil, nil)
				if err != nil {
					t.Fatal(err)
				}
				body := strings.TrimSuffix(strings.TrimPrefix(original, "<|im_start|>assistant\n"), "<|im_end|>\n<|im_start|>assistant\n")
				p := parsers.ParserForName("ripple")
				p.Init(nil, nil, nil)
				parsed := api.Message{Role: "assistant"}
				for i := 0; i <= len(body); i++ {
					chunk := ""
					if i < len(body) {
						chunk = body[i : i+1]
					}
					c, r, tc, err := p.Add(chunk, i == len(body))
					if err != nil {
						t.Fatal(err)
					}
					parsed.Content += c
					parsed.Thinking += r
					parsed.ToolCalls = append(parsed.ToolCalls, tc...)
				}
				got, err := RenderWithRenderer("ripple", []api.Message{parsed}, nil, nil)
				if err != nil {
					t.Fatal(err)
				}
				if got != original {
					t.Fatalf("roundtrip got %q\nwant %q", got, original)
				}
			}
		}
	}
}

func TestRippleRendererReasoningHistory(t *testing.T) {
	for _, tt := range []struct{ name, content, reasoning, want string }{
		{"separate", " Answer \n", " Reason \n", "<think>\n Reason \n\n</think>\n Answer \n"},
		{"already tagged", "<think>\noriginal\n</think>\nAnswer", "ignored", "<think>\noriginal\n</think>\nAnswer"},
		{"leading whitespace already tagged", "\x1c\t\u2003<think>original</think>Answer", "ignored", "\x1c\t\u2003<think>original</think>Answer"},
		{"nonleading tag", "Text <think>literal</think>", "Reason", "<think>\nReason\n</think>\nText <think>literal</think>"},
		{"empty reasoning", "Answer", "", "Answer"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			for _, think := range []*api.ThinkValue{nil, {Value: true}, {Value: false}, {Value: "high"}} {
				got, err := RenderWithRenderer("ripple", []api.Message{{Role: "assistant", Content: tt.content, Thinking: tt.reasoning}, {Role: "user", Content: "Next"}}, nil, think)
				if err != nil {
					t.Fatal(err)
				}
				want := "<|im_start|>assistant\n" + tt.want + "<|im_end|>\n<|im_start|>user\nNext<|im_end|>\n<|im_start|>assistant\n"
				if got != want {
					t.Fatalf("got %q\nwant %q", got, want)
				}
			}
		})
	}
}

func TestRippleRendererMessages(t *testing.T) {
	for _, tt := range []struct {
		name     string
		messages []api.Message
		want     string
	}{
		{"plain", []api.Message{{Role: "user", Content: " Hi\n"}}, "<|im_start|>user\n Hi\n<|im_end|>\n<|im_start|>assistant\n"},
		{"system", []api.Message{{Role: "system", Content: "Be precise."}, {Role: "user", Content: "Hi"}}, "<|im_start|>system\nBe precise.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"},
		{"empty system", []api.Message{{Role: "system"}}, "<|im_start|>system\n<|im_end|>\n<|im_start|>assistant\n"},
		{"later system", []api.Message{{Role: "user", Content: "Hi"}, {Role: "system", Content: "Now"}}, "<|im_start|>user\nHi<|im_end|>\n<|im_start|>system\nNow<|im_end|>\n<|im_start|>assistant\n"},
		{"assistant is closed not prefilled", []api.Message{{Role: "assistant", Content: "Part"}}, "<|im_start|>assistant\nPart<|im_end|>\n<|im_start|>assistant\n"},
		{"unknown role ignored", []api.Message{{Role: "developer", Content: "ignored"}}, "<|im_start|>assistant\n"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := RenderWithRenderer("ripple", tt.messages, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Fatalf("got %q\nwant %q", got, tt.want)
			}
		})
	}
	if rendererForName("ripple") == nil {
		t.Fatal("missing ripple renderer")
	}
	if got := LeadingBOSForRenderer("ripple"); got != "" {
		t.Fatalf("BOS = %q", got)
	}
	if got := ThinkingForRenderer("ripple"); got != nil {
		t.Fatalf("unexpected thinking control: %+v", got)
	}
}
