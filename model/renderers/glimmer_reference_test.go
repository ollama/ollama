package renderers

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/model/parsers"
)

// glimmerChatTemplate is copied byte-for-byte from hf/chat_template.jinja at
// publisher revision a4e59da52a7bc87ae7251dd5545c0dd437c44b68.
// SHA-256: cfc67e5f349f37690dfd31ed1f18bc4442a9dd32fe39a648f993cb4eb3cae678.
const glimmerChatTemplate = "testdata/glimmer_chat_template.jinja"

const (
	glimmerRefBOS         = "<|begin_of_text|>"
	glimmerRefStart       = "<|start|>"
	glimmerRefMessage     = "<|message|>"
	glimmerRefEOM         = "<|eom|>"
	glimmerRefEOT         = "<|eot|>"
	glimmerRefCurrentDate = "2026-07-29"
)

const glimmerRefDefaultSystem = `You are a helpful AI assistant.
Knowledge cutoff: 2026-01-04.
Current date: 2026-07-29.

Reasoning strength: high.

# Valid recipients: "self", "user".`

const glimmerRefWeatherToolDefinitions = `In this environment you have access to a set of tools you can use to answer the user's question.

You can invoke a function by writing a "<atem:function_calls>" block like the following:
<atem:function_calls>
<atem:invoke name="$FUNCTION_NAME">
<atem:parameter name="$PARAMETER_NAME">$PARAMETER_VALUE</atem:parameter>
...
</atem:invoke>
</atem:function_calls>

String and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular expressions.
Here are the functions available in JSONSchema format:
// Tool metadata
{"name": "get_weather", "description": ""}
// Function schemas
{"name": "get_weather", "description": "Get the weather", "parameters": {"type": "object", "required": ["city"], "properties": {"city": {"type": "string"}, "units": {"type": "string"}}}}

Here's an example of how to call a function in the tool set:
(If the tool namespace is not specified, invoke the function directly as ` + "`example_function_name`" + ` rather than ` + "`example_tool_name.example_function_name`" + `)

to=example_tool_name.example_function_name

<atem:function_calls>
<atem:invoke name="example_tool_name.example_function_name">
<atem:parameter name="example_parameter_1">value_1</atem:parameter>
<atem:parameter name="example_parameter_2">This is the value for the second parameter
that can span
"multiple" lines
</atem:parameter>
</atem:invoke>
</atem:function_calls>`

type glimmerJinjaMessage struct {
	Role             string         `json:"role"`
	Content          any            `json:"content"`
	Name             string         `json:"name,omitempty"`
	ToolCallID       string         `json:"tool_call_id,omitempty"`
	ReasoningContent string         `json:"reasoning_content,omitempty"`
	ToolCalls        []api.ToolCall `json:"tool_calls,omitempty"`
	Recipient        string         `json:"recipient,omitempty"`
	EndTurn          *bool          `json:"end_turn,omitempty"`
}

type glimmerJinjaContentPart struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

func glimmerRefPrompt(parts ...string) string {
	return glimmerRefBOS + strings.Join(parts, "") + glimmerRefStart + "assistant"
}

func glimmerRefMsg(header, content, end string) string {
	return glimmerRefStart + header + glimmerRefMessage + content + end
}

func glimmerReferenceWeatherTool() api.Tool {
	return api.Tool{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "get_weather",
			Description: "Get the weather",
			Parameters: api.ToolFunctionParameters{
				Type:     "object",
				Required: []string{"city"},
				Properties: testPropsOrdered([]orderedProp{
					{Key: "city", Value: api.ToolProperty{Type: api.PropertyType{"string"}}},
					{Key: "units", Value: api.ToolProperty{Type: api.PropertyType{"string"}}},
				}),
			},
		},
	}
}

func TestGlimmerRendererMatchesJinja2Reference(t *testing.T) {
	verifyJinja2 := os.Getenv("VERIFY_JINJA2") != ""
	if verifyJinja2 {
		requireGlimmerJinja2(t)
		t.Log("VERIFY_JINJA2=1: verifying expected values against Glimmer Jinja2 template")
	}

	weatherTool := glimmerReferenceWeatherTool()
	weatherArgs := api.NewToolCallFunctionArguments()
	weatherArgs.Set("city", "SF")
	weatherArgs.Set("units", "fahrenheit")

	tests := []struct {
		name     string
		messages []api.Message
		tools    []api.Tool
		think    *api.ThinkValue
		expected string
	}{
		{
			name:     "default system",
			messages: []api.Message{{Role: "user", Content: "Hello"}},
			expected: glimmerRefPrompt(
				glimmerRefMsg("system", glimmerRefDefaultSystem, glimmerRefEOT),
				glimmerRefMsg("user", "Hello", glimmerRefEOT),
			),
		},
		{
			name: "explicit system image thinking and answer",
			messages: []api.Message{
				{Role: "system", Content: "Be concise."},
				{Role: "user", Content: "Read this.", Images: []api.ImageData{{1}}},
				{Role: "assistant", Thinking: "I should inspect it.", Content: "It says hello."},
			},
			expected: glimmerRefPrompt(
				glimmerRefMsg("system", `Be concise.

Reasoning strength: high.

# Valid recipients: "self", "user".`, glimmerRefEOT),
				glimmerRefMsg("user", "<|patch|>Read this.", glimmerRefEOT),
				glimmerRefMsg("assistant to=self", "I should inspect it.", glimmerRefEOM),
				glimmerRefMsg("assistant to=user", "It says hello.", glimmerRefEOT),
			),
		},
		{
			name: "explicit system reasoning effort is normalized and not duplicated",
			messages: []api.Message{
				{Role: "system", Content: "You are direct.\nReasoning effort: medium."},
				{Role: "user", Content: "Hello"},
			},
			expected: glimmerRefPrompt(
				glimmerRefMsg("system", `You are direct.
Reasoning strength: medium.

# Valid recipients: "self", "user".`, glimmerRefEOT),
				glimmerRefMsg("user", "Hello", glimmerRefEOT),
			),
		},
		{
			name: "explicit system reasoning strength is not duplicated",
			messages: []api.Message{
				{Role: "system", Content: "You are direct.\nReasoning strength: low."},
				{Role: "user", Content: "Hello"},
			},
			think: &api.ThinkValue{Value: "high"},
			expected: glimmerRefPrompt(
				glimmerRefMsg("system", `You are direct.
Reasoning strength: low.

# Valid recipients: "self", "user".`, glimmerRefEOT),
				glimmerRefMsg("user", "Hello", glimmerRefEOT),
			),
		},
		{
			name: "tool call and result",
			messages: []api.Message{
				{Role: "user", Content: "Weather?"},
				{Role: "assistant", Thinking: "Need current data.", ToolCalls: []api.ToolCall{{
					ID:       "call_weather",
					Function: api.ToolCallFunction{Name: "get_weather", Arguments: weatherArgs},
				}}},
				{Role: "tool", ToolCallID: "call_weather", Content: `{"temp":65}`},
				{Role: "assistant", Content: "It is 65F."},
			},
			tools: []api.Tool{weatherTool},
			expected: glimmerRefPrompt(
				glimmerRefMsg("system", `You are a helpful AI assistant.
Knowledge cutoff: 2026-01-04.
Current date: 2026-07-29.

Reasoning strength: high.

`+glimmerRefWeatherToolDefinitions+`

# Valid recipients: "self", "get_weather.*", "user".`, glimmerRefEOT),
				glimmerRefMsg("user", "Weather?", glimmerRefEOT),
				glimmerRefMsg("assistant to=self", "Need current data.", glimmerRefEOM),
				glimmerRefMsg("assistant to=get_weather", `<atem:function_calls>
<atem:invoke name="get_weather">
<atem:parameter name="city">SF</atem:parameter>
<atem:parameter name="units">fahrenheit</atem:parameter>
</atem:invoke>
</atem:function_calls>`, glimmerRefEOT),
				glimmerRefMsg("tool get_weather", `<tool_output name="get_weather">
{"temp":65}
</tool_output>`, glimmerRefEOT),
				glimmerRefMsg("assistant to=user", "It is 65F.", glimmerRefEOT),
			),
		},
		{
			name: "adjacent assistant tool calls use message boundary",
			messages: []api.Message{
				{Role: "user", Content: "Run twice."},
				{Role: "assistant", ToolCalls: []api.ToolCall{{
					Function: api.ToolCallFunction{Name: "get_weather", Arguments: weatherArgs},
				}}},
				{Role: "assistant", ToolCalls: []api.ToolCall{{
					Function: api.ToolCallFunction{Name: "get_weather", Arguments: weatherArgs},
				}}},
				{Role: "user", Content: "Continue."},
			},
			expected: glimmerRefPrompt(
				glimmerRefMsg("system", glimmerRefDefaultSystem, glimmerRefEOT),
				glimmerRefMsg("user", "Run twice.", glimmerRefEOT),
				glimmerRefMsg("assistant to=get_weather", `<atem:function_calls>
<atem:invoke name="get_weather">
<atem:parameter name="city">SF</atem:parameter>
<atem:parameter name="units">fahrenheit</atem:parameter>
</atem:invoke>
</atem:function_calls>`, glimmerRefEOM),
				glimmerRefMsg("assistant to=get_weather", `<atem:function_calls>
<atem:invoke name="get_weather">
<atem:parameter name="city">SF</atem:parameter>
<atem:parameter name="units">fahrenheit</atem:parameter>
</atem:invoke>
</atem:function_calls>`, glimmerRefEOT),
				glimmerRefMsg("user", "Continue.", glimmerRefEOT),
			),
		},
		{
			name:     "explicit reasoning level",
			messages: []api.Message{{Role: "user", Content: "Solve."}},
			think:    &api.ThinkValue{Value: "low"},
			expected: glimmerRefPrompt(
				glimmerRefMsg("system", strings.Replace(glimmerRefDefaultSystem, "high", "low", 1), glimmerRefEOT),
				glimmerRefMsg("user", "Solve.", glimmerRefEOT),
			),
		},
		{
			name:     "thinking disabled",
			messages: []api.Message{{Role: "user", Content: "Answer directly."}},
			think:    &api.ThinkValue{Value: false},
			expected: glimmerRefPrompt(
				glimmerRefMsg("system", strings.Replace(glimmerRefDefaultSystem, "high", "none", 1), glimmerRefEOT),
				glimmerRefMsg("user", "Answer directly.", glimmerRefEOT),
			),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := (&GlimmerRenderer{currentDate: glimmerRefCurrentDate}).Render(tt.messages, tt.tools, tt.think)
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.expected {
				t.Fatalf("renderer output mismatch:\ngot:  %q\nwant: %q", got, tt.expected)
			}

			if verifyJinja2 {
				jinja2Output := renderGlimmerWithJinja2(t, tt.messages, tt.tools, tt.think)
				if jinja2Output != tt.expected {
					fmt.Fprintf(os.Stderr, "\nJINJA2 OUTPUT for %s:\n%q\n\n", tt.name, jinja2Output)
					t.Fatalf("hardcoded expected value doesn't match Jinja2 template:\ngot:  %q\nwant: %q", tt.expected, jinja2Output)
				}
			}
		})
	}
}

func TestGlimmerRendererMatchesJinja2ExpandedParity(t *testing.T) {
	if os.Getenv("VERIFY_JINJA2") == "" {
		t.Skip("set VERIFY_JINJA2=1 to run expanded Jinja2 parity checks")
	}
	requireGlimmerJinja2(t)

	weatherTool := glimmerReferenceWeatherTool()
	readTool := api.Tool{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "filesystem.read",
			Description: `Read a file's contents`,
			Parameters: api.ToolFunctionParameters{
				Type:     "object",
				Required: []string{"path"},
				Properties: testPropsOrdered([]orderedProp{
					{Key: "path", Value: api.ToolProperty{Type: api.PropertyType{"string"}}},
					{Key: "lines", Value: api.ToolProperty{Type: api.PropertyType{"array"}, Items: map[string]any{"type": "integer"}}},
				}),
			},
		},
	}

	weatherArgs := api.NewToolCallFunctionArguments()
	weatherArgs.Set("city", "SF")
	readArgs := api.NewToolCallFunctionArguments()
	readArgs.Set("path", "README.md")
	readArgs.Set("lines", []any{1, 5})
	allValues := api.NewToolCallFunctionArguments()
	allValues.Set("string", " keep spaces ")
	allValues.Set("integer", 3)
	allValues.Set("number", 3.5)
	allValues.Set("boolean", true)
	allValues.Set("null", nil)
	allValues.Set("array", []any{"héllo", 2})
	allValues.Set("object", map[string]any{"b": 2, "a": "<value>"})

	tests := []struct {
		name     string
		messages []api.Message
		tools    []api.Tool
		think    *api.ThinkValue
	}{
		{
			name:     "minimal user",
			messages: []api.Message{{Role: "user", Content: "Hello"}},
		},
		{
			name: "explicit system multi turn",
			messages: []api.Message{
				{Role: "system", Content: "Be concise."},
				{Role: "user", Content: "One"},
				{Role: "assistant", Content: "Two"},
				{Role: "user", Content: "Three"},
			},
		},
		{
			name: "late and repeated system messages",
			messages: []api.Message{
				{Role: "user", Content: "One"},
				{Role: "system", Content: "First policy."},
				{Role: "system", Content: "Second policy."},
				{Role: "user", Content: "Two"},
			},
			tools: []api.Tool{weatherTool},
		},
		{
			name: "multiple images",
			messages: []api.Message{{
				Role:    "user",
				Content: "Compare them.",
				Images:  []api.ImageData{{1}, {2}},
			}},
		},
		{
			name: "unsupported role omitted",
			messages: []api.Message{
				{Role: "developer", ToolName: "policy", Content: "Follow policy."},
				{Role: "user", Content: "Hi"},
			},
		},
		{
			name: "assistant thinking and final",
			messages: []api.Message{
				{Role: "user", Content: "Solve"},
				{Role: "assistant", Thinking: "Private reasoning.", Content: "Answer."},
			},
		},
		{
			name: "multiple tool calls and out of order results",
			messages: []api.Message{
				{Role: "user", Content: "Weather and file?"},
				{Role: "assistant", Thinking: "Need both.", ToolCalls: []api.ToolCall{
					{
						ID:       "call_weather",
						Function: api.ToolCallFunction{Name: "get_weather", Arguments: weatherArgs},
					},
					{
						ID:       "call_read",
						Function: api.ToolCallFunction{Name: "filesystem.read", Arguments: readArgs},
					},
				}},
				{Role: "tool", ToolCallID: "call_read", Content: "README"},
				{Role: "tool", ToolCallID: "call_weather", Content: `{"temp":65}`},
				{Role: "assistant", Content: "Done."},
			},
			tools: []api.Tool{weatherTool, readTool},
		},
		{
			name: "explicit tool name",
			messages: []api.Message{
				{Role: "tool", ToolName: "filesystem.read", Content: "README"},
			},
			tools: []api.Tool{readTool},
		},
		{
			name: "unresolved tool call id",
			messages: []api.Message{
				{Role: "tool", ToolCallID: "missing_call", Content: "missing"},
			},
		},
		{
			name: "empty tool result name",
			messages: []api.Message{
				{Role: "tool", Content: "orphan"},
			},
		},
		{
			name: "all ATEM value types",
			messages: []api.Message{
				{Role: "user", Content: "Run."},
				{Role: "assistant", ToolCalls: []api.ToolCall{{
					Function: api.ToolCallFunction{Name: "filesystem.read", Arguments: allValues},
				}}},
			},
			tools: []api.Tool{readTool},
		},
		{
			name:     "thinking true",
			messages: []api.Message{{Role: "user", Content: "Solve."}},
			think:    &api.ThinkValue{Value: true},
		},
		{
			name:     "thinking false",
			messages: []api.Message{{Role: "user", Content: "Answer."}},
			think:    &api.ThinkValue{Value: false},
		},
		{
			name:     "reasoning max",
			messages: []api.Message{{Role: "user", Content: "Solve."}},
			think:    &api.ThinkValue{Value: "max"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := (&GlimmerRenderer{currentDate: glimmerRefCurrentDate}).Render(tt.messages, tt.tools, tt.think)
			if err != nil {
				t.Fatal(err)
			}
			want := renderGlimmerWithJinja2(t, tt.messages, tt.tools, tt.think)
			if got != want {
				t.Fatalf("renderer output doesn't match Jinja2 template:\ngot:  %q\nwant: %q", got, want)
			}
		})
	}
}

func TestGlimmerRendererKnownJinja2DifferenceThinkingContinuation(t *testing.T) {
	messages := []api.Message{
		{Role: "user", Content: "Solve"},
		{Role: "assistant", Thinking: "Private reasoning."},
	}
	expected := glimmerRefPrompt(
		glimmerRefMsg("system", glimmerRefDefaultSystem, glimmerRefEOT),
		glimmerRefMsg("user", "Solve", glimmerRefEOT),
		glimmerRefMsg("assistant to=self", "Private reasoning.", glimmerRefEOM),
	)

	got, err := (&GlimmerRenderer{currentDate: glimmerRefCurrentDate}).Render(messages, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got != expected {
		t.Fatalf("renderer continuation mismatch:\ngot:  %q\nwant: %q", got, expected)
	}

	if os.Getenv("VERIFY_JINJA2") != "" {
		requireGlimmerJinja2(t)

		// api.Message cannot directly represent the reference template's
		// unfinished assistant message. Mechanical conversion therefore renders
		// this history as completed reasoning followed by an empty user answer.
		completeExpected := glimmerRefPrompt(
			glimmerRefMsg("system", glimmerRefDefaultSystem, glimmerRefEOT),
			glimmerRefMsg("user", "Solve", glimmerRefEOT),
			glimmerRefMsg("assistant to=self", "Private reasoning.", glimmerRefEOM),
			glimmerRefMsg("assistant to=user", "", glimmerRefEOT),
		)
		completeOutput := renderGlimmerWithJinja2(t, messages, nil, nil)
		if completeOutput != completeExpected {
			t.Fatalf("mechanical Jinja2 conversion mismatch:\ngot:  %q\nwant: %q", completeOutput, completeExpected)
		}
		if completeOutput == got {
			t.Fatal("known unfinished-message representation difference disappeared")
		}

		// The renderer's partial output must still match the publisher template
		// when given the reference representation of an unfinished self message.
		endTurn := false
		partialOutput := renderGlimmerJinjaMessages(t, []glimmerJinjaMessage{
			{Role: "user", Content: "Solve"},
			{
				Role:      "assistant",
				Content:   "Private reasoning.",
				Recipient: "self",
				EndTurn:   &endTurn,
			},
		}, nil, nil)
		if partialOutput != expected {
			t.Fatalf("hardcoded continuation doesn't match Jinja2 template:\ngot:  %q\nwant: %q", expected, partialOutput)
		}
	}
}

func TestGlimmerParserRendererToolLoopRoundTrip(t *testing.T) {
	verifyJinja2 := os.Getenv("VERIFY_JINJA2") != ""
	if verifyJinja2 {
		requireGlimmerJinja2(t)
	}

	issueTool := api.Tool{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "issue",
			Description: "Read an issue and its recent comments.",
			Parameters: api.ToolFunctionParameters{
				Type:     "object",
				Required: []string{"number"},
				Properties: testPropsOrdered([]orderedProp{{
					Key:   "number",
					Value: api.ToolProperty{Type: api.PropertyType{"integer"}},
				}}),
			},
		},
	}
	readTool := api.Tool{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "read",
			Description: "Read a repository file.",
			Parameters: api.ToolFunctionParameters{
				Type:     "object",
				Required: []string{"path"},
				Properties: testPropsOrdered([]orderedProp{{
					Key:   "path",
					Value: api.ToolProperty{Type: api.PropertyType{"string"}},
				}}),
			},
		},
	}
	tools := []api.Tool{issueTool, readTool}
	messages := []api.Message{{
		Role:    "user",
		Content: "Investigate issue 1736 and inspect server/download.go.",
	}}

	parse := func(raw string) api.Message {
		t.Helper()
		parser := parsers.ParserForName("glimmer")
		parser.Init(tools, nil, nil)
		content, thinking, calls, err := parser.Add(raw, true)
		if err != nil {
			t.Fatalf("parse Glimmer assistant turn: %v", err)
		}
		return api.Message{
			Role:      "assistant",
			Content:   content,
			Thinking:  thinking,
			ToolCalls: calls,
		}
	}
	verifyPrompt := func(stage string) {
		t.Helper()
		got, err := (&GlimmerRenderer{currentDate: glimmerRefCurrentDate}).Render(messages, tools, nil)
		if err != nil {
			t.Fatalf("%s render: %v", stage, err)
		}
		if !strings.HasSuffix(got, glimmerRefStart+"assistant") {
			t.Fatalf("%s prompt does not end with the assistant generation prefix", stage)
		}
		if verifyJinja2 {
			want := renderGlimmerWithJinja2(t, messages, tools, nil)
			if got != want {
				t.Fatalf("%s prompt doesn't match Jinja2:\ngot:  %q\nwant: %q", stage, got, want)
			}
		}
	}

	verifyPrompt("initial")

	issueCall := parse(` to=self<|message|>I need the issue details.<|eom|>` +
		`<|start|>assistant to=issue<|message|><atem:function_calls>
<atem:invoke name="issue">
<atem:parameter name="number">1736</atem:parameter>
</atem:invoke>
</atem:function_calls><|eot|>`)
	if issueCall.Thinking != "I need the issue details." || len(issueCall.ToolCalls) != 1 {
		t.Fatalf("issue turn = %#v", issueCall)
	}
	if number, ok := issueCall.ToolCalls[0].Function.Arguments.Get("number"); !ok || number != 1736 {
		t.Fatalf("issue number = %#v, %v; want 1736", number, ok)
	}
	issueCall.ToolCalls[0].ID = "call_issue"
	messages = append(messages, issueCall, api.Message{
		Role:       "tool",
		ToolCallID: "call_issue",
		Content:    "Issue 1736 links the proposed first-byte fix.",
	})
	verifyPrompt("after issue")

	readCall := parse(` to=self<|message|>Now inspect the implementation.<|eom|>` +
		`<|start|>assistant to=read<|message|><atem:function_calls>
<atem:invoke name="read">
<atem:parameter name="path">server/download.go</atem:parameter>
</atem:invoke>
</atem:function_calls><|eot|>`)
	if readCall.Thinking != "Now inspect the implementation." || len(readCall.ToolCalls) != 1 {
		t.Fatalf("read turn = %#v", readCall)
	}
	if path, ok := readCall.ToolCalls[0].Function.Arguments.Get("path"); !ok || path != "server/download.go" {
		t.Fatalf("read path = %#v, %v; want server/download.go", path, ok)
	}
	readCall.ToolCalls[0].ID = "call_read"
	messages = append(messages, readCall, api.Message{
		Role:       "tool",
		ToolCallID: "call_read",
		Content:    "The inactivity clock starts only after the first body byte.",
	})
	verifyPrompt("after read")

	answer := parse(` to=self<|message|>I have enough evidence.<|eom|>` +
		`<|start|>assistant to=user<|message|>The proposed fix is plausible.<|eot|>`)
	if answer.Thinking != "I have enough evidence." || answer.Content != "The proposed fix is plausible." || len(answer.ToolCalls) != 0 {
		t.Fatalf("answer turn = %#v", answer)
	}
	messages = append(messages, answer)
	verifyPrompt("after answer")
}

func renderGlimmerWithJinja2(t *testing.T, messages []api.Message, tools []api.Tool, think *api.ThinkValue) string {
	t.Helper()

	var strength any
	if think != nil {
		switch {
		case !think.Bool():
			strength = "none"
		case think.IsString():
			strength = think.String()
		default:
			strength = "high"
		}
	}
	return renderGlimmerJinjaMessages(t, glimmerMessagesForJinja(messages), tools, strength)
}

func renderGlimmerJinjaMessages(t *testing.T, messages []glimmerJinjaMessage, tools []api.Tool, strength any) string {
	t.Helper()

	templatePath, err := filepath.Abs(glimmerChatTemplate)
	if err != nil {
		t.Fatalf("failed to get template path: %v", err)
	}

	messagesJSON, err := json.Marshal(messages)
	if err != nil {
		t.Fatalf("failed to marshal messages: %v", err)
	}
	toolsJSON, err := json.Marshal(tools)
	if err != nil {
		t.Fatalf("failed to marshal tools: %v", err)
	}

	strengthJSON, err := json.Marshal(strength)
	if err != nil {
		t.Fatalf("failed to marshal reasoning strength: %v", err)
	}

	script := fmt.Sprintf(`
import json
from transformers.utils.chat_template_utils import _compile_jinja_template
tmpl = _compile_jinja_template(open(%q).read())
kwargs = {
    "messages": json.loads(%q),
    "tools": json.loads(%q),
    "bos_token": %q,
    "add_generation_prompt": True,
    "current_date": %q,
}
reasoning_strength = json.loads(%q)
if reasoning_strength is not None:
    kwargs["reasoning_strength"] = reasoning_strength
print(tmpl.render(**kwargs), end="")
`, templatePath, string(messagesJSON), string(toolsJSON), glimmerRefBOS, glimmerRefCurrentDate, string(strengthJSON))

	cmd := glimmerPythonCommand(t, "-c", script)
	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("Jinja2 render failed: %v\nstderr: %s", err, stderr.String())
	}
	return stdout.String()
}

func requireGlimmerJinja2(t *testing.T) {
	t.Helper()
	if err := glimmerPythonCommand(t, "-c", "import transformers").Run(); err != nil {
		t.Fatal("VERIFY_JINJA2=1 requires .venv/bin/python with transformers 5.x or uv with downloadable transformers")
	}
}

func glimmerPythonCommand(t *testing.T, args ...string) *exec.Cmd {
	t.Helper()
	if python, ok := findGlimmerVenvPython(); ok {
		return exec.CommandContext(t.Context(), python, args...)
	}

	uvArgs := append([]string{"run", "--with", "transformers>=5,<6", "python"}, args...)
	return exec.CommandContext(t.Context(), "uv", uvArgs...)
}

func findGlimmerVenvPython() (string, bool) {
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

func glimmerMessagesForJinja(messages []api.Message) []glimmerJinjaMessage {
	result := make([]glimmerJinjaMessage, 0, len(messages))
	for _, message := range messages {
		result = append(result, glimmerJinjaMessage{
			Role:             message.Role,
			Content:          glimmerJinjaContent(message.Content, len(message.Images)),
			Name:             message.ToolName,
			ToolCallID:       message.ToolCallID,
			ReasoningContent: message.Thinking,
			ToolCalls:        message.ToolCalls,
		})
	}
	return result
}

func glimmerJinjaContent(content string, imageCount int) any {
	if imageCount == 0 {
		return content
	}

	parts := make([]glimmerJinjaContentPart, 0, imageCount+1)
	for range imageCount {
		parts = append(parts, glimmerJinjaContentPart{Type: "image"})
	}
	if content != "" {
		parts = append(parts, glimmerJinjaContentPart{Type: "text", Text: content})
	}
	return parts
}
