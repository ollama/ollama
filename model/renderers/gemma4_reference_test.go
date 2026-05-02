package renderers

// TestGemma4RendererMatchesReference verifies our renderer matches the HF
// Jinja2 chat template exactly.
//
// To regenerate expected values, save gemma4Jinja2Template (below) to
// gemma4_chat_template.jinja2 and run:
//
//   python3 -c "
//   from jinja2 import Environment; import json
//   tmpl = Environment().from_string(open('gemma4_chat_template.jinja2').read())
//   msgs = [{'role':'user','content':'Hello'}]
//   print(repr(tmpl.render(messages=msgs, bos_token='<bos>', add_generation_prompt=True)))
//   "

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/stretchr/testify/assert"
)

// The full Jinja2 template is committed as testdata/gemma4_chat_template.jinja2.
// Run with VERIFY_JINJA2=1 to verify expected values against the template using Python.

func bashRefTool() []api.Tool {
	return []api.Tool{{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "bash",
			Description: "Run a command",
			Parameters: api.ToolFunctionParameters{
				Type:     "object",
				Required: []string{"command"},
				Properties: testPropsMap(map[string]api.ToolProperty{
					"command": {Type: api.PropertyType{"string"}, Description: "The command"},
				}),
			},
		},
	}}
}

func bashAndReadRefTools() []api.Tool {
	return []api.Tool{
		bashRefTool()[0],
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "read",
				Description: "Read a file",
				Parameters: api.ToolFunctionParameters{
					Type:     "object",
					Required: []string{"path"},
					Properties: testPropsMap(map[string]api.ToolProperty{
						"path": {Type: api.PropertyType{"string"}, Description: "File path"},
					}),
				},
			},
		},
	}
}

func weatherTool() []api.Tool {
	return []api.Tool{{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "get_weather",
			Description: "Get weather",
			Parameters: api.ToolFunctionParameters{
				Type: "object",
				Properties: testPropsMap(map[string]api.ToolProperty{
					"city": {Type: api.PropertyType{"string"}, Description: "City"},
				}),
			},
		},
	}}
}

func addTool() []api.Tool {
	return []api.Tool{{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "add",
			Description: "Add numbers",
			Parameters: api.ToolFunctionParameters{
				Type: "object",
				Properties: testPropsMap(map[string]api.ToolProperty{
					"a": {Type: api.PropertyType{"number"}},
					"b": {Type: api.PropertyType{"number"}},
				}),
			},
		},
	}}
}

func flagTool() []api.Tool {
	return []api.Tool{{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "set_flag",
			Description: "Set a flag",
			Parameters: api.ToolFunctionParameters{
				Type: "object",
				Properties: testPropsMap(map[string]api.ToolProperty{
					"enabled": {Type: api.PropertyType{"boolean"}, Description: "Flag value"},
				}),
			},
		},
	}}
}

func modeTool() []api.Tool {
	return []api.Tool{{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "set_mode",
			Description: "Set mode",
			Parameters: api.ToolFunctionParameters{
				Type: "object",
				Properties: testPropsMap(map[string]api.ToolProperty{
					"mode": {Type: api.PropertyType{"string"}, Description: "The mode", Enum: []any{"fast", "slow"}},
				}),
			},
		},
	}}
}

func bashSmallTool() []api.Tool {
	return []api.Tool{{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "bash",
			Description: "Run",
			Parameters: api.ToolFunctionParameters{
				Type:     "object",
				Required: []string{"command"},
				Properties: testPropsMap(map[string]api.ToolProperty{
					"command": {Type: api.PropertyType{"string"}, Description: "Cmd"},
				}),
			},
		},
	}}
}

func nestedTool() []api.Tool {
	return []api.Tool{{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "create",
			Description: "Create item",
			Parameters: api.ToolFunctionParameters{
				Type: "object",
				Properties: testPropsMap(map[string]api.ToolProperty{
					"name": {Type: api.PropertyType{"string"}, Description: "Name"},
					"config": {Type: api.PropertyType{"object"}, Description: "Config", Properties: testPropsMap(map[string]api.ToolProperty{
						"enabled": {Type: api.PropertyType{"boolean"}, Description: "On/off"},
					})},
				}),
			},
		},
	}}
}

func arrayTool() []api.Tool {
	return []api.Tool{{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "batch",
			Description: "Run batch",
			Parameters: api.ToolFunctionParameters{
				Type: "object",
				Properties: testPropsMap(map[string]api.ToolProperty{
					"commands": {Type: api.PropertyType{"array"}, Description: "Commands", Items: map[string]any{"type": "string"}},
				}),
			},
		},
	}}
}

func configureTool() []api.Tool {
	return []api.Tool{{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "configure",
			Description: "Configure",
			Parameters: api.ToolFunctionParameters{
				Type: "object",
				Properties: testPropsMap(map[string]api.ToolProperty{
					"config": {Type: api.PropertyType{"object"}, Description: "Config"},
				}),
			},
		},
	}}
}

func batchArrayTool() []api.Tool {
	return []api.Tool{{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "batch",
			Description: "Run batch",
			Parameters: api.ToolFunctionParameters{
				Type: "object",
				Properties: testPropsMap(map[string]api.ToolProperty{
					"ids": {Type: api.PropertyType{"array"}, Description: "IDs"},
				}),
			},
		},
	}}
}

func countTool() []api.Tool {
	return []api.Tool{{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "count",
			Description: "Count items",
			Parameters: api.ToolFunctionParameters{
				Type: "object",
				Properties: testPropsMap(map[string]api.ToolProperty{
					"n": {Type: api.PropertyType{"number"}},
				}),
			},
		},
	}}
}

func enumNoDescTool() []api.Tool {
	return []api.Tool{{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "set_level",
			Description: "Set level",
			Parameters: api.ToolFunctionParameters{
				Type: "object",
				Properties: testPropsMap(map[string]api.ToolProperty{
					"level": {Type: api.PropertyType{"string"}, Enum: []any{"low", "high"}},
				}),
			},
		},
	}}
}

func nestedRequiredTool() []api.Tool {
	return []api.Tool{{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "create_user",
			Description: "Create user",
			Parameters: api.ToolFunctionParameters{
				Type: "object",
				Properties: testPropsMap(map[string]api.ToolProperty{
					"profile": {
						Type: api.PropertyType{"object"}, Description: "Profile",
						Required: []string{"name"},
						Properties: testPropsMap(map[string]api.ToolProperty{
							"name": {Type: api.PropertyType{"string"}, Description: "Name"},
							"age":  {Type: api.PropertyType{"number"}, Description: "Age"},
						}),
					},
				}),
			},
		},
	}}
}

func calcTool() []api.Tool {
	return []api.Tool{{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "calc",
			Description: "Calculate",
			Parameters: api.ToolFunctionParameters{
				Type: "object",
				Properties: testPropsMap(map[string]api.ToolProperty{
					"value": {Type: api.PropertyType{"number"}, Description: "Value"},
				}),
			},
		},
	}}
}

func rawTool() []api.Tool {
	return []api.Tool{{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "raw",
			Description: "Raw input",
			Parameters: api.ToolFunctionParameters{
				Type: "object",
			},
		},
	}}
}

func moveTool() []api.Tool {
	return []api.Tool{{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "move",
			Description: "Move",
			Parameters: api.ToolFunctionParameters{
				Type:     "object",
				Required: []string{"x", "y"},
				Properties: testPropsMap(map[string]api.ToolProperty{
					"x": {Type: api.PropertyType{"number"}, Description: "X"},
					"y": {Type: api.PropertyType{"number"}, Description: "Y"},
				}),
			},
		},
	}}
}

func arrayNoItemsTool() []api.Tool {
	return []api.Tool{{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "tag",
			Description: "Tag items",
			Parameters: api.ToolFunctionParameters{
				Type: "object",
				Properties: testPropsMap(map[string]api.ToolProperty{
					"tags": {Type: api.PropertyType{"array"}, Description: "Tags"},
				}),
			},
		},
	}}
}

func objectNoDescTool() []api.Tool {
	return []api.Tool{{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "update",
			Description: "Update settings",
			Parameters: api.ToolFunctionParameters{
				Type: "object",
				Properties: testPropsMap(map[string]api.ToolProperty{
					"settings": {Type: api.PropertyType{"object"}, Properties: testPropsMap(map[string]api.ToolProperty{
						"verbose": {Type: api.PropertyType{"boolean"}, Description: "Verbose mode"},
					})},
				}),
			},
		},
	}}
}

func searchTool() []api.Tool {
	return []api.Tool{{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "search",
			Description: "Search",
			Parameters: api.ToolFunctionParameters{
				Type: "object",
				Properties: testPropsMap(map[string]api.ToolProperty{
					"query":  {Type: api.PropertyType{"string"}, Description: "Search query"},
					"limit":  {Type: api.PropertyType{"number"}},
					"offset": {Type: api.PropertyType{"number"}, Description: "Start offset"},
				}),
			},
		},
	}}
}

var (
	bashSmallDeclRef      = `<|tool>declaration:bash{description:<|"|>Run<|"|>,parameters:{properties:{command:{description:<|"|>Cmd<|"|>,type:<|"|>STRING<|"|>}},required:[<|"|>command<|"|>],type:<|"|>OBJECT<|"|>}}<tool|>`
	nestedDeclRef         = `<|tool>declaration:create{description:<|"|>Create item<|"|>,parameters:{properties:{config:{description:<|"|>Config<|"|>,properties:{enabled:{description:<|"|>On/off<|"|>,type:<|"|>BOOLEAN<|"|>}},type:<|"|>OBJECT<|"|>},name:{description:<|"|>Name<|"|>,type:<|"|>STRING<|"|>}},type:<|"|>OBJECT<|"|>}}<tool|>`
	arrayDeclRef          = `<|tool>declaration:batch{description:<|"|>Run batch<|"|>,parameters:{properties:{commands:{description:<|"|>Commands<|"|>,items:{type:<|"|>STRING<|"|>},type:<|"|>ARRAY<|"|>}},type:<|"|>OBJECT<|"|>}}<tool|>`
	bashDeclRef           = `<|tool>declaration:bash{description:<|"|>Run a command<|"|>,parameters:{properties:{command:{description:<|"|>The command<|"|>,type:<|"|>STRING<|"|>}},required:[<|"|>command<|"|>],type:<|"|>OBJECT<|"|>}}<tool|>`
	readDeclRef           = `<|tool>declaration:read{description:<|"|>Read a file<|"|>,parameters:{properties:{path:{description:<|"|>File path<|"|>,type:<|"|>STRING<|"|>}},required:[<|"|>path<|"|>],type:<|"|>OBJECT<|"|>}}<tool|>`
	weatherDeclRef        = `<|tool>declaration:get_weather{description:<|"|>Get weather<|"|>,parameters:{properties:{city:{description:<|"|>City<|"|>,type:<|"|>STRING<|"|>}},type:<|"|>OBJECT<|"|>}}<tool|>`
	addDeclRef            = `<|tool>declaration:add{description:<|"|>Add numbers<|"|>,parameters:{properties:{a:{type:<|"|>NUMBER<|"|>},b:{type:<|"|>NUMBER<|"|>}},type:<|"|>OBJECT<|"|>}}<tool|>`
	flagDeclRef           = `<|tool>declaration:set_flag{description:<|"|>Set a flag<|"|>,parameters:{properties:{enabled:{description:<|"|>Flag value<|"|>,type:<|"|>BOOLEAN<|"|>}},type:<|"|>OBJECT<|"|>}}<tool|>`
	modeDeclRef           = `<|tool>declaration:set_mode{description:<|"|>Set mode<|"|>,parameters:{properties:{mode:{description:<|"|>The mode<|"|>,enum:[<|"|>fast<|"|>,<|"|>slow<|"|>],type:<|"|>STRING<|"|>}},type:<|"|>OBJECT<|"|>}}<tool|>`
	configureDeclRef      = `<|tool>declaration:configure{description:<|"|>Configure<|"|>,parameters:{properties:{config:{description:<|"|>Config<|"|>,properties:{},type:<|"|>OBJECT<|"|>}},type:<|"|>OBJECT<|"|>}}<tool|>`
	batchArrayDeclRef     = `<|tool>declaration:batch{description:<|"|>Run batch<|"|>,parameters:{properties:{ids:{description:<|"|>IDs<|"|>,type:<|"|>ARRAY<|"|>}},type:<|"|>OBJECT<|"|>}}<tool|>`
	countDeclRef          = `<|tool>declaration:count{description:<|"|>Count items<|"|>,parameters:{properties:{n:{type:<|"|>NUMBER<|"|>}},type:<|"|>OBJECT<|"|>}}<tool|>`
	enumNoDescDeclRef     = `<|tool>declaration:set_level{description:<|"|>Set level<|"|>,parameters:{properties:{level:{enum:[<|"|>low<|"|>,<|"|>high<|"|>],type:<|"|>STRING<|"|>}},type:<|"|>OBJECT<|"|>}}<tool|>`
	searchDeclRef         = `<|tool>declaration:search{description:<|"|>Search<|"|>,parameters:{properties:{limit:{type:<|"|>NUMBER<|"|>},offset:{description:<|"|>Start offset<|"|>,type:<|"|>NUMBER<|"|>},query:{description:<|"|>Search query<|"|>,type:<|"|>STRING<|"|>}},type:<|"|>OBJECT<|"|>}}<tool|>`
	arrayNoItemsDeclRef   = `<|tool>declaration:tag{description:<|"|>Tag items<|"|>,parameters:{properties:{tags:{description:<|"|>Tags<|"|>,type:<|"|>ARRAY<|"|>}},type:<|"|>OBJECT<|"|>}}<tool|>`
	objectNoDescDeclRef   = `<|tool>declaration:update{description:<|"|>Update settings<|"|>,parameters:{properties:{settings:{,properties:{verbose:{description:<|"|>Verbose mode<|"|>,type:<|"|>BOOLEAN<|"|>}}type:<|"|>OBJECT<|"|>}},type:<|"|>OBJECT<|"|>}}<tool|>`
	nestedRequiredDeclRef = `<|tool>declaration:create_user{description:<|"|>Create user<|"|>,parameters:{properties:{profile:{description:<|"|>Profile<|"|>,properties:{age:{description:<|"|>Age<|"|>,type:<|"|>NUMBER<|"|>},name:{description:<|"|>Name<|"|>,type:<|"|>STRING<|"|>}},required:[<|"|>name<|"|>],type:<|"|>OBJECT<|"|>}},type:<|"|>OBJECT<|"|>}}<tool|>`
	calcDeclRef           = `<|tool>declaration:calc{description:<|"|>Calculate<|"|>,parameters:{properties:{value:{description:<|"|>Value<|"|>,type:<|"|>NUMBER<|"|>}},type:<|"|>OBJECT<|"|>}}<tool|>`
	rawDeclRef            = `<|tool>declaration:raw{description:<|"|>Raw input<|"|>,parameters:{type:<|"|>OBJECT<|"|>}}<tool|>`
	moveDeclRef           = `<|tool>declaration:move{description:<|"|>Move<|"|>,parameters:{properties:{x:{description:<|"|>X<|"|>,type:<|"|>NUMBER<|"|>},y:{description:<|"|>Y<|"|>,type:<|"|>NUMBER<|"|>}},required:[<|"|>x<|"|>,<|"|>y<|"|>],type:<|"|>OBJECT<|"|>}}<tool|>`
)

func TestGemma4RendererMatchesReference(t *testing.T) {
	q := `<|"|>`

	tests := []struct {
		name     string
		messages []api.Message
		tools    []api.Tool
		think    *api.ThinkValue
		expected string
	}{
		// === Header block paths ===
		{
			name:     "user_only",
			messages: []api.Message{{Role: "user", Content: "Hello"}},
			expected: "<bos><|turn>user\nHello<turn|>\n<|turn>model\n",
		},
		{
			name: "system_user",
			messages: []api.Message{
				{Role: "system", Content: "You are helpful."},
				{Role: "user", Content: "Hi"},
			},
			expected: "<bos><|turn>system\nYou are helpful.<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n",
		},
		{
			name: "developer_user",
			messages: []api.Message{
				{Role: "developer", Content: "You are helpful."},
				{Role: "user", Content: "Hi"},
			},
			expected: "<bos><|turn>system\nYou are helpful.<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n",
		},
		{
			name:     "tools_no_system",
			messages: []api.Message{{Role: "user", Content: "Hi"}},
			tools:    bashRefTool(),
			expected: "<bos><|turn>system\n" + bashDeclRef + "<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n",
		},
		{
			name: "system_tools",
			messages: []api.Message{
				{Role: "system", Content: "You are helpful."},
				{Role: "user", Content: "Hi"},
			},
			tools:    bashRefTool(),
			expected: "<bos><|turn>system\nYou are helpful." + bashDeclRef + "<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n",
		},
		{
			name:     "thinking_no_system",
			messages: []api.Message{{Role: "user", Content: "Hi"}},
			think:    thinkTrue(),
			expected: "<bos><|turn>system\n<|think|><turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n",
		},
		{
			name: "thinking_system",
			messages: []api.Message{
				{Role: "system", Content: "You are helpful."},
				{Role: "user", Content: "Hi"},
			},
			think:    thinkTrue(),
			expected: "<bos><|turn>system\n<|think|>You are helpful.<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n",
		},
		{
			name:     "thinking_tools",
			messages: []api.Message{{Role: "user", Content: "Hi"}},
			tools:    bashRefTool(),
			think:    thinkTrue(),
			expected: "<bos><|turn>system\n<|think|>" + bashDeclRef + "<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n",
		},
		{
			name: "thinking_system_tools",
			messages: []api.Message{
				{Role: "system", Content: "You are helpful."},
				{Role: "user", Content: "Hi"},
			},
			tools:    bashRefTool(),
			think:    thinkTrue(),
			expected: "<bos><|turn>system\n<|think|>You are helpful." + bashDeclRef + "<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n",
		},

		// === Message loop paths ===
		{
			name: "multi_turn",
			messages: []api.Message{
				{Role: "system", Content: "You are helpful."},
				{Role: "user", Content: "Hi"},
				{Role: "assistant", Content: "Hello!"},
				{Role: "user", Content: "More"},
			},
			expected: "<bos><|turn>system\nYou are helpful.<turn|>\n" +
				"<|turn>user\nHi<turn|>\n" +
				"<|turn>model\nHello!<turn|>\n" +
				"<|turn>user\nMore<turn|>\n" +
				"<|turn>model\n",
		},
		{
			// Tool call with structured args → tool response as separate <|turn>tool turn
			name: "tool_call_response",
			messages: []api.Message{
				{Role: "system", Content: "You are helpful."},
				{Role: "user", Content: "List files"},
				{Role: "assistant", ToolCalls: []api.ToolCall{{
					Function: api.ToolCallFunction{
						Name:      "bash",
						Arguments: testArgs(map[string]any{"command": "ls"}),
					},
				}}},
				{Role: "tool", Content: "file1.txt\nfile2.txt"},
			},
			tools: bashRefTool(),
			expected: "<bos><|turn>system\nYou are helpful." + bashDeclRef + "<turn|>\n" +
				"<|turn>user\nList files<turn|>\n" +
				"<|turn>model\n<|tool_call>call:bash{command:" + q + "ls" + q + "}<tool_call|><turn|>\n" +
				"<|turn>tool\nfile1.txt\nfile2.txt<turn|>\n" +
				"<|turn>model\n",
		},
		{
			// Full round trip: call → response → assistant reply → user follow-up
			name: "full_round_trip",
			messages: []api.Message{
				{Role: "system", Content: "You are helpful."},
				{Role: "user", Content: "List files"},
				{Role: "assistant", ToolCalls: []api.ToolCall{{
					Function: api.ToolCallFunction{
						Name:      "bash",
						Arguments: testArgs(map[string]any{"command": "ls"}),
					},
				}}},
				{Role: "tool", Content: "file1.txt\nfile2.txt"},
				{Role: "assistant", Content: "Here are the files."},
				{Role: "user", Content: "Read file1.txt"},
			},
			tools: bashAndReadRefTools(),
			expected: "<bos><|turn>system\nYou are helpful." + bashDeclRef + readDeclRef + "<turn|>\n" +
				"<|turn>user\nList files<turn|>\n" +
				"<|turn>model\n<|tool_call>call:bash{command:" + q + "ls" + q + "}<tool_call|><turn|>\n" +
				"<|turn>tool\nfile1.txt\nfile2.txt<turn|>\n" +
				"<|turn>model\nHere are the files.<turn|>\n" +
				"<|turn>user\nRead file1.txt<turn|>\n" +
				"<|turn>model\n",
		},
		{
			// Multiple tool calls + multiple tool responses
			name: "multiple_tool_calls",
			messages: []api.Message{
				{Role: "system", Content: "You are helpful."},
				{Role: "user", Content: "List and read"},
				{Role: "assistant", ToolCalls: []api.ToolCall{
					{Function: api.ToolCallFunction{Name: "bash", Arguments: testArgs(map[string]any{"command": "ls"})}},
					{Function: api.ToolCallFunction{Name: "read", Arguments: testArgs(map[string]any{"path": "go.mod"})}},
				}},
				{Role: "tool", Content: "file1.txt\nfile2.txt"},
				{Role: "tool", Content: "module example.com/foo"},
			},
			tools: bashAndReadRefTools(),
			expected: "<bos><|turn>system\nYou are helpful." + bashDeclRef + readDeclRef + "<turn|>\n" +
				"<|turn>user\nList and read<turn|>\n" +
				"<|turn>model\n<|tool_call>call:bash{command:" + q + "ls" + q + "}<tool_call|>" +
				"<|tool_call>call:read{path:" + q + "go.mod" + q + "}<tool_call|><turn|>\n" +
				"<|turn>tool\nfile1.txt\nfile2.txt<turn|>\n" +
				"<|turn>tool\nmodule example.com/foo<turn|>\n" +
				"<|turn>model\n",
		},
		{
			// Thinking content in assistant history should be stripped
			name: "strip_thinking_history",
			messages: []api.Message{
				{Role: "user", Content: "What is 2+2?"},
				{Role: "assistant", Content: "<|channel>thought\nThinking...<channel|>4"},
				{Role: "user", Content: "And 3+3?"},
			},
			expected: "<bos><|turn>user\nWhat is 2+2?<turn|>\n" +
				"<|turn>model\n4<turn|>\n" +
				"<|turn>user\nAnd 3+3?<turn|>\n" +
				"<|turn>model\n",
		},
		// === Additional edge cases ported from original tests ===
		{
			// Assistant content with tool call — template emits tool_calls before content
			name: "assistant_content_with_tool_call",
			messages: []api.Message{
				{Role: "user", Content: "Weather?"},
				{Role: "assistant", Content: "Let me check.", ToolCalls: []api.ToolCall{{
					Function: api.ToolCallFunction{Name: "get_weather", Arguments: testArgs(map[string]any{"city": "Paris"})},
				}}},
				{Role: "tool", Content: "Sunny"},
			},
			tools: weatherTool(),
			expected: "<bos><|turn>system\n" + weatherDeclRef + "<turn|>\n" +
				"<|turn>user\nWeather?<turn|>\n" +
				"<|turn>model\n<|tool_call>call:get_weather{city:" + q + "Paris" + q + "}<tool_call|>Let me check.<turn|>\n" +
				"<|turn>tool\nSunny<turn|>\n" +
				"<|turn>model\n",
		},
		{
			// Numeric tool call arguments
			name: "numeric_arguments",
			messages: []api.Message{
				{Role: "user", Content: "Add"},
				{Role: "assistant", ToolCalls: []api.ToolCall{{
					Function: api.ToolCallFunction{Name: "add", Arguments: testArgs(map[string]any{"a": float64(1), "b": float64(2)})},
				}}},
				{Role: "tool", Content: `{"result": 3}`},
			},
			tools: addTool(),
			expected: "<bos><|turn>system\n" + addDeclRef + "<turn|>\n" +
				"<|turn>user\nAdd<turn|>\n" +
				"<|turn>model\n<|tool_call>call:add{a:1,b:2}<tool_call|><turn|>\n" +
				"<|turn>tool\n{\"result\": 3}<turn|>\n" +
				"<|turn>model\n",
		},
		{
			// Boolean tool call argument
			name: "boolean_argument",
			messages: []api.Message{
				{Role: "user", Content: "Set flag"},
				{Role: "assistant", ToolCalls: []api.ToolCall{{
					Function: api.ToolCallFunction{Name: "set_flag", Arguments: testArgs(map[string]any{"enabled": true})},
				}}},
				{Role: "tool", Content: "done"},
			},
			tools: flagTool(),
			expected: "<bos><|turn>system\n" + flagDeclRef + "<turn|>\n" +
				"<|turn>user\nSet flag<turn|>\n" +
				"<|turn>model\n<|tool_call>call:set_flag{enabled:true}<tool_call|><turn|>\n" +
				"<|turn>tool\ndone<turn|>\n" +
				"<|turn>model\n",
		},
		{
			// Tool with enum parameter
			name:     "tool_with_enum",
			messages: []api.Message{{Role: "user", Content: "Test"}},
			tools:    modeTool(),
			expected: "<bos><|turn>system\n" + modeDeclRef + "<turn|>\n" +
				"<|turn>user\nTest<turn|>\n<|turn>model\n",
		},
		{
			name:     "unicode_content",
			messages: []api.Message{{Role: "user", Content: "こんにちは"}},
			expected: "<bos><|turn>user\nこんにちは<turn|>\n<|turn>model\n",
		},
		{
			name:     "newlines_in_content",
			messages: []api.Message{{Role: "user", Content: "Line 1\nLine 2\nLine 3"}},
			expected: "<bos><|turn>user\nLine 1\nLine 2\nLine 3<turn|>\n<|turn>model\n",
		},
		{
			// Tool response (raw JSON) followed by user message
			name: "json_tool_response_then_user",
			messages: []api.Message{
				{Role: "user", Content: "Weather?"},
				{Role: "assistant", ToolCalls: []api.ToolCall{{
					Function: api.ToolCallFunction{Name: "get_weather", Arguments: testArgs(map[string]any{"city": "Tokyo"})},
				}}},
				{Role: "tool", Content: `{"temperature": 15, "weather": "sunny"}`},
				{Role: "user", Content: "Thanks!"},
			},
			tools: weatherTool(),
			expected: "<bos><|turn>system\n" + weatherDeclRef + "<turn|>\n" +
				"<|turn>user\nWeather?<turn|>\n" +
				"<|turn>model\n<|tool_call>call:get_weather{city:" + q + "Tokyo" + q + "}<tool_call|><turn|>\n" +
				"<|turn>tool\n{\"temperature\": 15, \"weather\": \"sunny\"}<turn|>\n" +
				"<|turn>user\nThanks!<turn|>\n" +
				"<|turn>model\n",
		},
		// === Ordering and whitespace edge cases ===
		{
			// Tool call arguments are sorted alphabetically
			name: "sorted_args",
			messages: []api.Message{
				{Role: "user", Content: "Go"},
				{Role: "assistant", ToolCalls: []api.ToolCall{{
					Function: api.ToolCallFunction{Name: "bash", Arguments: testArgs(map[string]any{"zzz": "last", "aaa": "first", "mmm": "middle"})},
				}}},
				{Role: "tool", Content: "ok"},
			},
			tools: bashSmallTool(),
			expected: "<bos><|turn>system\n" + bashSmallDeclRef + "<turn|>\n" +
				"<|turn>user\nGo<turn|>\n" +
				"<|turn>model\n<|tool_call>call:bash{aaa:" + q + "first" + q + ",mmm:" + q + "middle" + q + ",zzz:" + q + "last" + q + "}<tool_call|><turn|>\n" +
				"<|turn>tool\nok<turn|>\n" +
				"<|turn>model\n",
		},
		{
			// User content with whitespace is trimmed
			name:     "user_content_trimmed",
			messages: []api.Message{{Role: "user", Content: "  hello  "}},
			expected: "<bos><|turn>user\nhello<turn|>\n<|turn>model\n",
		},
		{
			// Empty tool call arguments
			name: "empty_tool_args",
			messages: []api.Message{
				{Role: "user", Content: "Go"},
				{Role: "assistant", ToolCalls: []api.ToolCall{{
					Function: api.ToolCallFunction{Name: "bash", Arguments: testArgs(map[string]any{})},
				}}},
				{Role: "tool", Content: "ok"},
			},
			tools: bashSmallTool(),
			expected: "<bos><|turn>system\n" + bashSmallDeclRef + "<turn|>\n" +
				"<|turn>user\nGo<turn|>\n" +
				"<|turn>model\n<|tool_call>call:bash{}<tool_call|><turn|>\n" +
				"<|turn>tool\nok<turn|>\n" +
				"<|turn>model\n",
		},
		{
			// Nested object properties in tool declaration
			name:     "nested_object_tool",
			messages: []api.Message{{Role: "user", Content: "Create"}},
			tools:    nestedTool(),
			expected: "<bos><|turn>system\n" + nestedDeclRef + "<turn|>\n" +
				"<|turn>user\nCreate<turn|>\n<|turn>model\n",
		},
		{
			// Array type in tool declaration
			name:     "array_tool",
			messages: []api.Message{{Role: "user", Content: "Batch"}},
			tools:    arrayTool(),
			expected: "<bos><|turn>system\n" + arrayDeclRef + "<turn|>\n" +
				"<|turn>user\nBatch<turn|>\n<|turn>model\n",
		},
		{
			// Assistant whitespace is trimmed (strip_thinking includes | trim)
			name: "assistant_whitespace_trimmed",
			messages: []api.Message{
				{Role: "user", Content: "Hi"},
				{Role: "assistant", Content: "  spaced  "},
				{Role: "user", Content: "More"},
			},
			expected: "<bos><|turn>user\nHi<turn|>\n" +
				"<|turn>model\nspaced<turn|>\n" +
				"<|turn>user\nMore<turn|>\n" +
				"<|turn>model\n",
		},
		{
			// Three sequential tool responses
			name: "three_tool_responses",
			messages: []api.Message{
				{Role: "user", Content: "Do three things"},
				{Role: "assistant", ToolCalls: []api.ToolCall{
					{Function: api.ToolCallFunction{Name: "bash", Arguments: testArgs(map[string]any{"command": "a"})}},
					{Function: api.ToolCallFunction{Name: "bash", Arguments: testArgs(map[string]any{"command": "b"})}},
					{Function: api.ToolCallFunction{Name: "bash", Arguments: testArgs(map[string]any{"command": "c"})}},
				}},
				{Role: "tool", Content: "result-a"},
				{Role: "tool", Content: "result-b"},
				{Role: "tool", Content: "result-c"},
			},
			tools: bashSmallTool(),
			expected: "<bos><|turn>system\n" + bashSmallDeclRef + "<turn|>\n" +
				"<|turn>user\nDo three things<turn|>\n" +
				"<|turn>model\n<|tool_call>call:bash{command:" + q + "a" + q + "}<tool_call|>" +
				"<|tool_call>call:bash{command:" + q + "b" + q + "}<tool_call|>" +
				"<|tool_call>call:bash{command:" + q + "c" + q + "}<tool_call|><turn|>\n" +
				"<|turn>tool\nresult-a<turn|>\n" +
				"<|turn>tool\nresult-b<turn|>\n" +
				"<|turn>tool\nresult-c<turn|>\n" +
				"<|turn>model\n",
		},
		{
			// Assistant with only tool calls, no content field
			name: "tool_calls_no_content",
			messages: []api.Message{
				{Role: "user", Content: "Go"},
				{Role: "assistant", ToolCalls: []api.ToolCall{{
					Function: api.ToolCallFunction{Name: "bash", Arguments: testArgs(map[string]any{"command": "ls"})},
				}}},
				{Role: "tool", Content: "files"},
			},
			tools: bashSmallTool(),
			expected: "<bos><|turn>system\n" + bashSmallDeclRef + "<turn|>\n" +
				"<|turn>user\nGo<turn|>\n" +
				"<|turn>model\n<|tool_call>call:bash{command:" + q + "ls" + q + "}<tool_call|><turn|>\n" +
				"<|turn>tool\nfiles<turn|>\n" +
				"<|turn>model\n",
		},

		// === Coverage gap cases ===
		{
			// Multiple thinking blocks stripped from assistant history
			name: "multiple_thinking_blocks",
			messages: []api.Message{
				{Role: "user", Content: "Hi"},
				{Role: "assistant", Content: "<|channel>Think1<channel|>Middle<|channel>Think2<channel|>Done"},
				{Role: "user", Content: "More"},
			},
			expected: "<bos><|turn>user\nHi<turn|>\n" +
				"<|turn>model\nMiddleDone<turn|>\n" +
				"<|turn>user\nMore<turn|>\n" +
				"<|turn>model\n",
		},
		{
			// Property with no description — just type
			name:     "property_no_description",
			messages: []api.Message{{Role: "user", Content: "Count"}},
			tools:    countTool(),
			expected: "<bos><|turn>system\n" + countDeclRef + "<turn|>\n" +
				"<|turn>user\nCount<turn|>\n<|turn>model\n",
		},
		{
			// System message with leading/trailing whitespace is trimmed
			name: "system_message_trimmed",
			messages: []api.Message{
				{Role: "system", Content: "  You are helpful.  "},
				{Role: "user", Content: "Hi"},
			},
			expected: "<bos><|turn>system\nYou are helpful.<turn|>\n" +
				"<|turn>user\nHi<turn|>\n<|turn>model\n",
		},
		{
			// Deeply nested map in tool call arguments (3 levels)
			name: "nested_map_args",
			messages: []api.Message{
				{Role: "user", Content: "Go"},
				{Role: "assistant", ToolCalls: []api.ToolCall{{
					Function: api.ToolCallFunction{Name: "configure", Arguments: testArgs(map[string]any{
						"config": map[string]any{"db": map[string]any{"host": "localhost", "port": float64(5432)}},
					})},
				}}},
				{Role: "tool", Content: "ok"},
			},
			tools: configureTool(),
			expected: "<bos><|turn>system\n" + configureDeclRef + "<turn|>\n" +
				"<|turn>user\nGo<turn|>\n" +
				"<|turn>model\n<|tool_call>call:configure{config:{db:{host:" + q + "localhost" + q + ",port:5432}}}<tool_call|><turn|>\n" +
				"<|turn>tool\nok<turn|>\n" +
				"<|turn>model\n",
		},
		{
			// Array values in tool call arguments
			name: "array_in_args",
			messages: []api.Message{
				{Role: "user", Content: "Go"},
				{Role: "assistant", ToolCalls: []api.ToolCall{{
					Function: api.ToolCallFunction{Name: "batch", Arguments: testArgs(map[string]any{
						"ids": []any{float64(1), float64(2), float64(3)},
					})},
				}}},
				{Role: "tool", Content: "done"},
			},
			tools: batchArrayTool(),
			expected: "<bos><|turn>system\n" + batchArrayDeclRef + "<turn|>\n" +
				"<|turn>user\nGo<turn|>\n" +
				"<|turn>model\n<|tool_call>call:batch{ids:[1,2,3]}<tool_call|><turn|>\n" +
				"<|turn>tool\ndone<turn|>\n" +
				"<|turn>model\n",
		},
		{
			// Mixed types in array argument (string, number, bool)
			name: "mixed_array_args",
			messages: []api.Message{
				{Role: "user", Content: "Go"},
				{Role: "assistant", ToolCalls: []api.ToolCall{{
					Function: api.ToolCallFunction{Name: "batch", Arguments: testArgs(map[string]any{
						"ids": []any{"a", float64(1), true},
					})},
				}}},
				{Role: "tool", Content: "done"},
			},
			tools: batchArrayTool(),
			expected: "<bos><|turn>system\n" + batchArrayDeclRef + "<turn|>\n" +
				"<|turn>user\nGo<turn|>\n" +
				"<|turn>model\n<|tool_call>call:batch{ids:[" + q + "a" + q + ",1,true]}<tool_call|><turn|>\n" +
				"<|turn>tool\ndone<turn|>\n" +
				"<|turn>model\n",
		},
		{
			// Enum property without description
			name:     "enum_no_description",
			messages: []api.Message{{Role: "user", Content: "Set"}},
			tools:    enumNoDescTool(),
			expected: "<bos><|turn>system\n" + enumNoDescDeclRef + "<turn|>\n" +
				"<|turn>user\nSet<turn|>\n<|turn>model\n",
		},
		{
			// System message that is only whitespace (trims to empty)
			name: "system_whitespace_only",
			messages: []api.Message{
				{Role: "system", Content: "   "},
				{Role: "user", Content: "Hi"},
			},
			expected: "<bos><|turn>system\n<turn|>\n" +
				"<|turn>user\nHi<turn|>\n<|turn>model\n",
		},
		{
			// Empty assistant content (empty string, not nil)
			name: "empty_assistant_content",
			messages: []api.Message{
				{Role: "user", Content: "Hi"},
				{Role: "assistant", Content: ""},
				{Role: "user", Content: "More"},
			},
			expected: "<bos><|turn>user\nHi<turn|>\n" +
				"<|turn>model\n<turn|>\n" +
				"<|turn>user\nMore<turn|>\n" +
				"<|turn>model\n",
		},
		{
			// Map argument with string keys (keys NOT escaped with <|"|>)
			name: "map_arg_string_keys",
			messages: []api.Message{
				{Role: "user", Content: "Go"},
				{Role: "assistant", ToolCalls: []api.ToolCall{{
					Function: api.ToolCallFunction{Name: "configure", Arguments: testArgs(map[string]any{
						"config": map[string]any{"key": "value"},
					})},
				}}},
				{Role: "tool", Content: "ok"},
			},
			tools: configureTool(),
			expected: "<bos><|turn>system\n" + configureDeclRef + "<turn|>\n" +
				"<|turn>user\nGo<turn|>\n" +
				"<|turn>model\n<|tool_call>call:configure{config:{key:" + q + "value" + q + "}}<tool_call|><turn|>\n" +
				"<|turn>tool\nok<turn|>\n" +
				"<|turn>model\n",
		},
		{
			// Mixed properties: some with description, some without
			name:     "mixed_desc_no_desc",
			messages: []api.Message{{Role: "user", Content: "Search"}},
			tools:    searchTool(),
			expected: "<bos><|turn>system\n" + searchDeclRef + "<turn|>\n" +
				"<|turn>user\nSearch<turn|>\n<|turn>model\n",
		},

		// === Round 3 coverage gaps ===
		{
			// Tool content with whitespace is trimmed (template does | trim for all non-model)
			name: "tool_content_trimmed",
			messages: []api.Message{
				{Role: "user", Content: "Go"},
				{Role: "assistant", ToolCalls: []api.ToolCall{{
					Function: api.ToolCallFunction{Name: "bash", Arguments: testArgs(map[string]any{"command": "ls"})},
				}}},
				{Role: "tool", Content: "  result  "},
			},
			tools: bashSmallTool(),
			expected: "<bos><|turn>system\n" + bashSmallDeclRef + "<turn|>\n" +
				"<|turn>user\nGo<turn|>\n" +
				"<|turn>model\n<|tool_call>call:bash{command:" + q + "ls" + q + "}<tool_call|><turn|>\n" +
				"<|turn>tool\nresult<turn|>\n" +
				"<|turn>model\n",
		},
		{
			// Empty system message still emits system turn
			name: "empty_system_message",
			messages: []api.Message{
				{Role: "system", Content: ""},
				{Role: "user", Content: "Hi"},
			},
			expected: "<bos><|turn>system\n<turn|>\n" +
				"<|turn>user\nHi<turn|>\n<|turn>model\n",
		},
		{
			// Nested OBJECT property with required field
			name:     "nested_object_with_required",
			messages: []api.Message{{Role: "user", Content: "Create"}},
			tools:    nestedRequiredTool(),
			expected: "<bos><|turn>system\n" + nestedRequiredDeclRef + "<turn|>\n" +
				"<|turn>user\nCreate<turn|>\n<|turn>model\n",
		},
		{
			// Non-integer float in tool call argument
			name: "float_argument",
			messages: []api.Message{
				{Role: "user", Content: "Calc"},
				{Role: "assistant", ToolCalls: []api.ToolCall{{
					Function: api.ToolCallFunction{Name: "calc", Arguments: testArgs(map[string]any{"value": 3.14})},
				}}},
				{Role: "tool", Content: "ok"},
			},
			tools: calcTool(),
			expected: "<bos><|turn>system\n" + calcDeclRef + "<turn|>\n" +
				"<|turn>user\nCalc<turn|>\n" +
				"<|turn>model\n<|tool_call>call:calc{value:3.14}<tool_call|><turn|>\n" +
				"<|turn>tool\nok<turn|>\n" +
				"<|turn>model\n",
		},
		{
			// Thinking in the last assistant message (stripped before generation prompt)
			name: "thinking_in_last_assistant",
			messages: []api.Message{
				{Role: "user", Content: "Hi"},
				{Role: "assistant", Content: "<|channel>thinking<channel|>Result"},
			},
			expected: "<bos><|turn>user\nHi<turn|>\n" +
				"<|turn>model\nResult<turn|>\n" +
				"<|turn>model\n",
		},
		{
			// Tool content with newlines and leading/trailing whitespace trimmed
			name: "tool_content_multiline_whitespace",
			messages: []api.Message{
				{Role: "user", Content: "Go"},
				{Role: "assistant", ToolCalls: []api.ToolCall{{
					Function: api.ToolCallFunction{Name: "bash", Arguments: testArgs(map[string]any{"command": "ls"})},
				}}},
				{Role: "tool", Content: "\n  file1\n  file2\n"},
			},
			tools: bashSmallTool(),
			expected: "<bos><|turn>system\n" + bashSmallDeclRef + "<turn|>\n" +
				"<|turn>user\nGo<turn|>\n" +
				"<|turn>model\n<|tool_call>call:bash{command:" + q + "ls" + q + "}<tool_call|><turn|>\n" +
				"<|turn>tool\nfile1\n  file2<turn|>\n" +
				"<|turn>model\n",
		},
		{
			// Tool with parameters having only type, no properties
			name:     "tool_params_type_only",
			messages: []api.Message{{Role: "user", Content: "Raw"}},
			tools:    rawTool(),
			expected: "<bos><|turn>system\n" + rawDeclRef + "<turn|>\n" +
				"<|turn>user\nRaw<turn|>\n<|turn>model\n",
		},
		{
			// Multiple required fields at top level
			name:     "multiple_required",
			messages: []api.Message{{Role: "user", Content: "Move"}},
			tools:    moveTool(),
			expected: "<bos><|turn>system\n" + moveDeclRef + "<turn|>\n" +
				"<|turn>user\nMove<turn|>\n<|turn>model\n",
		},
		{
			// Assistant content that is ONLY thinking (strips to empty)
			name: "assistant_only_thinking",
			messages: []api.Message{
				{Role: "user", Content: "Hi"},
				{Role: "assistant", Content: "<|channel>just thinking<channel|>"},
				{Role: "user", Content: "More"},
			},
			expected: "<bos><|turn>user\nHi<turn|>\n" +
				"<|turn>model\n<turn|>\n" +
				"<|turn>user\nMore<turn|>\n" +
				"<|turn>model\n",
		},

		// === Round 4: final coverage gaps ===
		{
			// Thinking enabled with tool calls in same conversation (full agentic scenario)
			name: "thinking_with_tool_calls",
			messages: []api.Message{
				{Role: "system", Content: "You are helpful."},
				{Role: "user", Content: "List files"},
				{Role: "assistant", Content: "<|channel>I should use bash<channel|>", ToolCalls: []api.ToolCall{{
					Function: api.ToolCallFunction{Name: "bash", Arguments: testArgs(map[string]any{"command": "ls"})},
				}}},
				{Role: "tool", Content: "file1.txt"},
				{Role: "assistant", Content: "Here are the files."},
				{Role: "user", Content: "Thanks"},
			},
			tools: bashSmallTool(),
			think: thinkTrue(),
			expected: "<bos><|turn>system\n<|think|>You are helpful." + bashSmallDeclRef + "<turn|>\n" +
				"<|turn>user\nList files<turn|>\n" +
				"<|turn>model\n<|tool_call>call:bash{command:" + q + "ls" + q + "}<tool_call|><turn|>\n" +
				"<|turn>tool\nfile1.txt<turn|>\n" +
				"<|turn>model\nHere are the files.<turn|>\n" +
				"<|turn>user\nThanks<turn|>\n" +
				"<|turn>model\n",
		},
		{
			// Array property without items specification
			name:     "array_without_items",
			messages: []api.Message{{Role: "user", Content: "Tag"}},
			tools:    arrayNoItemsTool(),
			expected: "<bos><|turn>system\n" + arrayNoItemsDeclRef + "<turn|>\n" +
				"<|turn>user\nTag<turn|>\n<|turn>model\n",
		},
		{
			// OBJECT property without description but with nested properties —
			// template hardcodes leading comma on ,properties: and does NOT
			// add comma before type: when description is absent
			name:     "object_no_desc_with_properties",
			messages: []api.Message{{Role: "user", Content: "Update"}},
			tools:    objectNoDescTool(),
			expected: "<bos><|turn>system\n" + objectNoDescDeclRef + "<turn|>\n" +
				"<|turn>user\nUpdate<turn|>\n<|turn>model\n",
		},

		// === Round 5: coding agent patterns ===
		{
			// Chained tool calls — assistant calls tool, gets result, calls another
			// tool, gets result, then the model responds. No user messages in between.
			name: "chained_tool_calls",
			messages: []api.Message{
				{Role: "user", Content: "Set up the project"},
				{Role: "assistant", ToolCalls: []api.ToolCall{{
					Function: api.ToolCallFunction{Name: "bash", Arguments: testArgs(map[string]any{"command": "mkdir src"})},
				}}},
				{Role: "tool", Content: ""},
				{Role: "assistant", ToolCalls: []api.ToolCall{{
					Function: api.ToolCallFunction{Name: "bash", Arguments: testArgs(map[string]any{"command": "touch src/main.go"})},
				}}},
				{Role: "tool", Content: ""},
				{Role: "assistant", Content: "Done."},
				{Role: "user", Content: "Thanks"},
			},
			tools: bashSmallTool(),
			expected: "<bos><|turn>system\n" + bashSmallDeclRef + "<turn|>\n" +
				"<|turn>user\nSet up the project<turn|>\n" +
				"<|turn>model\n<|tool_call>call:bash{command:" + q + "mkdir src" + q + "}<tool_call|><turn|>\n" +
				"<|turn>tool\n<turn|>\n" +
				"<|turn>model\n<|tool_call>call:bash{command:" + q + "touch src/main.go" + q + "}<tool_call|><turn|>\n" +
				"<|turn>tool\n<turn|>\n" +
				"<|turn>model\nDone.<turn|>\n" +
				"<|turn>user\nThanks<turn|>\n" +
				"<|turn>model\n",
		},
		{
			// Tool call with thinking that strips to real remaining content
			name: "tool_call_thinking_with_remaining_content",
			messages: []api.Message{
				{Role: "user", Content: "List files"},
				{Role: "assistant", Content: "<|channel>I need to check the directory<channel|>Let me list the files.", ToolCalls: []api.ToolCall{{
					Function: api.ToolCallFunction{Name: "bash", Arguments: testArgs(map[string]any{"command": "ls"})},
				}}},
				{Role: "tool", Content: "main.go\ngo.mod"},
				{Role: "user", Content: "OK"},
			},
			tools: bashSmallTool(),
			expected: "<bos><|turn>system\n" + bashSmallDeclRef + "<turn|>\n" +
				"<|turn>user\nList files<turn|>\n" +
				"<|turn>model\n<|tool_call>call:bash{command:" + q + "ls" + q + "}<tool_call|>Let me list the files.<turn|>\n" +
				"<|turn>tool\nmain.go\ngo.mod<turn|>\n" +
				"<|turn>user\nOK<turn|>\n" +
				"<|turn>model\n",
		},
		{
			// Argument value containing newlines (multi-line script)
			name: "argument_with_newlines",
			messages: []api.Message{
				{Role: "user", Content: "Run it"},
				{Role: "assistant", ToolCalls: []api.ToolCall{{
					Function: api.ToolCallFunction{Name: "bash", Arguments: testArgs(map[string]any{"command": "echo hello\necho world"})},
				}}},
				{Role: "tool", Content: "hello\nworld"},
			},
			tools: bashSmallTool(),
			expected: "<bos><|turn>system\n" + bashSmallDeclRef + "<turn|>\n" +
				"<|turn>user\nRun it<turn|>\n" +
				"<|turn>model\n<|tool_call>call:bash{command:" + q + "echo hello\necho world" + q + "}<tool_call|><turn|>\n" +
				"<|turn>tool\nhello\nworld<turn|>\n" +
				"<|turn>model\n",
		},
		{
			// Empty string argument value
			name: "empty_string_argument",
			messages: []api.Message{
				{Role: "user", Content: "Go"},
				{Role: "assistant", ToolCalls: []api.ToolCall{{
					Function: api.ToolCallFunction{Name: "bash", Arguments: testArgs(map[string]any{"command": ""})},
				}}},
				{Role: "tool", Content: "error"},
			},
			tools: bashSmallTool(),
			expected: "<bos><|turn>system\n" + bashSmallDeclRef + "<turn|>\n" +
				"<|turn>user\nGo<turn|>\n" +
				"<|turn>model\n<|tool_call>call:bash{command:" + q + q + "}<tool_call|><turn|>\n" +
				"<|turn>tool\nerror<turn|>\n" +
				"<|turn>model\n",
		},
	}

	verifyJinja2 := os.Getenv("VERIFY_JINJA2") != ""
	if verifyJinja2 {
		// Verify python3 and jinja2 are available
		if err := exec.Command("python3", "-c", "import jinja2").Run(); err != nil {
			t.Fatal("VERIFY_JINJA2=1 requires python3 with jinja2: pip install jinja2")
		}
		t.Log("VERIFY_JINJA2=1: verifying expected values against Jinja2 template")
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Compare our renderer against the hardcoded expected value
			renderer := &Gemma4Renderer{useImgTags: RenderImgTags}
			got, err := renderer.Render(tt.messages, tt.tools, tt.think)
			assert.NoError(t, err)
			assert.Equal(t, tt.expected, got)

			// When VERIFY_JINJA2=1, also verify the expected value against
			// the real Jinja2 template rendered by Python.
			if verifyJinja2 {
				jinja2Output := renderWithJinja2(t, tt.messages, tt.tools, tt.think)
				if jinja2Output != tt.expected || jinja2Output != got {
					fmt.Fprintf(os.Stderr, "\nJINJA2 OUTPUT for %s (copy-paste as expected):\n%q\n\n", tt.name, jinja2Output)
				}
				assert.Equal(t, jinja2Output, tt.expected,
					"hardcoded expected value doesn't match Jinja2 template output")
				assert.Equal(t, jinja2Output, got,
					"renderer output doesn't match Jinja2 template output")
			}
		})
	}
}

// renderWithJinja2 shells out to python3 to render messages through the
// Jinja2 chat template. Returns the rendered string.
func renderWithJinja2(t *testing.T, messages []api.Message, tools []api.Tool, think *api.ThinkValue) string {
	t.Helper()

	templatePath, err := filepath.Abs("testdata/gemma4_chat_template.jinja2")
	if err != nil {
		t.Fatalf("failed to get template path: %v", err)
	}

	// Convert messages to the format the Jinja2 template expects.
	// The template uses message['tool_calls'] with function.arguments as a dict.
	type jinja2ToolCall struct {
		Function struct {
			Name      string `json:"name"`
			Arguments any    `json:"arguments"`
		} `json:"function"`
	}
	type jinja2Message struct {
		Role      string           `json:"role"`
		Content   string           `json:"content,omitempty"`
		ToolCalls []jinja2ToolCall `json:"tool_calls,omitempty"`
	}

	var jMsgs []jinja2Message
	for _, m := range messages {
		jm := jinja2Message{Role: m.Role, Content: m.Content}
		for _, tc := range m.ToolCalls {
			jtc := jinja2ToolCall{}
			jtc.Function.Name = tc.Function.Name
			// Convert ToolCallFunctionArguments to a map
			var args map[string]any
			raw, _ := tc.Function.Arguments.MarshalJSON()
			json.Unmarshal(raw, &args)
			jtc.Function.Arguments = args
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
		b, _ := json.Marshal(tools)
		toolsJSON = string(b)
	}

	thinking := "False"
	if think != nil && think.Bool() {
		thinking = "True"
	}

	script := fmt.Sprintf(`
import json
from jinja2 import Environment
tmpl = Environment().from_string(open(%q).read())
msgs = json.loads(%q)
tools = json.loads(%q) if %q != "None" else None
kwargs = {"messages": msgs, "bos_token": "<bos>", "add_generation_prompt": True}
if tools:
    kwargs["tools"] = tools
if %s:
    kwargs["enable_thinking"] = True
print(tmpl.render(**kwargs), end="")
`, templatePath, string(msgsJSON), toolsJSON, toolsJSON, thinking)

	cmd := exec.Command("python3", "-c", script)
	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("python3 failed: %v\nstderr: %s", err, stderr.String())
	}
	return stdout.String()
}

func thinkTrue() *api.ThinkValue {
	return &api.ThinkValue{Value: true}
}
