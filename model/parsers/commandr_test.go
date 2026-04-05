package parsers

import (
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCommandRParser_SingleToolCall(t *testing.T) {
	parser := &CommandRParser{}
	parser.Init(nil, nil, nil)

	// Simulate streaming: content comes in chunks
	content, thinking, calls, err := parser.Add("Action: ```json\n", false)
	require.NoError(t, err)
	assert.Empty(t, content)
	assert.Empty(t, thinking)
	assert.Empty(t, calls)

	content, thinking, calls, err = parser.Add(`[
    {
        "tool_name": "get_weather",
        "parameters": {"location": "San Francisco", "unit": "celsius"}
    }
]`, false)
	require.NoError(t, err)
	assert.Empty(t, content)
	assert.Empty(t, thinking)
	assert.Empty(t, calls)

	content, thinking, calls, err = parser.Add("\n```", false)
	require.NoError(t, err)
	assert.Empty(t, content)
	assert.Empty(t, thinking)
	require.Len(t, calls, 1)
	assert.Equal(t, "get_weather", calls[0].Function.Name)

	location, ok := calls[0].Function.Arguments.Get("location")
	assert.True(t, ok)
	assert.Equal(t, "San Francisco", location)

	unit, ok := calls[0].Function.Arguments.Get("unit")
	assert.True(t, ok)
	assert.Equal(t, "celsius", unit)
}

func TestCommandRParser_MultipleToolCalls(t *testing.T) {
	parser := &CommandRParser{}
	parser.Init(nil, nil, nil)

	input := `Action: ` + "```json\n" + `[
    {
        "tool_name": "get_weather",
        "parameters": {"location": "New York"}
    },
    {
        "tool_name": "get_time",
        "parameters": {"timezone": "EST"}
    }
]` + "\n```"

	content, thinking, calls, err := parser.Add(input, false)
	require.NoError(t, err)
	assert.Empty(t, content)
	assert.Empty(t, thinking)
	require.Len(t, calls, 2)
	assert.Equal(t, "get_weather", calls[0].Function.Name)
	assert.Equal(t, "get_time", calls[1].Function.Name)
}

func TestCommandRParser_DirectlyAnswer(t *testing.T) {
	parser := &CommandRParser{}
	parser.Init(nil, nil, nil)

	// When command-r decides not to use tools, it outputs directly-answer
	input := `Action: ` + "```json\n" + `[
    {
        "tool_name": "directly-answer",
        "parameters": {}
    }
]` + "\n```"

	_, thinking, calls, err := parser.Add(input, false)
	require.NoError(t, err)
	assert.Empty(t, thinking)
	// directly-answer should be filtered out
	assert.Empty(t, calls)
}

func TestCommandRParser_ContentBeforeAction(t *testing.T) {
	parser := &CommandRParser{}
	parser.Init(nil, nil, nil)

	// Command-r sometimes outputs thinking text before the Action:
	c, _, calls, err := parser.Add("I'll look up the weather for you.\n", false)
	require.NoError(t, err)
	assert.Equal(t, "I'll look up the weather for you.\n", c)
	assert.Empty(t, calls)

	c, _, calls, err = parser.Add("Action: ```json\n", false)
	require.NoError(t, err)
	assert.Empty(t, c)
	assert.Empty(t, calls)

	content, _, calls, err := parser.Add(`[{"tool_name": "get_weather", "parameters": {"location": "NYC"}}]`+"\n```", false)
	require.NoError(t, err)
	assert.Empty(t, content)
	require.Len(t, calls, 1)
	assert.Equal(t, "get_weather", calls[0].Function.Name)
}

func TestCommandRParser_PlainTextResponse(t *testing.T) {
	parser := &CommandRParser{}
	parser.Init(nil, nil, nil)

	// When no tools are called, content should flow through
	content, _, calls, err := parser.Add("Hello! How can I help you today?", false)
	require.NoError(t, err)
	assert.Equal(t, "Hello! How can I help you today?", content)
	assert.Empty(t, calls)
}

func TestCommandRParser_StreamingTokenByToken(t *testing.T) {
	parser := &CommandRParser{}
	parser.Init(nil, nil, nil)

	// Simulate token-by-token streaming like Ollama does
	tokens := []string{
		"Action", ":", " ```", "json", "\n",
		"[\n    {\n", `        "tool_name": "bash",`, "\n",
		`        "parameters": {"command": "ls -la"}`, "\n",
		"    }\n]", "\n```",
	}

	var allContent string
	var allCalls []api.ToolCall

	for i, token := range tokens {
		done := i == len(tokens)-1
		content, _, calls, err := parser.Add(token, done && false)
		require.NoError(t, err)
		allContent += content
		allCalls = append(allCalls, calls...)
	}

	assert.Empty(t, allContent)
	require.Len(t, allCalls, 1)
	assert.Equal(t, "bash", allCalls[0].Function.Name)
	cmd, ok := allCalls[0].Function.Arguments.Get("command")
	assert.True(t, ok)
	assert.Equal(t, "ls -la", cmd)
}

func TestCommandRParser_DoneWithPendingToolCalls(t *testing.T) {
	parser := &CommandRParser{}
	parser.Init(nil, nil, nil)

	// Simulate a case where the model output ends without closing ```
	parser.Add("Action: ```json\n", false)
	content, _, calls, err := parser.Add(`[{"tool_name": "bash", "parameters": {"command": "pwd"}}]`, true)
	require.NoError(t, err)

	// Should still parse the tool calls on done
	if len(calls) > 0 {
		assert.Equal(t, "bash", calls[0].Function.Name)
	} else {
		// Or return as content if parsing failed
		assert.NotEmpty(t, content)
	}
}

func TestCommandRParser_PartialActionTag(t *testing.T) {
	parser := &CommandRParser{}
	parser.Init(nil, nil, nil)

	// Test partial overlap with "Action:"
	content, _, calls, err := parser.Add("Here is my response. Act", false)
	require.NoError(t, err)
	// "Act" overlaps with "Action:" so it should be withheld
	assert.Equal(t, "Here is my response. ", content)
	assert.Empty(t, calls)

	// Continue with non-matching text
	content, _, calls, err = parser.Add("ually, let me think...", false)
	require.NoError(t, err)
	assert.Equal(t, "Actually, let me think...", content)
	assert.Empty(t, calls)
}

func TestCommandRParser_SingleObject(t *testing.T) {
	parser := &CommandRParser{}
	parser.Init(nil, nil, nil)

	// Some versions output single object without array wrapper
	input := `Action: ` + "```json\n" + `{
    "tool_name": "read_file",
    "parameters": {"path": "/etc/hosts"}
}` + "\n```"

	content, _, calls, err := parser.Add(input, false)
	require.NoError(t, err)
	assert.Empty(t, content)
	require.Len(t, calls, 1)
	assert.Equal(t, "read_file", calls[0].Function.Name)
}

func TestParserForName_CommandR(t *testing.T) {
	parser := ParserForName("command-r")
	require.NotNil(t, parser)
	assert.True(t, parser.HasToolSupport())
	assert.False(t, parser.HasThinkingSupport())
}
