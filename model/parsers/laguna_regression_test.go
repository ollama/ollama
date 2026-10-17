package parsers

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/ollama/ollama/api"
)

// Regression tests for https://github.com/ollama/ollama/issues/17602:
// ordinary JSON in streamed content must pass through untouched even when
// tools are declared; it must neither be swallowed into a fabricated tool
// call nor abort the reply with a parse error.

func lagunaIssue17602Tools() []api.Tool {
	return []api.Tool{
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "get_weather",
				Description: "Get the weather for a city",
			},
		},
	}
}

func lagunaIssue17602Stream(t *testing.T, parser *LagunaParser, chunks []string) (string, []api.ToolCall) {
	t.Helper()
	var content strings.Builder
	var calls []api.ToolCall
	for i, chunk := range chunks {
		c, _, cs, err := parser.Add(chunk, i == len(chunks)-1)
		require.NoError(t, err)
		content.WriteString(c)
		calls = append(calls, cs...)
	}
	return content.String(), calls
}

func TestLagunaIssue17602JSONWithNameFieldStaysContent(t *testing.T) {
	var parser LagunaParser
	parser.Init(lagunaIssue17602Tools(), nil, nil)

	content, calls := lagunaIssue17602Stream(t, &parser, []string{`Config: {"name": "my-app", "version": 2}`})
	require.Empty(t, calls)
	require.Equal(t, `Config: {"name": "my-app", "version": 2}`, content)
}

func TestLagunaIssue17602StreamedJSONWithNameFieldStaysContent(t *testing.T) {
	var parser LagunaParser
	parser.Init(lagunaIssue17602Tools(), nil, nil)

	content, calls := lagunaIssue17602Stream(t, &parser, []string{`{"na`, `me": "Bob", "age": 8}`})
	require.Empty(t, calls)
	require.Equal(t, `{"name": "Bob", "age": 8}`, content)
}

func TestLagunaIssue17602UnknownToolJSONStaysContent(t *testing.T) {
	var parser LagunaParser
	parser.Init(lagunaIssue17602Tools(), nil, nil)

	content, calls := lagunaIssue17602Stream(t, &parser, []string{`{"name": "not_a_tool", "arguments": {"x": 1}}`})
	require.Empty(t, calls)
	require.Equal(t, `{"name": "not_a_tool", "arguments": {"x": 1}}`, content)
}

func TestLagunaIssue17602PlainJSONMidStreamDoesNotAbort(t *testing.T) {
	var parser LagunaParser
	parser.Init(lagunaIssue17602Tools(), nil, nil)

	content, _, calls, err := parser.Add(`Result: {"value": 42}`, false)
	require.NoError(t, err)
	require.Empty(t, calls)
	require.Equal(t, `Result: {"value": 42}`, content)
}

func TestLagunaIssue17602RealStandaloneJSONToolCallStillParsed(t *testing.T) {
	var parser LagunaParser
	parser.Init(lagunaIssue17602Tools(), nil, nil)

	content, calls := lagunaIssue17602Stream(t, &parser, []string{`{"name": "get_weather", "arguments": {"city": "Paris"}}`})
	require.Equal(t, "", content)
	require.Len(t, calls, 1)
}
