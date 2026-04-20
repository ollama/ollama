package parsers

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func lagunaTestTools() []api.Tool {
	props := api.NewToolPropertiesMap()
	props.Set("location", api.ToolProperty{Type: api.PropertyType{"string"}})
	props.Set("days", api.ToolProperty{Type: api.PropertyType{"integer"}})
	return []api.Tool{{
		Function: api.ToolFunction{
			Name: "get_weather",
			Parameters: api.ToolFunctionParameters{
				Properties: props,
			},
		},
	}}
}

func TestLagunaParserToolCall(t *testing.T) {
	parser := ParserForName("laguna")
	if parser == nil {
		t.Fatal("expected laguna parser")
	}
	if !parser.HasToolSupport() || !parser.HasThinkingSupport() {
		t.Fatal("laguna parser should advertise tools and thinking")
	}

	parser.Init(lagunaTestTools(), nil, nil)
	content, thinking, calls, err := parser.Add("<tool_call>get_weather\n<arg_key>location</arg_key>\n<arg_value>Paris</arg_value>\n<arg_key>days</arg_key>\n<arg_value>3</arg_value>\n</tool_call>", true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "" || thinking != "" {
		t.Fatalf("content=%q thinking=%q, want empty", content, thinking)
	}
	if len(calls) != 1 {
		t.Fatalf("calls=%d, want 1", len(calls))
	}
	if calls[0].Function.Name != "get_weather" {
		t.Fatalf("name=%q, want get_weather", calls[0].Function.Name)
	}
	if got, _ := calls[0].Function.Arguments.Get("location"); got != "Paris" {
		t.Fatalf("location=%v, want Paris", got)
	}
	if got, _ := calls[0].Function.Arguments.Get("days"); got != 3 {
		t.Fatalf("days=%v, want 3", got)
	}
}

func TestLagunaParserJSONToolCall(t *testing.T) {
	parser := ParserForName("laguna")
	parser.Init(lagunaTestTools(), nil, nil)

	_, _, calls, err := parser.Add("<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Paris\",\"days\":3}}\n</tool_call>", true)
	if err != nil {
		t.Fatal(err)
	}
	if len(calls) != 1 {
		t.Fatalf("calls=%d, want 1", len(calls))
	}
	if calls[0].Function.Name != "get_weather" {
		t.Fatalf("name=%q, want get_weather", calls[0].Function.Name)
	}
	if got, _ := calls[0].Function.Arguments.Get("location"); got != "Paris" {
		t.Fatalf("location=%v, want Paris", got)
	}
	if got, _ := calls[0].Function.Arguments.Get("days"); got != float64(3) {
		t.Fatalf("days=%v, want 3", got)
	}
}

func TestLagunaParserStandaloneJSONToolCall(t *testing.T) {
	parser := ParserForName("laguna")
	parser.Init(lagunaTestTools(), nil, nil)

	content, thinking, calls, err := parser.Add("{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Paris\",\"days\":3}}", true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "" || thinking != "" {
		t.Fatalf("content=%q thinking=%q", content, thinking)
	}
	if len(calls) != 1 {
		t.Fatalf("calls=%d, want 1", len(calls))
	}
	if calls[0].Function.Name != "get_weather" {
		t.Fatalf("name=%q, want get_weather", calls[0].Function.Name)
	}
}

func TestLagunaParserStandaloneJSONToolCallAfterLeadingContent(t *testing.T) {
	parser := ParserForName("laguna")
	parser.Init(lagunaTestTools(), nil, nil)

	content, thinking, calls, err := parser.Add("Let me call the weather tool.\n{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Paris\"}}", true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "Let me call the weather tool." || thinking != "" {
		t.Fatalf("content=%q thinking=%q", content, thinking)
	}
	if len(calls) != 1 {
		t.Fatalf("calls=%d, want 1", len(calls))
	}
	if calls[0].Function.Name != "get_weather" {
		t.Fatalf("name=%q, want get_weather", calls[0].Function.Name)
	}
}

func TestLagunaParserStreamingStandaloneJSONToolCall(t *testing.T) {
	parser := ParserForName("laguna")
	parser.Init(lagunaTestTools(), nil, nil)

	content, thinking, calls, err := parser.Add("{\"name\":\"get_weather\",\"arguments\":{\"location\":\"San Francisco,", false)
	if err != nil {
		t.Fatal(err)
	}
	if content != "" || thinking != "" || len(calls) != 0 {
		t.Fatalf("first chunk content=%q thinking=%q calls=%d", content, thinking, len(calls))
	}

	content, thinking, calls, err = parser.Add(" CA\"}}", true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "" || thinking != "" || len(calls) != 1 {
		t.Fatalf("second chunk content=%q thinking=%q calls=%d", content, thinking, len(calls))
	}
	if calls[0].Function.Name != "get_weather" {
		t.Fatalf("name=%q, want get_weather", calls[0].Function.Name)
	}
	if got, _ := calls[0].Function.Arguments.Get("location"); got != "San Francisco, CA" {
		t.Fatalf("location=%v, want San Francisco, CA", got)
	}
}

func TestLagunaParserNameLineJSONToolCall(t *testing.T) {
	parser := ParserForName("laguna")
	parser.Init(lagunaTestTools(), nil, nil)

	_, _, calls, err := parser.Add("<tool_call>get_weather\n{\"location\":\"San Francisco\"}</tool_call>", true)
	if err != nil {
		t.Fatal(err)
	}
	if len(calls) != 1 {
		t.Fatalf("calls=%d, want 1", len(calls))
	}
	if calls[0].Function.Name != "get_weather" {
		t.Fatalf("name=%q, want get_weather", calls[0].Function.Name)
	}
	if got, _ := calls[0].Function.Arguments.Get("location"); got != "San Francisco" {
		t.Fatalf("location=%v, want San Francisco", got)
	}
}

func TestLagunaParserNormalizesCommonToolAliases(t *testing.T) {
	props := api.NewToolPropertiesMap()
	props.Set("path", api.ToolProperty{Type: api.PropertyType{"string"}})
	tools := []api.Tool{{
		Function: api.ToolFunction{
			Name: "read",
			Parameters: api.ToolFunctionParameters{
				Properties: props,
			},
		},
	}}

	parser := ParserForName("laguna")
	parser.Init(tools, nil, nil)

	_, _, calls, err := parser.Add("<tool_call>\n{\"name\":\"read_file\",\"arguments\":{\"path\":\"./go.mod\"}}\n</tool_call>", true)
	if err != nil {
		t.Fatal(err)
	}
	if len(calls) != 1 {
		t.Fatalf("calls=%d, want 1", len(calls))
	}
	if calls[0].Function.Name != "read" {
		t.Fatalf("name=%q, want read", calls[0].Function.Name)
	}
	if got, _ := calls[0].Function.Arguments.Get("path"); got != "./go.mod" {
		t.Fatalf("path=%v, want ./go.mod", got)
	}
}

func TestLagunaParserIgnoresDuplicatedNestedToolCall(t *testing.T) {
	props := api.NewToolPropertiesMap()
	props.Set("name", api.ToolProperty{Type: api.PropertyType{"string"}})
	tools := []api.Tool{{
		Function: api.ToolFunction{
			Name: "skill",
			Parameters: api.ToolFunctionParameters{
				Properties: props,
			},
		},
	}}

	parser := ParserForName("laguna")
	parser.Init(tools, nil, nil)

	_, _, calls, err := parser.Add("<tool_call>skill\n{\"name\":\"git-diff-review\"}\n<tool_call>skill\n{\"name\":\"git-diff-review\"}</tool_call>", true)
	if err != nil {
		t.Fatal(err)
	}
	if len(calls) != 1 {
		t.Fatalf("calls=%d, want 1", len(calls))
	}
	if calls[0].Function.Name != "skill" {
		t.Fatalf("name=%q, want skill", calls[0].Function.Name)
	}
	if got, _ := calls[0].Function.Arguments.Get("name"); got != "git-diff-review" {
		t.Fatalf("name arg=%v, want git-diff-review", got)
	}
}

func TestLagunaParserThinkingThenTool(t *testing.T) {
	parser := ParserForName("laguna")
	parser.Init(lagunaTestTools(), nil, &api.ThinkValue{Value: true})

	content, thinking, calls, err := parser.Add("<think>Need current weather.</think>\n<tool_call>get_weather\n<arg_key>location</arg_key>\n<arg_value>SF</arg_value>\n</tool_call>", true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "" {
		t.Fatalf("content=%q, want empty", content)
	}
	if thinking != "Need current weather." {
		t.Fatalf("thinking=%q, want reasoning", thinking)
	}
	if len(calls) != 1 || calls[0].Function.Name != "get_weather" {
		t.Fatalf("unexpected calls: %#v", calls)
	}
}

func TestLagunaParserUserTaggedToolAlias(t *testing.T) {
	parser := ParserForName("laguna")
	parser.Init(lagunaTestTools(), nil, nil)

	content, thinking, calls, err := parser.Add("<user>get_weather\n<arg_key>location</arg_key>\n<arg_value>San Francisco, CA</arg_value>\n</user>", true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "" || thinking != "" {
		t.Fatalf("content=%q thinking=%q, want empty", content, thinking)
	}
	if len(calls) != 1 {
		t.Fatalf("calls=%d, want 1", len(calls))
	}
	if calls[0].Function.Name != "get_weather" {
		t.Fatalf("name=%q, want get_weather", calls[0].Function.Name)
	}
	if got, _ := calls[0].Function.Arguments.Get("location"); got != "San Francisco, CA" {
		t.Fatalf("location=%v, want San Francisco, CA", got)
	}
}

func TestLagunaParserUserTaggedToolAliasWithLeadingContent(t *testing.T) {
	parser := ParserForName("laguna")
	props := api.NewToolPropertiesMap()
	props.Set("path", api.ToolProperty{Type: api.PropertyType{"string"}})
	tools := []api.Tool{{
		Function: api.ToolFunction{
			Name: "read",
			Parameters: api.ToolFunctionParameters{
				Properties: props,
			},
		},
	}}
	parser.Init(tools, nil, nil)

	content, thinking, calls, err := parser.Add("I'll read the file for you.\n<user>read\n<arg_key>path</arg_key>\n<arg_value>/Users/test/code/myproject/go.mod</arg_value>\n</user>", true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "I'll read the file for you." || thinking != "" {
		t.Fatalf("content=%q thinking=%q", content, thinking)
	}
	if len(calls) != 1 {
		t.Fatalf("calls=%d, want 1", len(calls))
	}
	if calls[0].Function.Name != "read" {
		t.Fatalf("name=%q, want read", calls[0].Function.Name)
	}
	if got, _ := calls[0].Function.Arguments.Get("path"); got != "/Users/test/code/myproject/go.mod" {
		t.Fatalf("path=%v, want /Users/test/code/myproject/go.mod", got)
	}
}

func TestLagunaParserUserTaggedJSONToolCallWithLeadingContent(t *testing.T) {
	parser := ParserForName("laguna")
	props := api.NewToolPropertiesMap()
	props.Set("command", api.ToolProperty{Type: api.PropertyType{"string"}})
	tools := []api.Tool{{
		Function: api.ToolFunction{
			Name: "bash",
			Parameters: api.ToolFunctionParameters{
				Properties: props,
			},
		},
	}}
	parser.Init(tools, nil, nil)

	content, thinking, calls, err := parser.Add("I'll run git diff for you.<user>\n{\"name\":\"bash\",\"arguments\":{\"command\":\"git diff main\"}}\n</user>", true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "I'll run git diff for you." || thinking != "" {
		t.Fatalf("content=%q thinking=%q", content, thinking)
	}
	if len(calls) != 1 {
		t.Fatalf("calls=%d, want 1", len(calls))
	}
	if calls[0].Function.Name != "bash" {
		t.Fatalf("name=%q, want bash", calls[0].Function.Name)
	}
	if got, _ := calls[0].Function.Arguments.Get("command"); got != "git diff main" {
		t.Fatalf("command=%v, want git diff main", got)
	}
}

func TestLagunaParserStreamingUserTaggedToolAliasAfterContent(t *testing.T) {
	parser := ParserForName("laguna")
	props := api.NewToolPropertiesMap()
	props.Set("path", api.ToolProperty{Type: api.PropertyType{"string"}})
	tools := []api.Tool{{
		Function: api.ToolFunction{
			Name: "read",
			Parameters: api.ToolFunctionParameters{
				Properties: props,
			},
		},
	}}
	parser.Init(tools, nil, nil)

	content, thinking, calls, err := parser.Add("I'll read the file for you.<us", false)
	if err != nil {
		t.Fatal(err)
	}
	if content != "I'll read the file for you." || thinking != "" || len(calls) != 0 {
		t.Fatalf("first chunk content=%q thinking=%q calls=%d", content, thinking, len(calls))
	}

	content, thinking, calls, err = parser.Add("er>read\n<arg_key>path</arg_key>\n<arg_value>/Users/test/code/myproject/go.mod</arg_value>\n</user>", true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "" || thinking != "" {
		t.Fatalf("second chunk content=%q thinking=%q", content, thinking)
	}
	if len(calls) != 1 {
		t.Fatalf("calls=%d, want 1", len(calls))
	}
	if calls[0].Function.Name != "read" {
		t.Fatalf("name=%q, want read", calls[0].Function.Name)
	}
	if got, _ := calls[0].Function.Arguments.Get("path"); got != "/Users/test/code/myproject/go.mod" {
		t.Fatalf("path=%v, want /Users/test/code/myproject/go.mod", got)
	}
}

func TestLagunaParserUserTaggedNonToolContent(t *testing.T) {
	parser := ParserForName("laguna")
	parser.Init(lagunaTestTools(), nil, nil)

	content, thinking, calls, err := parser.Add("<user>hello</user>", true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "<user>hello</user>" || thinking != "" || len(calls) != 0 {
		t.Fatalf("content=%q thinking=%q calls=%d", content, thinking, len(calls))
	}
}

func TestLagunaParserThinkingDefaultsOn(t *testing.T) {
	parser := ParserForName("laguna")
	parser.Init(nil, nil, nil)
	content, thinking, calls, err := parser.Add("<think>Need to reason.</think>\nDirect answer.", true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "Direct answer." || thinking != "Need to reason." || len(calls) != 0 {
		t.Fatalf("content=%q thinking=%q calls=%d", content, thinking, len(calls))
	}
}

func TestLagunaParserThinkingDefaultsOnWhenToolsPresent(t *testing.T) {
	parser := ParserForName("laguna")
	parser.Init(lagunaTestTools(), nil, nil)
	content, thinking, calls, err := parser.Add("<think>Need to reason.</think>\n<tool_call>get_weather\n<arg_key>location</arg_key>\n<arg_value>Paris</arg_value>\n</tool_call>", true)
	if err != nil {
		t.Fatal(err)
	}
	if thinking != "Need to reason." || len(calls) != 1 {
		t.Fatalf("content=%q thinking=%q calls=%d", content, thinking, len(calls))
	}
	if content != "" {
		t.Fatalf("content=%q, want thinking block suppressed from content when default thinking is enabled", content)
	}
}

func TestLagunaParserThinkingExplicitlyDisabled(t *testing.T) {
	parser := ParserForName("laguna")
	parser.Init(nil, nil, &api.ThinkValue{Value: false})
	content, thinking, calls, err := parser.Add("<think>Hidden?</think>\nDirect answer.", true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "Direct answer." || thinking != "" || len(calls) != 0 {
		t.Fatalf("content=%q thinking=%q calls=%d", content, thinking, len(calls))
	}
}

func TestLagunaParserThinkingExplicitlyDisabledDropsLeadingCloseTag(t *testing.T) {
	parser := ParserForName("laguna")
	parser.Init(nil, nil, &api.ThinkValue{Value: false})
	content, thinking, calls, err := parser.Add("</think>\nTokyo\n", true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "Tokyo\n" || thinking != "" || len(calls) != 0 {
		t.Fatalf("content=%q thinking=%q calls=%d", content, thinking, len(calls))
	}
}

func TestLagunaParserThinkingEnabledDropsLeadingCloseTag(t *testing.T) {
	parser := ParserForName("laguna")
	parser.Init(nil, nil, &api.ThinkValue{Value: true})
	content, thinking, calls, err := parser.Add("</think>\nTokyo\n", true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "Tokyo\n" || thinking != "" || len(calls) != 0 {
		t.Fatalf("content=%q thinking=%q calls=%d", content, thinking, len(calls))
	}
}

func TestLagunaParserThinkingDefaultOnDropsLeadingCloseTag(t *testing.T) {
	parser := ParserForName("laguna")
	parser.Init(nil, nil, nil)
	content, thinking, calls, err := parser.Add("</think>\nTokyo\n", true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "Tokyo\n" || thinking != "" || len(calls) != 0 {
		t.Fatalf("content=%q thinking=%q calls=%d", content, thinking, len(calls))
	}
}

func TestLagunaParserThinkingEnabledUntaggedAnswerIsContent(t *testing.T) {
	parser := ParserForName("laguna")
	parser.Init(nil, nil, &api.ThinkValue{Value: true})
	content, thinking, calls, err := parser.Add("Direct answer.", true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "Direct answer." || thinking != "" || len(calls) != 0 {
		t.Fatalf("content=%q thinking=%q calls=%d", content, thinking, len(calls))
	}
}

func TestLagunaParserSplitToolTag(t *testing.T) {
	parser := ParserForName("laguna")
	parser.Init(lagunaTestTools(), nil, &api.ThinkValue{Value: true})

	content, thinking, calls, err := parser.Add("<think>Need lookup<tool_c", false)
	if err != nil {
		t.Fatal(err)
	}
	if content != "" || thinking != "Need lookup" || len(calls) != 0 {
		t.Fatalf("first chunk content=%q thinking=%q calls=%d", content, thinking, len(calls))
	}

	content, thinking, calls, err = parser.Add("all>get_weather\n<arg_key>location</arg_key>\n<arg_value>SF</arg_value>\n</tool_call>", true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "" || thinking != "" || len(calls) != 1 {
		t.Fatalf("second chunk content=%q thinking=%q calls=%d", content, thinking, len(calls))
	}
}
