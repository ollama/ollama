package parsers

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func sayToolLaguna() []api.Tool {
	props := api.NewToolPropertiesMap()
	props.Set("text", api.ToolProperty{Type: api.PropertyType{"string"}})
	return []api.Tool{{
		Function: api.ToolFunction{
			Name:        "say",
			Description: "say a string",
			Parameters: api.ToolFunctionParameters{
				Type:       "object",
				Properties: props,
				Required:   []string{"text"},
			},
		},
	}}
}

// A literal "<tool_call>" inside an <arg_value> block is part of the
// argument's value, not the start of a new call. See #19902.
func TestLagunaNestedToolCallOpenerKeepsArgument(t *testing.T) {
	p := &LagunaParser{}
	p.Init(sayToolLaguna(), nil, nil)

	input := "<tool_call>say\n<arg_key>text</arg_key>\n<arg_value>before <tool_call> after</arg_value>\n</tool_call>"
	_, _, calls, err := p.Add(input, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(calls) != 1 {
		t.Fatalf("got %d calls, want 1", len(calls))
	}
	if calls[0].Function.Name != "say" {
		t.Fatalf("name = %q, want say", calls[0].Function.Name)
	}
	got, ok := calls[0].Function.Arguments.Get("text")
	if !ok {
		t.Fatal("text argument missing")
	}
	if got != "before <tool_call> after" {
		t.Errorf("text = %q, want %q", got, "before <tool_call> after")
	}
}

// Control: the same shape without the nested opener still parses.
func TestLagunaPlainArgumentStillParses(t *testing.T) {
	p := &LagunaParser{}
	p.Init(sayToolLaguna(), nil, nil)

	input := "<tool_call>say\n<arg_key>text</arg_key>\n<arg_value>before SAFE after</arg_value>\n</tool_call>"
	_, _, calls, err := p.Add(input, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(calls) != 1 {
		t.Fatalf("got %d calls, want 1", len(calls))
	}
	got, _ := calls[0].Function.Arguments.Get("text")
	if got != "before SAFE after" {
		t.Errorf("text = %q, want %q", got, "before SAFE after")
	}
}

// Two calls in a row still split correctly even when the first one's value
// contains a nested opener.
func TestLagunaSequentialCallsAfterNestedOpener(t *testing.T) {
	p := &LagunaParser{}
	p.Init(sayToolLaguna(), nil, nil)

	input := "<tool_call>say\n<arg_key>text</arg_key>\n<arg_value>a <tool_call> b</arg_value>\n</tool_call>" +
		"<tool_call>say\n<arg_key>text</arg_key>\n<arg_value>second</arg_value>\n</tool_call>"
	_, _, calls, err := p.Add(input, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(calls) != 2 {
		t.Fatalf("got %d calls, want 2", len(calls))
	}
	if first, _ := calls[0].Function.Arguments.Get("text"); first != "a <tool_call> b" {
		t.Errorf("first text = %q", first)
	}
	if second, _ := calls[1].Function.Arguments.Get("text"); second != "second" {
		t.Errorf("second text = %q", second)
	}
}
