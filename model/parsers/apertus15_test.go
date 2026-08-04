package parsers

import (
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestApertus15ParserCapabilities(t *testing.T) {
	parser := ParserForName("apertus1.5")
	if parser == nil {
		t.Fatal("apertus1.5 parser is not registered")
	}
	if !parser.HasToolSupport() {
		t.Fatal("apertus1.5 parser should support tools")
	}
	if !parser.HasThinkingSupport() {
		t.Fatal("apertus1.5 parser should support thinking")
	}
	for _, token := range []string{
		apertus15InnerStart,
		apertus15InnerEnd,
		apertus15ToolsStart,
		apertus15ToolsEnd,
	} {
		if !slices.Contains(parser.PreservedTokens(), token) {
			t.Fatalf("PreservedTokens() does not contain %q", token)
		}
	}
}

func TestApertus15ParserContentAndThinking(t *testing.T) {
	parser := &Apertus15Parser{}
	parser.Init(nil, nil, &api.ThinkValue{Value: true})
	content, thinking, calls, err := parser.Add(
		apertus15AssistantStart+
			"Before."+
			apertus15InnerStart+"Reason carefully."+apertus15InnerEnd+
			"After."+
			apertus15AssistantEnd,
		true,
	)
	if err != nil {
		t.Fatal(err)
	}
	if content != "Before.After." {
		t.Fatalf("content = %q, want %q", content, "Before.After.")
	}
	if thinking != "Reason carefully." {
		t.Fatalf("thinking = %q, want %q", thinking, "Reason carefully.")
	}
	if len(calls) != 0 {
		t.Fatalf("tool calls = %d, want 0", len(calls))
	}
}

func TestApertus15ParserSuppressesUnexpectedThinking(t *testing.T) {
	parser := &Apertus15Parser{}
	parser.Init(nil, nil, &api.ThinkValue{Value: false})
	content, thinking, _, err := parser.Add(
		apertus15InnerStart+"Hidden."+apertus15InnerEnd+"Visible.",
		true,
	)
	if err != nil {
		t.Fatal(err)
	}
	if content != "Visible." || thinking != "" {
		t.Fatalf("content = %q, thinking = %q, want visible content and suppressed thinking", content, thinking)
	}
}

func TestApertus15ParserToolCalls(t *testing.T) {
	parser := &Apertus15Parser{}
	parser.Init(nil, nil, &api.ThinkValue{Value: true})
	raw := "I will check." +
		apertus15ToolsStart +
		`[{"get_weather":{"city":"Zurich"}},{"get_time":{"offset":2,"dst":true}}]` +
		apertus15ToolsEnd
	content, thinking, calls, err := parser.Add(raw, true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "I will check." || thinking != "" {
		t.Fatalf("content = %q, thinking = %q", content, thinking)
	}
	if len(calls) != 2 {
		t.Fatalf("tool calls = %d, want 2", len(calls))
	}
	if calls[0].Function.Index != 0 || calls[0].Function.Name != "get_weather" {
		t.Fatalf("first tool call = %#v", calls[0].Function)
	}
	if city, ok := calls[0].Function.Arguments.Get("city"); !ok || city != "Zurich" {
		t.Fatalf("city argument = %#v, %v", city, ok)
	}
	if calls[1].Function.Index != 1 || calls[1].Function.Name != "get_time" {
		t.Fatalf("second tool call = %#v", calls[1].Function)
	}
}

func TestApertus15ParserToolCallInsideThinking(t *testing.T) {
	parser := &Apertus15Parser{}
	parser.Init(nil, nil, &api.ThinkValue{Value: true})
	content, thinking, calls, err := parser.Add(
		apertus15InnerStart+"Need weather."+
			apertus15ToolsStart+`[{"get_weather":{"city":"Bern"}}]`+apertus15ToolsEnd,
		true,
	)
	if err != nil {
		t.Fatal(err)
	}
	if content != "" || thinking != "Need weather." || len(calls) != 1 {
		t.Fatalf("content = %q, thinking = %q, calls = %d", content, thinking, len(calls))
	}
}

func TestApertus15ParserAcceptsConsumedToolSuffix(t *testing.T) {
	parser := &Apertus15Parser{}
	parser.Init(nil, nil, nil)
	_, _, calls, err := parser.Add(
		apertus15ToolsStart+`[{"lookup":{"id":"42"}}]`,
		true,
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(calls) != 1 || calls[0].Function.Name != "lookup" {
		t.Fatalf("tool calls = %#v", calls)
	}
}

func TestApertus15ParserMultipleToolEnvelopesWithContent(t *testing.T) {
	parser := &Apertus15Parser{}
	parser.Init(nil, nil, nil)
	content, thinking, calls, err := parser.Add(
		"Before."+
			apertus15ToolsStart+`[{"first":{"x":1}}]`+apertus15ToolsEnd+
			"Between."+
			apertus15ToolsStart+`[{"second":{"y":2}}]`+apertus15ToolsEnd+
			"After.",
		true,
	)
	if err != nil {
		t.Fatal(err)
	}
	if content != "Before.Between.After." || thinking != "" {
		t.Fatalf("content = %q, thinking = %q", content, thinking)
	}
	if len(calls) != 2 || calls[0].Function.Index != 0 || calls[0].Function.Name != "first" ||
		calls[1].Function.Index != 1 || calls[1].Function.Name != "second" {
		t.Fatalf("tool calls = %#v", calls)
	}
}

func TestApertus15ParserInitResetsToolCallIndexes(t *testing.T) {
	parser := &Apertus15Parser{}
	for _, name := range []string{"first", "second"} {
		parser.Init(nil, nil, nil)
		_, _, calls, err := parser.Add(
			apertus15ToolsStart+`[{"`+name+`":{}}]`+apertus15ToolsEnd,
			true,
		)
		if err != nil {
			t.Fatal(err)
		}
		if len(calls) != 1 || calls[0].Function.Index != 0 || calls[0].Function.Name != name {
			t.Fatalf("tool calls after Init = %#v", calls)
		}
	}
}

func TestApertus15ParserEveryByteBoundary(t *testing.T) {
	raw := apertus15AssistantStart +
		apertus15InnerStart + "Think." + apertus15InnerEnd +
		"Answer." +
		apertus15ToolsStart + `[{"lookup":{"id":"42"}}]` + apertus15ToolsEnd +
		apertus15AssistantEnd

	for split := 0; split <= len(raw); split++ {
		t.Run("", func(t *testing.T) {
			parser := &Apertus15Parser{}
			parser.Init(nil, nil, &api.ThinkValue{Value: true})
			var content, thinking strings.Builder
			var calls []api.ToolCall

			for index, chunk := range []string{raw[:split], raw[split:]} {
				gotContent, gotThinking, gotCalls, err := parser.Add(chunk, index == 1)
				if err != nil {
					t.Fatalf("split %d: %v", split, err)
				}
				content.WriteString(gotContent)
				thinking.WriteString(gotThinking)
				calls = append(calls, gotCalls...)
			}
			if content.String() != "Answer." || thinking.String() != "Think." ||
				len(calls) != 1 || calls[0].Function.Name != "lookup" {
				t.Fatalf("split %d: content = %q, thinking = %q, calls = %#v", split, content.String(), thinking.String(), calls)
			}
		})
	}
}

func TestApertus15ParserByteAtATime(t *testing.T) {
	raw := apertus15InnerStart + "Reason." + apertus15InnerEnd +
		apertus15ToolsStart + `[{"first":{"x":1}},{"second":{"y":[1,2]}}]` + apertus15ToolsEnd
	parser := &Apertus15Parser{}
	parser.Init(nil, nil, &api.ThinkValue{Value: true})
	var content, thinking strings.Builder
	var calls []api.ToolCall
	for index := range len(raw) {
		gotContent, gotThinking, gotCalls, err := parser.Add(raw[index:index+1], false)
		if err != nil {
			t.Fatal(err)
		}
		content.WriteString(gotContent)
		thinking.WriteString(gotThinking)
		calls = append(calls, gotCalls...)
	}
	gotContent, gotThinking, gotCalls, err := parser.Add("", true)
	if err != nil {
		t.Fatal(err)
	}
	content.WriteString(gotContent)
	thinking.WriteString(gotThinking)
	calls = append(calls, gotCalls...)
	if content.String() != "" || thinking.String() != "Reason." || len(calls) != 2 {
		t.Fatalf("content = %q, thinking = %q, calls = %d", content.String(), thinking.String(), len(calls))
	}
}

func TestApertus15ParserRejectsMalformedToolCalls(t *testing.T) {
	tests := []string{
		apertus15ToolsStart + `[{"lookup":` + apertus15ToolsEnd,
		apertus15ToolsStart + `[{"one":{},"two":{}}]` + apertus15ToolsEnd,
		apertus15ToolsStart + `[{"lookup":"not-an-object"}]` + apertus15ToolsEnd,
		apertus15ToolsStart + `[{"lookup":[1,2]}]` + apertus15ToolsEnd,
		apertus15ToolsStart + `[{"lookup":null}]` + apertus15ToolsEnd,
	}
	for _, raw := range tests {
		parser := &Apertus15Parser{}
		parser.Init(nil, nil, nil)
		if _, _, _, err := parser.Add(raw, true); err == nil {
			t.Fatalf("expected parsing error for %q", raw)
		}
	}
}
