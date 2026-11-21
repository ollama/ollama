package parsers

import (
	"encoding/json"
	"reflect"
	"slices"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestInkling2ThinkingClose(t *testing.T) {
	for _, tc := range []struct {
		name  string
		think *api.ThinkValue
		want  []string
	}{
		{name: "default", want: []string{inklingModel + inklingText}},
		{name: "low", think: &api.ThinkValue{Value: "low"}, want: []string{inklingModel + inklingText}},
		{name: "false", think: &api.ThinkValue{Value: false}},
		{name: "none", think: &api.ThinkValue{Value: "none"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := ParserForName("inkling2")
			p.Init(nil, nil, tc.think)
			if got := p.ThinkingClose(); !slices.Equal(got, tc.want) {
				t.Fatalf("ThinkingClose() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestInkling2FramedJSONPreservesMarkerText(t *testing.T) {
	const answer = `{"text":"<|end_message|><|content_thinking|>"}`
	response := inklingModel + inklingThought + "checking" + inklingEnd + inklingModel + inklingText + answer
	for split := range len(response) + 1 {
		p := inkling2Parser{}
		p.StructuredOutputFrames(inklingEnd)
		first, firstThought, firstCalls, err := p.Add(response[:split], false)
		if err != nil || len(firstCalls) != 0 {
			t.Fatalf("split %d first chunk: %v, %v", split, err, firstCalls)
		}
		last, lastThought, lastCalls, err := p.Add(response[split:], true)
		if err != nil || len(lastCalls) != 0 || first+last != answer || firstThought+lastThought != "checking" {
			t.Fatalf("split %d: content %q, thinking %q, calls %v, err %v", split, first+last, firstThought+lastThought, lastCalls, err)
		}
	}
}

func TestInkling2ToolIntegerPrecision(t *testing.T) {
	for _, tc := range []struct {
		raw  string
		want any
	}{
		{"9007199254740991", float64(1<<53 - 1)},
		{"-9007199254740991", -float64(1<<53 - 1)},
		{"9007199254740992", json.Number("9007199254740992")},
		{"-9007199254740992", json.Number("-9007199254740992")},
		{"9007199254740993", json.Number("9007199254740993")},
		{"-9007199254740993", json.Number("-9007199254740993")},
		{"1e20", float64(1e20)},
	} {
		var args inkling2ToolArguments
		if err := json.Unmarshal([]byte(`{"id":`+tc.raw+`}`), &args); err != nil {
			t.Fatal(err)
		}
		if got, _ := args.Get("id"); got != tc.want {
			t.Errorf("%s decoded as %v (%T), want %v (%T)", tc.raw, got, got, tc.want, tc.want)
		}
	}
}

func TestInkling2ParserChunkBoundaries(t *testing.T) {
	response := inklingModel + inklingThought + "check the source 🦉" + inklingEnd +
		inklingModel + "lookup" + inklingTool + `{"name":"lookup","args":{"id":17,"parts":[true,null,"café",{"account":9007199254740993,"group":18446744073709551615}]}}` + inklingEnd +
		inklingModel + "save" + inklingTool + `{"name":"save","args":{}}` + inklingEnd +
		inklingModel + inklingText + "The answer is 17." + inklingEnd + inklingEndSample
	for split := range len(response) + 1 {
		p := ParserForName("inkling2")
		p.Init(nil, nil, nil)
		first, firstThought, firstCalls, err := p.Add(response[:split], false)
		if err != nil {
			t.Fatalf("split %d: %v", split, err)
		}
		last, lastThought, lastCalls, err := p.Add(response[split:], true)
		if err != nil {
			t.Fatalf("split %d: %v", split, err)
		}
		if first+last != "The answer is 17." || firstThought+lastThought != "check the source 🦉" {
			t.Errorf("split %d: content %q, thinking %q", split, first+last, firstThought+lastThought)
		}
		calls := append(firstCalls, lastCalls...)
		want := map[string]any{"id": float64(17), "parts": []any{true, nil, "café", map[string]any{"account": json.Number("9007199254740993"), "group": json.Number("18446744073709551615")}}}
		if len(calls) != 2 || calls[0].Function.Name != "lookup" || !reflect.DeepEqual(calls[0].Function.Arguments.ToMap(), want) ||
			calls[1].Function.Name != "save" || calls[1].Function.Arguments.Len() != 0 {
			t.Errorf("split %d: tool calls %v", split, calls)
		}
	}
}

func TestInkling2TruncatedToolCall(t *testing.T) {
	const complete = inklingModel + "save" + inklingTool + `{"name":"save","args":{}}` + inklingEnd
	const header = inklingModel + "lookup" + inklingTool
	const payload = `{"name":"lookup","args":{"text":"before <|end_message|> after","id":[17]}}`
	for end := range len(payload) + 1 {
		p := inkling2Parser{}
		_, _, calls, err := p.Add(complete+header+payload[:end], false)
		if err != nil || len(calls) != 1 || calls[0].Function.Name != "save" {
			t.Fatalf("offset %d before done: calls %v, error %v", end, calls, err)
		}
		content, thinking, calls, err := p.Add("", true)
		if err != nil || content != "" || thinking != "" {
			t.Fatalf("offset %d on done: content %q, thinking %q, error %v", end, content, thinking, err)
		}
		if end == len(payload) {
			if len(calls) != 1 || calls[0].Function.Name != "lookup" {
				t.Fatalf("complete call: %v", calls)
			}
		} else if len(calls) != 0 {
			t.Fatalf("offset %d emitted a truncated call: %v", end, calls)
		}
	}
	for _, body := range []string{`{"name":!}`, `{"name":!`, `{"name":` + inklingEnd, `{}`, `tru`, `["`} {
		p := inkling2Parser{}
		if _, _, _, err := p.Add(header+body, true); err == nil {
			t.Errorf("expected error for malformed call %q", body)
		}
	}
}

func TestInkling2ParserJinjaToolCalls(t *testing.T) {
	// The Jinja template leaves literal control spellings in JSON strings and
	// accepts an empty string as a function name.
	const value = `before \"quoted\" <|end_message|> <|content_model_end_sampling|> after`
	response := inklingModel + "lookup" + inklingTool + `{"name":"lookup","args":{"text":"` + value + `"}}` + inklingEnd +
		inklingModel + inklingTool + `{"name":"","args":{}}` + inklingEnd +
		inklingModel + inklingText + "Done." + inklingEnd + inklingEndSample
	for split := range len(response) + 1 {
		p := inkling2Parser{}
		var content string
		var calls []api.ToolCall
		for i, chunk := range []string{response[:split], response[split:]} {
			text, thinking, parsed, err := p.Add(chunk, i == 1)
			if err != nil || thinking != "" {
				t.Fatalf("split %d chunk %d: thinking %q, error %v", split, i, thinking, err)
			}
			content += text
			calls = append(calls, parsed...)
		}
		if content != "Done." || len(calls) != 2 {
			t.Fatalf("split %d: content %q, tool calls %v", split, content, calls)
		}
		text, ok := calls[0].Function.Arguments.Get("text")
		if !ok || text != `before "quoted" <|end_message|> <|content_model_end_sampling|> after` ||
			calls[0].Function.Name != "lookup" || calls[1].Function.Name != "" || calls[1].Function.Arguments.Len() != 0 {
			t.Fatalf("split %d: tool calls %v", split, calls)
		}
	}
}

func TestInkling2SuppressesThinkingWhenDisabled(t *testing.T) {
	response := inklingModel + inklingThought + "checking" + inklingEnd +
		inklingModel + "lookup" + inklingTool + `{"name":"lookup","args":{}}` + inklingEnd +
		inklingModel + inklingText + "Tokyo" + inklingEnd + inklingEndSample
	for _, level := range []any{false, "none"} {
		for split := range len(response) + 1 {
			p := inkling2Parser{}
			p.Init(nil, nil, &api.ThinkValue{Value: level})
			first, firstThought, firstCalls, err := p.Add(response[:split], false)
			if err != nil {
				t.Fatalf("think %v split %d: %v", level, split, err)
			}
			last, lastThought, lastCalls, err := p.Add(response[split:], true)
			if err != nil || first+last != "Tokyo" || firstThought != "" || lastThought != "" {
				t.Fatalf("think %v split %d: content %q, thinking %q, error %v", level, split, first+last, firstThought+lastThought, err)
			}
			calls := append(firstCalls, lastCalls...)
			if len(calls) != 1 || calls[0].Function.Name != "lookup" {
				t.Fatalf("think %v split %d: tool calls %v", level, split, calls)
			}
		}
	}
}

func TestInkling2ToolTextAndErrorFrames(t *testing.T) {
	const literal = "quote " + inklingToolError + " and " + inklingToolText + " here"
	for _, tc := range []struct{ name, input, want string }{
		{"headerless", inklingToolText + "lookup x=1" + inklingEnd, "lookup x=1"},
		{"named", inklingModel + "lookup" + inklingToolText + "lookup x=1" + inklingEnd, "lookup x=1"},
		{"error", inklingModel + inklingToolError + "failed" + inklingEnd + inklingToolText + "retry", "retry"},
		{"thought", inklingModel + inklingThought + literal + inklingEnd + inklingModel + inklingText + "answer", "answer"},
		{"leading whitespace", "\n" + inklingModel + inklingThought + literal + inklingEnd + inklingToolText + "answer", "\nanswer"},
		{"unframed", literal, literal},
		{"text", inklingModel + inklingText + literal + inklingEnd, literal},
		{"tool text", inklingToolText + literal + inklingEnd, literal},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, tools := range [][]api.Tool{nil, {{Type: "function", Function: api.ToolFunction{Name: "lookup"}}}} {
				for split := range len(tc.input) + 1 {
					p := ParserForName("inkling2")
					p.Init(tools, nil, &api.ThinkValue{Value: false})
					var text string
					for i, chunk := range []string{tc.input[:split], tc.input[split:]} {
						body, thinking, calls, err := p.Add(chunk, i == 1)
						if err != nil || thinking != "" || len(calls) != 0 {
							t.Fatalf("split %d chunk %d: thinking %q, calls %v, error %v", split, i, thinking, calls, err)
						}
						text += body
					}
					if text != tc.want {
						t.Fatalf("split %d: content %q, want %q", split, text, tc.want)
					}
				}
			}
		})
	}
}

func FuzzInkling2Parser(f *testing.F) {
	f.Add(inklingModel+inklingThought+"checking"+inklingEnd+inklingModel+inklingText+"Tokyo"+inklingEnd, uint16(7), false)
	f.Add(inklingModel+"lookup"+inklingTool+`{"name":"lookup","args":{"x":"<|end_message|>"}}`+inklingEnd, uint16(23), false)
	f.Add(inklingModel+inklingTool+`{"name":`, uint16(9), false)
	f.Add(inklingModel+inklingText+`{"text":"<|end_message|>"}`, uint16(37), true)
	f.Add("0"+inklingEnd, uint16(3), true)
	f.Fuzz(func(t *testing.T, input string, offset uint16, structured bool) {
		if len(input) > 4096 {
			t.Skip("input exceeds the fuzz case budget")
		}
		newParser := func() *inkling2Parser {
			p := &inkling2Parser{}
			if structured {
				p.StructuredOutputFrames(inklingEnd)
			}
			return p
		}
		wantContent, wantThinking, wantCalls, wantErr := newParser().Add(input, true)
		p := newParser()
		split := int(offset) % (len(input) + 1)
		content, thinking, calls, err := p.Add(input[:split], false)
		if err == nil {
			var tail, tailThinking string
			var tailCalls []api.ToolCall
			tail, tailThinking, tailCalls, err = p.Add(input[split:], true)
			content += tail
			thinking += tailThinking
			calls = append(calls, tailCalls...)
		}
		if (err == nil) != (wantErr == nil) || content != wantContent || thinking != wantThinking || !reflect.DeepEqual(calls, wantCalls) {
			t.Fatalf("split at %d changed parse: content %q, thinking %q, calls %v, error %v; want %q, %q, %v, %v", split, content, thinking, calls, err, wantContent, wantThinking, wantCalls, wantErr)
		}
	})
}
