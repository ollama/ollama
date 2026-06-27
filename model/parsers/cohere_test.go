package parsers

import (
	"testing"

	"github.com/ollama/ollama/api"
)

// Model output begins inside <|START_THINKING|> when reasoning is on (the
// generation prompt ends with that tag).

func cohereAddAll(t *testing.T, p *CohereParser, chunks []string) (content, thinking string, calls []api.ToolCall) {
	t.Helper()
	for i, c := range chunks {
		done := i == len(chunks)-1
		ct, th, tc, err := p.Add(c, done)
		if err != nil {
			t.Fatal(err)
		}
		content += ct
		thinking += th
		calls = append(calls, tc...)
	}
	return content, thinking, calls
}

func TestCohereParseThinkingThenText(t *testing.T) {
	p := &CohereParser{}
	p.Init(nil, nil, nil)

	content, thinking, calls := cohereAddAll(t, p, []string{
		"Let me think", " about this.<|END_THINKING|>",
		"<|START_TEXT|>Hello", " world!<|END_TEXT|>",
	})
	if thinking != "Let me think about this." {
		t.Errorf("thinking = %q", thinking)
	}
	if content != "Hello world!" {
		t.Errorf("content = %q", content)
	}
	if len(calls) != 0 {
		t.Errorf("unexpected calls: %v", calls)
	}
}

func TestCohereParseSplitTags(t *testing.T) {
	p := &CohereParser{}
	p.Init(nil, nil, nil)

	// Tags split across chunk boundaries must not leak into output.
	content, thinking, _ := cohereAddAll(t, p, []string{
		"think<|END_TH", "INKING|><|STAR", "T_TEXT|>ans", "wer<|END_", "TEXT|>",
	})
	if thinking != "think" {
		t.Errorf("thinking = %q", thinking)
	}
	if content != "answer" {
		t.Errorf("content = %q", content)
	}
}

func TestCohereParseToolCall(t *testing.T) {
	p := &CohereParser{}
	p.Init(nil, nil, nil)

	content, thinking, calls := cohereAddAll(t, p, []string{
		"plan<|END_THINKING|><|START_ACTION|>[\n",
		`    {"tool_call_id": "0", "tool_name": "get_weather", "parameters": {"city": "Paris"}},`,
		`    {"tool_call_id": "1", "tool_name": "get_time", "parameters": {}}`,
		"\n]<|END_ACTION|>",
	})
	if thinking != "plan" {
		t.Errorf("thinking = %q", thinking)
	}
	if content != "" {
		t.Errorf("content = %q", content)
	}
	if len(calls) != 2 {
		t.Fatalf("calls = %d, want 2", len(calls))
	}
	if calls[0].Function.Name != "get_weather" || calls[1].Function.Name != "get_time" {
		t.Errorf("call names = %q, %q", calls[0].Function.Name, calls[1].Function.Name)
	}
	if v, ok := calls[0].Function.Arguments.Get("city"); !ok || v != "Paris" {
		t.Errorf("call 0 city = %v %v", v, ok)
	}
	if calls[0].Function.Index != 0 || calls[1].Function.Index != 1 {
		t.Errorf("call indices = %d, %d", calls[0].Function.Index, calls[1].Function.Index)
	}
}

func TestCohereParseReasoningOff(t *testing.T) {
	p := &CohereParser{}
	think := &api.ThinkValue{Value: false}
	p.Init(nil, nil, think)

	content, thinking, _ := cohereAddAll(t, p, []string{
		"<|START_TEXT|>direct answer<|END_TEXT|>",
	})
	if thinking != "" {
		t.Errorf("thinking = %q", thinking)
	}
	if content != "direct answer" {
		t.Errorf("content = %q", content)
	}
}

func TestCohereParseBareContent(t *testing.T) {
	// Models occasionally skip the START_TEXT wrapper; treat raw text after
	// thinking as content.
	p := &CohereParser{}
	p.Init(nil, nil, nil)

	content, thinking, _ := cohereAddAll(t, p, []string{
		"thought<|END_THINKING|>", "Just plain text", " output",
	})
	if thinking != "thought" {
		t.Errorf("thinking = %q", thinking)
	}
	if content != "Just plain text output" {
		t.Errorf("content = %q", content)
	}
}

func TestCohereParseEndOfTurnWithoutEndText(t *testing.T) {
	p := &CohereParser{}
	p.Init(nil, nil, nil)

	content, _, _ := cohereAddAll(t, p, []string{
		"t<|END_THINKING|><|START_TEXT|>answer<|END_OF_TURN_TOKEN|>",
	})
	if content != "answer" {
		t.Errorf("content = %q", content)
	}
}

func TestCohereParsePrefillContinuation(t *testing.T) {
	p := &CohereParser{}
	last := &api.Message{Role: "assistant", Content: "partial"}
	p.Init(nil, last, nil)

	content, thinking, _ := cohereAddAll(t, p, []string{" continued<|END_TEXT|>"})
	if thinking != "" {
		t.Errorf("thinking = %q", thinking)
	}
	if content != " continued" {
		t.Errorf("content = %q", content)
	}
}

func TestCohereParserRegistered(t *testing.T) {
	p := ParserForName("cohere")
	if p == nil {
		t.Fatal("cohere parser not registered")
	}
	if !p.HasToolSupport() || !p.HasThinkingSupport() {
		t.Error("cohere parser should support tools and thinking")
	}
}

func TestCohereParseMalformedActions(t *testing.T) {
	// One malformed call (unquoted value) must not drop its well-formed
	// sibling, and a missing comma between calls must not drop either.
	p := &CohereParser{}
	p.Init(nil, nil, nil)
	_, _, calls := cohereAddAll(t, p, []string{
		"plan<|END_THINKING|><|START_ACTION|>[\n",
		`    {"tool_call_id": "0", "tool_name": "set_alarm", "parameters": {"time": 15:30}},`,
		`    {"tool_call_id": "1", "tool_name": "get_weather", "parameters": {"city": "Oslo"}}`,
		"\n]<|END_ACTION|>",
	})
	if len(calls) != 1 || calls[0].Function.Name != "get_weather" {
		t.Fatalf("calls = %v, want the well-formed get_weather call", calls)
	}

	p = &CohereParser{}
	p.Init(nil, nil, nil)
	_, _, calls = cohereAddAll(t, p, []string{
		`<|END_THINKING|><|START_ACTION|>[`,
		`{"tool_call_id": "0", "tool_name": "a", "parameters": {}}`,
		`{"tool_call_id": "1", "tool_name": "b", "parameters": {"x": "{not json}"}}`,
		`]<|END_ACTION|>`,
	})
	if len(calls) != 2 || calls[0].Function.Name != "a" || calls[1].Function.Name != "b" {
		t.Fatalf("calls = %v, want both calls despite missing comma", calls)
	}
	if v, ok := calls[1].Function.Arguments.Get("x"); !ok || v != "{not json}" {
		t.Fatalf("braces inside string values must not split objects, got %v", v)
	}

	// Unparseable garbage yields no calls and no panic.
	p = &CohereParser{}
	p.Init(nil, nil, nil)
	_, _, calls = cohereAddAll(t, p, []string{"x<|END_THINKING|><|START_ACTION|>[!!!]<|END_ACTION|>"})
	if len(calls) != 0 {
		t.Fatalf("calls = %v, want none", calls)
	}
}

func TestCohereParseLegacyResponseMarkers(t *testing.T) {
	// Models trained on the older Command A template sometimes emit
	// <|START_RESPONSE|>/<|END_RESPONSE|> instead of START_TEXT/END_TEXT.
	p := &CohereParser{}
	p.Init(nil, nil, nil)
	content, thinking, _ := cohereAddAll(t, p, []string{
		"plan<|END_THINKING|>", "<|START_RESPONSE|>Hello", " there<|END_RESPONSE|>",
	})
	if thinking != "plan" {
		t.Errorf("thinking = %q", thinking)
	}
	if content != "Hello there" {
		t.Errorf("content = %q", content)
	}
}

func TestCohereParseStreamsBeforeDone(t *testing.T) {
	// Regression test: output after END_THINKING that opens with an
	// unrecognized tag must stream as it arrives, not buffer until the
	// generation finishes (it previously buffered forever, presenting as a
	// hung response).
	for _, tc := range []struct {
		name  string
		chunk string
	}{
		{"legacy response marker", "<|START_RESPONSE|>The answer is 42."},
		{"unrecognized tag", "<|TOOL_PLAN|>I should call the tool."},
		{"bare content", "Just plain text."},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := &CohereParser{}
			p.Init(nil, nil, nil)
			if _, _, _, err := p.Add("x<|END_THINKING|>", false); err != nil {
				t.Fatal(err)
			}
			content, _, _, err := p.Add(tc.chunk, false) // done=false: still streaming
			if err != nil {
				t.Fatal(err)
			}
			if content == "" {
				t.Fatalf("content did not stream before done for %q", tc.chunk)
			}
		})
	}
}

func TestCohereParseEndOfTurnBetweenBlocks(t *testing.T) {
	// A literal end-of-turn marker between blocks is consumed, not shown.
	p := &CohereParser{}
	p.Init(nil, nil, nil)
	content, thinking, _ := cohereAddAll(t, p, []string{
		"t<|END_THINKING|><|START_TEXT|>hi<|END_TEXT|><|END_OF_TURN_TOKEN|>",
	})
	if thinking != "t" || content != "hi" {
		t.Errorf("thinking = %q, content = %q", thinking, content)
	}
}

func TestCohereParseBareContentBeforeEndOfTurn(t *testing.T) {
	// Bare content followed by an end-of-turn marker keeps the content.
	p := &CohereParser{}
	p.Init(nil, nil, nil)
	content, _, _ := cohereAddAll(t, p, []string{
		"t<|END_THINKING|>plain answer<|END_OF_TURN_TOKEN|>",
	})
	if content != "plain answer" {
		t.Errorf("content = %q, want %q", content, "plain answer")
	}
}
