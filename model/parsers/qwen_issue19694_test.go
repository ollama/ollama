// Regression tests for ollama/ollama issue #19694:
// qwen3.5 / qwen3-coder parsers corrupt tool-call arguments or lose calls
// depending on content and stream chunking.
package parsers

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func issue19694Tools() []api.Tool {
	return []api.Tool{
		{
			Function: api.ToolFunction{
				Name: "write_file",
				Parameters: api.ToolFunctionParameters{
					Properties: func() *api.ToolPropertiesMap {
						props := api.NewToolPropertiesMap()
						props.Set("content", api.ToolProperty{Type: api.PropertyType{"string"}})
						return props
					}(),
				},
			},
		},
		{
			Function: api.ToolFunction{
				Name: "get_weather",
				Parameters: api.ToolFunctionParameters{
					Properties: func() *api.ToolPropertiesMap {
						props := api.NewToolPropertiesMap()
						props.Set("location", api.ToolProperty{Type: api.PropertyType{"string"}})
						return props
					}(),
				},
			},
		},
	}
}

// feedInChunks feeds s to the parser in n-byte chunks, marking only the
// last chunk done, and returns the accumulated content, thinking and calls.
func feedInChunks(t *testing.T, parser Parser, s string, n int) (string, string, []api.ToolCall) {
	t.Helper()
	var contentSb, thinkingSb strings.Builder
	var calls []api.ToolCall
	for i := 0; i < len(s); i += n {
		end := min(i+n, len(s))
		done := end == len(s)
		c, th, cs, err := parser.Add(s[i:end], done)
		if err != nil {
			t.Fatalf("chunk %d-%d: %v", i, end, err)
		}
		contentSb.WriteString(c)
		thinkingSb.WriteString(th)
		calls = append(calls, cs...)
	}
	return contentSb.String(), thinkingSb.String(), calls
}

// Case 1: `<Name=value>` inside a parameter value must not be rewritten.
func TestQwen3CoderParserTagLikeTextInValueNotRewritten(t *testing.T) {
	for _, value := range []string{
		"type Box<T=any>",
		"template <typename T=int>",
		"const el = <div className={x} data-value={a=b} />",
	} {
		parser := ParserForName("qwen3-coder")
		if parser == nil {
			t.Fatal("expected qwen3-coder parser")
		}
		parser.Init(issue19694Tools(), nil, nil)

		input := "<tool_call><function=write_file><parameter=content>\n" + value + "\n</parameter></function></tool_call>"
		_, _, calls, err := parser.Add(input, true)
		if err != nil {
			t.Fatalf("value %q: parse failed: %v", value, err)
		}
		if len(calls) != 1 {
			t.Fatalf("value %q: expected 1 tool call, got %d", value, len(calls))
		}
		content, ok := calls[0].Function.Arguments.Get("content")
		if !ok || content != value {
			t.Fatalf("value %q: expected content %q, got %v", value, value, content)
		}
	}
}

// Case 2: a literal `</tool_call>` inside a parameter value must not end the call.
func TestQwen3CoderParserLiteralToolCloseInValue(t *testing.T) {
	parser := ParserForName("qwen3-coder")
	if parser == nil {
		t.Fatal("expected qwen3-coder parser")
	}
	parser.Init(issue19694Tools(), nil, nil)

	input := "<tool_call><function=write_file><parameter=content>\nparser for </tool_call> format\n</parameter></function></tool_call>"
	_, _, calls, err := parser.Add(input, true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(calls))
	}
	content, ok := calls[0].Function.Arguments.Get("content")
	if !ok || content != "parser for </tool_call> format" {
		t.Fatalf("expected content %q, got %v", "parser for </tool_call> format", content)
	}
}

// Case 2, streamed: the literal close tag arrives in an early chunk, before
// the real close tag. The call must still be returned whole.
func TestQwen3CoderParserLiteralToolCloseInValueChunked(t *testing.T) {
	parser := ParserForName("qwen3-coder")
	if parser == nil {
		t.Fatal("expected qwen3-coder parser")
	}
	parser.Init(issue19694Tools(), nil, nil)

	input := "<tool_call><function=write_file><parameter=content>\nparser for </tool_call> format\n</parameter></function></tool_call>"
	_, _, calls := feedInChunks(t, parser, input, 7)
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(calls))
	}
	content, ok := calls[0].Function.Arguments.Get("content")
	if !ok || content != "parser for </tool_call> format" {
		t.Fatalf("expected content %q, got %v", "parser for </tool_call> format", content)
	}
}

// A malformed call (close tag present, but the function is never closed)
// must still produce today's XML parsing error at end of stream, not
// silently become ordinary content.
func TestQwen3CoderParserMalformedCallStillErrors(t *testing.T) {
	parser := ParserForName("qwen3-coder")
	if parser == nil {
		t.Fatal("expected qwen3-coder parser")
	}
	parser.Init(issue19694Tools(), nil, nil)

	input := "<tool_call><function=get_weather><parameter=location>\nSF\n</parameter></tool_call>"
	_, _, _, err := parser.Add(input, true)
	if err == nil {
		t.Fatalf("expected a parse error for a malformed call, got none")
	}
}

// Case 3: a literal `<tool_call>` inside thinking must stay thinking
// regardless of chunking.
func TestQwen35ParserLiteralToolCallInThinkingSurvivesChunking(t *testing.T) {
	input := "Thinking about <tool_call> format here</think>Answer."
	sizes := []int{len(input)}
	for size := 1; size <= 16; size++ {
		sizes = append(sizes, size)
	}
	for _, size := range sizes {
		t.Run("", func(t *testing.T) {
			parser := ParserForName("qwen3.5")
			if parser == nil {
				t.Fatal("expected qwen3.5 parser")
			}
			parser.Init(issue19694Tools(), nil, &api.ThinkValue{Value: true})
			content, thinking, calls := feedInChunks(t, parser, input, size)
			if thinking != "Thinking about <tool_call> format here" {
				t.Fatalf("chunk size %d: expected thinking %q, got %q", size, "Thinking about <tool_call> format here", thinking)
			}
			if content != "Answer." {
				t.Fatalf("chunk size %d: expected content %q, got %q", size, "Answer.", content)
			}
			if len(calls) != 0 {
				t.Fatalf("chunk size %d: expected no tool calls, got %d", size, len(calls))
			}
		})
	}
}

// Case 4: a real tool call without `</think>` must be returned even when chunked.
func TestQwen35ParserToolCallWithoutThinkCloseSurvivesChunking(t *testing.T) {
	input := "Need weather lookup<tool_call><function=get_weather><parameter=location>\nSF\n</parameter></function></tool_call>"
	// Size 1 fails on current main (the whole call is emitted as thinking);
	// every size 1..16 plus the single-chunk delivery must agree afterwards.
	sizes := []int{len(input)}
	for size := 1; size <= 16; size++ {
		sizes = append(sizes, size)
	}
	for _, size := range sizes {
		t.Run("", func(t *testing.T) {
			parser := ParserForName("qwen3.5")
			if parser == nil {
				t.Fatal("expected qwen3.5 parser")
			}
			parser.Init(issue19694Tools(), nil, &api.ThinkValue{Value: true})
			content, thinking, calls := feedInChunks(t, parser, input, size)
			if thinking != "Need weather lookup" {
				t.Fatalf("chunk size %d: expected thinking %q, got %q", size, "Need weather lookup", thinking)
			}
			if content != "" {
				t.Fatalf("chunk size %d: expected empty content, got %q", size, content)
			}
			if len(calls) != 1 {
				t.Fatalf("chunk size %d: expected 1 tool call, got %d", size, len(calls))
			}
			location, ok := calls[0].Function.Arguments.Get("location")
			if !ok || location != "SF" {
				t.Fatalf("chunk size %d: expected location %q, got %v", size, "SF", location)
			}
		})
	}
}

// Case 5: an empty final chunk must still flush a held tool call to the
// tool parser (here: a truncated call becomes content, not silence).
func TestQwen35ParserEmptyFinalChunkFlushesToolParser(t *testing.T) {
	parser := ParserForName("qwen3.5")
	if parser == nil {
		t.Fatal("expected qwen3.5 parser")
	}
	parser.Init(issue19694Tools(), nil, &api.ThinkValue{Value: true})

	input := "</think><tool_call><function=get_weather><parameter=location>\nSF"
	content, thinking, calls, err := parser.Add(input, false)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if thinking != "" || len(calls) != 0 {
		t.Fatalf("expected nothing emitted before the final chunk, got thinking %q and %d calls", thinking, len(calls))
	}
	content2, _, calls2, err := parser.Add("", true)
	if err != nil {
		t.Fatalf("parse failed on final chunk: %v", err)
	}
	content += content2
	calls = append(calls, calls2...)
	if len(calls) != 0 {
		t.Fatalf("expected no tool call for a truncated call, got %d", len(calls))
	}
	if content != "<tool_call><function=get_weather><parameter=location>\nSF" {
		t.Fatalf("expected truncated call flushed as content %q, got %q", "<tool_call><function=get_weather><parameter=location>\nSF", content)
	}
}

// Boundary (issue #19694 rule, condition 3): a call-shaped blob followed by
// ordinary prose and then </think> is NOT an implicit tool call — after the
// call, the stream must end or another call must start. The whole blob stays
// literal thinking text.
func TestQwen35ParserCallFollowedByProseIsLiteralThinking(t *testing.T) {
	parser := ParserForName("qwen3.5")
	if parser == nil {
		t.Fatal("expected qwen3.5 parser")
	}
	parser.Init(issue19694Tools(), nil, nil)

	input := "Thinking <tool_call><function=get_weather><parameter=location>\nSF\n</parameter></function></tool_call> some prose</think>Answer."
	content, thinking, calls, err := parser.Add(input, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(calls) != 0 {
		t.Fatalf("expected 0 tool calls, got %d", len(calls))
	}
	wantThinking := "Thinking <tool_call><function=get_weather><parameter=location>\nSF\n</parameter></function></tool_call> some prose"
	if thinking != wantThinking {
		t.Fatalf("expected thinking %q, got %q", wantThinking, thinking)
	}
	if content != "Answer." {
		t.Fatalf("expected content %q, got %q", "Answer.", content)
	}
}

// Case 5, extra: a complete call followed by an empty final chunk must still
// produce the call (and not turn it into content), and ordinary content
// followed by an empty final chunk must be intact.
func TestQwen35ParserEmptyFinalChunkKeepsCompletedCall(t *testing.T) {
	parser := ParserForName("qwen3.5")
	if parser == nil {
		t.Fatal("expected qwen3.5 parser")
	}
	parser.Init(issue19694Tools(), nil, nil)

	_, _, calls, err := parser.Add("</think><tool_call><function=get_weather><parameter=location>\nSF\n</parameter></function></tool_call>", false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call before the final chunk, got %d", len(calls))
	}
	content, _, calls, err := parser.Add("", true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(calls) != 0 {
		t.Fatalf("expected 0 additional tool calls from the empty final chunk, got %d", len(calls))
	}
	if content != "" {
		t.Fatalf("expected no content from the empty final chunk, got %q", content)
	}
}

func TestQwen35ParserEmptyFinalChunkKeepsOrdinaryContent(t *testing.T) {
	parser := ParserForName("qwen3.5")
	if parser == nil {
		t.Fatal("expected qwen3.5 parser")
	}
	parser.Init(issue19694Tools(), nil, nil)

	content, _, _, err := parser.Add("</think>Just an answer.", false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if content != "Just an answer." {
		t.Fatalf("expected content %q, got %q", "Just an answer.", content)
	}
	content, _, _, err = parser.Add("", true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if content != "" {
		t.Fatalf("expected no content from the empty final chunk, got %q", content)
	}
}
