package parsers

import (
	"encoding/json"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestDeepSeekV41ParserEverySplit(t *testing.T) {
	const output = "Reasoning</think>Checking\n\n<｜DSML｜ calls>\n" +
		"<｜DSML｜ invoke name=\"calendar::find\">\n" +
		"<｜DSML｜ parameter name=\"date\" string=\"true\">tomorrow</｜DSML｜ parameter>\n" +
		"<｜DSML｜ parameter name=\"limit\" string=\"false\">2</｜DSML｜ parameter>\n" +
		"</｜DSML｜ invoke>\n" +
		"<｜DSML｜ invoke name=\"calendar::find\">\n" +
		"<｜DSML｜ parameter name=\"filters\" string=\"false\">{\"private\": false}</｜DSML｜ parameter>\n" +
		"</｜DSML｜ invoke>\n</｜DSML｜ calls>"
	for split := range len(output) + 1 {
		p := &DeepSeekV41Parser{}
		p.Init(nil, nil, nil)
		if got := p.ThinkingClose(); len(got) != 1 || got[0] != "</think>" {
			t.Fatalf("split %d: thinking close = %v", split, got)
		}
		content, thinking, calls, err := p.Add(output[:split], false)
		if err != nil {
			t.Fatalf("split %d first chunk: %v", split, err)
		}
		moreContent, moreThinking, moreCalls, err := p.Add(output[split:], true)
		if err != nil {
			t.Fatalf("split %d final chunk: %v", split, err)
		}
		if got := content + moreContent; got != "Checking" {
			t.Fatalf("split %d: content = %q", split, got)
		}
		if got := thinking + moreThinking; got != "Reasoning" {
			t.Fatalf("split %d: thinking = %q", split, got)
		}
		calls = append(calls, moreCalls...)
		if len(calls) != 2 || calls[0].Function.Index != 0 || calls[1].Function.Index != 1 ||
			calls[0].Function.Name != "calendar::find" || calls[1].Function.Name != "calendar::find" {
			t.Fatalf("split %d: calls = %+v", split, calls)
		}
		if date, _ := calls[0].Function.Arguments.Get("date"); date != "tomorrow" {
			t.Fatalf("split %d: date = %v", split, date)
		}
		if limit, _ := calls[0].Function.Arguments.Get("limit"); limit != float64(2) {
			t.Fatalf("split %d: limit = %v", split, limit)
		}
		if filters, _ := calls[1].Function.Arguments.Get("filters"); !equalDeepSeek41JSON(filters, map[string]any{"private": false}) {
			t.Fatalf("split %d: filters = %v", split, filters)
		}
	}
}

func equalDeepSeek41JSON(a, b any) bool {
	x, errX := json.Marshal(a)
	y, errY := json.Marshal(b)
	return errX == nil && errY == nil && string(x) == string(y)
}

func TestDeepSeekV41ParserChatAndIncompleteCall(t *testing.T) {
	p := &DeepSeekV41Parser{}
	p.Init(nil, nil, &api.ThinkValue{Value: false})
	if len(p.ThinkingClose()) != 0 {
		t.Fatal("chat mode must start in content")
	}
	content, thinking, calls, err := p.Add("Hello", true)
	if err != nil || content != "Hello" || thinking != "" || len(calls) != 0 {
		t.Fatalf("chat response = %q, %q, %v, %v", content, thinking, calls, err)
	}
	p.Init(nil, nil, &api.ThinkValue{Value: false})
	if _, _, _, err := p.Add("<｜DSML｜ calls><｜DSML｜ invoke name=\"x\">", true); err == nil {
		t.Fatal("incomplete tool block accepted")
	}
}

func TestDeepSeekV41ParserByteByByte(t *testing.T) {
	const output = "thought</think>Answer\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"clock\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>"
	p := &DeepSeekV41Parser{}
	p.Init(nil, nil, nil)
	var content, thinking string
	var calls []api.ToolCall
	for i := range len(output) {
		chunkContent, chunkThinking, chunkCalls, err := p.Add(output[i:i+1], i == len(output)-1)
		if err != nil {
			t.Fatalf("byte %d: %v", i, err)
		}
		content += chunkContent
		thinking += chunkThinking
		calls = append(calls, chunkCalls...)
	}
	if content != "Answer" || thinking != "thought" || len(calls) != 1 || calls[0].Function.Name != "clock" {
		t.Fatalf("parsed = content %q, thinking %q, calls %+v", content, thinking, calls)
	}
}
