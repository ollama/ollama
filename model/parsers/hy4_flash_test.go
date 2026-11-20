package parsers

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestHy4FlashParserStreaming(t *testing.T) {
	reasoning, answer := "查一下。", "Done.\n"
	arg := "<arg_key:opensource>count</arg_key:opensource><arg_value:opensource>3</arg_value:opensource>"
	text := reasoning + "</think:opensource><tool_calls:opensource>" +
		"<tool_call:opensource>search" + arg + "</tool_call:opensource>" +
		"<tool_call:opensource>ping</tool_call:opensource></tool_calls:opensource>" + answer
	for split := range len(text) + 1 {
		p := &hy4FlashParser{}
		p.Init(nil, nil, nil)
		left, firstThink, first, err := p.Add(text[:split], false)
		if err != nil {
			t.Fatalf("split %d: %v", split, err)
		}
		right, nextThink, next, err := p.Add(text[split:], true)
		if err != nil {
			t.Fatalf("split %d: %v", split, err)
		}
		if got := left + right; got != answer {
			t.Fatalf("split %d content %q", split, got)
		}
		if got := firstThink + nextThink; got != reasoning {
			t.Fatalf("split %d thinking %q", split, got)
		}
		calls := append(first, next...)
		if len(calls) != 2 || calls[0].Function.Name != "search" || calls[1].Function.Name != "ping" || calls[1].Function.Index != 1 {
			t.Fatalf("split %d calls %+v", split, calls)
		}
	}
}

func TestHy4FlashParserDirectAndIncomplete(t *testing.T) {
	p := &hy4FlashParser{}
	p.Init(nil, nil, &api.ThinkValue{Value: false})
	input := "visible <tool_calls:opensource><tool_call:opensource>partial"
	text, thought, calls, err := p.Add(input, true)
	if err != nil || thought != "" || len(calls) != 0 || strings.TrimSpace(text) != input {
		t.Fatalf("content %q, thinking %q, calls %+v, error %v", text, thought, calls, err)
	}
}
