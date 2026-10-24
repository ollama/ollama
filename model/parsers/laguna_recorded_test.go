package parsers

import (
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

// TestLagunaRecordedMCPReply replays the exact token stream recorded from a
// failing /api/chat request, one token per Add call, the way the server feeds
// the parser. The live request declared web_search and web_fetch, streamed with
// thinking enabled, and died with "empty Laguna tool call name" partway through
// a JSON configuration snippet, losing the response. No model or server is
// involved here.
func TestLagunaRecordedMCPReply(t *testing.T) {
	raw, err := os.ReadFile("testdata/laguna_mcp_tokens.json")
	if err != nil {
		t.Fatalf("read recorded tokens: %v", err)
	}
	var tokens []string
	if err := json.Unmarshal(raw, &tokens); err != nil {
		t.Fatalf("decode recorded tokens: %v", err)
	}

	tools := []api.Tool{
		{Function: api.ToolFunction{Name: "web_search"}},
		{Function: api.ToolFunction{Name: "web_fetch"}},
	}

	p := &LagunaV8Parser{}
	p.Init(tools, nil, &api.ThinkValue{Value: true})

	var content string
	var calls []api.ToolCall
	var full int
	for i, tok := range tokens {
		full += len(tok)
		c, _, cl, e := p.Add(tok, i == len(tokens)-1)
		content += c
		calls = append(calls, cl...)
		if e != nil {
			t.Fatalf("parser destroyed the reply at token %d/%d: %v (delivered %d of %d bytes)",
				i+1, len(tokens), e, len(content), full)
		}
	}

	t.Logf("no error: calls=%d content=%d/%d bytes", len(calls), len(content), full)
	if len(calls) > 0 {
		t.Errorf("reply turned into %d tool calls: %+v", len(calls), calls)
	}
}

// TestLagunaRecordedMCPReply2 replays the parser input stream captured (via
// OLLAMA_DEBUG=DEBUG-4 routes.go trace logging) from a request that failed
// live on 2026-08-06 with "empty Laguna tool call name". The reply is markdown
// containing a JSON config block; the env object's opening brace arrived as a
// single ` {"` token, so consumeStandaloneJSONTool latched onto it and errored
// when the object closed without a "name" key.
func TestLagunaRecordedMCPReply2(t *testing.T) {
	raw, err := os.ReadFile("testdata/laguna_mcp_tokens2.json")
	if err != nil {
		t.Fatalf("read recorded tokens: %v", err)
	}
	var tokens []string
	if err := json.Unmarshal(raw, &tokens); err != nil {
		t.Fatalf("decode recorded tokens: %v", err)
	}

	tools := []api.Tool{
		{Function: api.ToolFunction{Name: "web_search"}},
		{Function: api.ToolFunction{Name: "web_fetch"}},
	}

	p := &LagunaV8Parser{}
	p.Init(tools, nil, &api.ThinkValue{Value: true})

	var content string
	var calls []api.ToolCall
	for i, tok := range tokens {
		c, _, cl, e := p.Add(tok, false)
		content += c
		calls = append(calls, cl...)
		if e != nil {
			t.Fatalf("parser destroyed the reply at token %d/%d: %v", i+1, len(tokens), e)
		}
	}
	if len(calls) > 0 {
		t.Errorf("reply turned into %d tool calls: %+v", len(calls), calls)
	}
	// The recorded stream opens with the closing tag of the primed <think>, and
	// the parser drops that tag and the whitespace around it.
	want := strings.TrimLeft(strings.Join(tokens, ""), " \n")
	want = strings.TrimPrefix(want, "</think>")
	if content != want {
		t.Errorf("content altered\n got: %q\nwant: %q", content, want)
	}
}
