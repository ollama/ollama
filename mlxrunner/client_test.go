package mlxrunner

import (
	"encoding/json"
	"errors"
	"io/fs"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/xgrammar"
	"github.com/ollama/ollama/model/parsers"
	"github.com/ollama/ollama/types/model"
)

func TestRequestGrammar(t *testing.T) {
	schema := `{"type":"object","properties":{"answer":{"type":"string"}}}`
	tag := `{"type":"structural_tag","format":{"type":"json_schema","json_schema":` + schema + `}}`
	for _, tt := range []struct {
		name string
		req  llm.CompletionRequest
		want string
	}{
		{name: "unset"},
		{name: "null", req: llm.CompletionRequest{Format: json.RawMessage(`null`)}},
		{name: "empty", req: llm.CompletionRequest{Format: json.RawMessage(`""`)}},
		{
			name: "json",
			req:  llm.CompletionRequest{Format: json.RawMessage(`"json"`)},
			want: `{"type":"structural_tag","format":{"type":"json_schema","json_schema":{"type":"object"}}}`,
		},
		{name: "schema", req: llm.CompletionRequest{Format: json.RawMessage(schema)}, want: tag},
		{
			name: "schema after thinking",
			req:  llm.CompletionRequest{Format: json.RawMessage(schema), ThinkingClose: []string{"</think>"}},
			want: `{"type":"structural_tag","format":{"type":"sequence","elements":[{"type":"any_text","excludes":["</think>"]},` +
				`{"type":"optional","content":{"type":"sequence","elements":[{"type":"const_string","value":"</think>"},` +
				`{"type":"json_schema","json_schema":` + schema + `}]}}]}}`,
		},
		{
			name: "schema after thinking with two closings",
			req:  llm.CompletionRequest{Format: json.RawMessage(schema), ThinkingClose: []string{"<|final|>", "<|final|>json"}},
			want: `{"type":"structural_tag","format":{"type":"sequence","elements":[{"type":"any_text","excludes":["<|final|>","<|final|>json"]},` +
				`{"type":"optional","content":{"type":"sequence","elements":[{"type":"or","elements":[{"type":"const_string","value":"<|final|>"},{"type":"const_string","value":"<|final|>json"}]},` +
				`{"type":"json_schema","json_schema":` + schema + `}]}}]}}`,
		},
		{name: "thinking without a format", req: llm.CompletionRequest{ThinkingClose: []string{"</think>"}}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := string(requestGrammar(tt.req)); got != tt.want {
				t.Fatalf("requestGrammar = %s, want %s", got, tt.want)
			}
		})
	}
}

func TestInkling2StructuredGrammarFrames(t *testing.T) {
	const (
		thoughtOpen int32 = iota
		thoughtClose
		contentOpen
		plain
		jsonBody
		eos
	)
	pieces := []string{"<|message_model|><|content_thinking|>", "<|end_message|>", "<|message_model|><|content_text|>", "plain text", `{"x":"<|end_message|>"}`, "<eos>"}
	path, err := mlx.LoadedLibraryPath()
	if err != nil {
		t.Skipf("native MLX payload is unavailable: %v", err)
	}
	compiler, err := xgrammar.New(filepath.Dir(path), pieces, len(pieces), []int32{eos}, 1, 0)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("native xgrammar payload is unavailable: %v", err)
	}
	if err != nil {
		t.Fatal(err)
	}
	defer compiler.Close()
	for _, tc := range []struct {
		name     string
		prompt   string
		suppress bool
		accepted []int32
		rejected int32
	}{
		{prompt: pieces[thoughtClose], name: "plain text before a frame", rejected: plain},
		{prompt: pieces[thoughtClose], name: "plain text after thinking", accepted: []int32{thoughtOpen, plain, thoughtClose}, rejected: plain},
		{prompt: pieces[thoughtClose], name: "stop before an answer", accepted: []int32{thoughtOpen, plain, thoughtClose}, rejected: eos},
		{prompt: pieces[thoughtClose], name: "plain text inside the answer frame", accepted: []int32{contentOpen}, rejected: plain},
		{prompt: pieces[thoughtClose], name: "answer after thinking", accepted: []int32{thoughtOpen, plain, thoughtClose, contentOpen, jsonBody, eos}, rejected: -1},
		{prompt: pieces[thoughtClose], name: "answer without thinking", accepted: []int32{contentOpen, jsonBody, eos}, rejected: -1},
		{name: "continue prefilled thought", prompt: pieces[thoughtOpen], accepted: []int32{plain, thoughtClose, contentOpen, jsonBody, eos}, rejected: -1},
		{name: "close prefilled thought even when hidden", prompt: pieces[thoughtOpen], suppress: true, accepted: []int32{plain, thoughtClose, contentOpen, jsonBody, eos}, rejected: -1},
		{name: "early stop in prefilled thought", prompt: pieces[thoughtOpen], accepted: []int32{plain}, rejected: eos},
		{name: "prefilled answer", prompt: pieces[contentOpen], accepted: []int32{jsonBody, eos}, rejected: -1},
		{name: "disabled thought", prompt: pieces[thoughtClose], suppress: true, rejected: thoughtOpen},
		{name: "unframed answer with thinking disabled", prompt: pieces[thoughtClose], suppress: true, rejected: jsonBody},
		{name: "framed answer with thinking disabled", prompt: pieces[thoughtClose], suppress: true, accepted: []int32{contentOpen, jsonBody, eos}, rejected: -1},
	} {
		p := parsers.ParserForName("inkling2")
		p.Init(nil, nil, &api.ThinkValue{Value: !tc.suppress})
		framed := p.(interface {
			StructuredOutputFrames(string) *model.StructuredOutputFrames
		})
		frames := framed.StructuredOutputFrames(tc.prompt)
		format := requestGrammar(llm.CompletionRequest{Format: json.RawMessage(`"json"`), FormatFrames: frames, ThinkingClose: p.ThinkingClose()})
		matcher, err := compiler.Compile(string(format))
		if err != nil {
			t.Fatal(err)
		}
		for _, token := range tc.accepted {
			if err := matcher.Accept(token); err != nil {
				t.Errorf("%s rejected token %q: %v", tc.name, pieces[token], err)
				break
			}
		}
		if tc.rejected >= 0 {
			if err := matcher.Accept(tc.rejected); err == nil {
				t.Errorf("%s accepted token %q", tc.name, pieces[tc.rejected])
			}
		} else if !matcher.Terminated() {
			t.Errorf("%s did not reach the stop token", tc.name)
		}
		matcher.Close()
	}
}
