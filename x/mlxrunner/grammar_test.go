package mlxrunner

import (
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/xgrammar"
)

func TestParseGrammar(t *testing.T) {
	tests := []struct {
		name    string
		format  string
		kind    xgrammar.Kind
		source  string
		want    bool
		wantErr bool
	}{
		{name: "unset"},
		{name: "null", format: `null`},
		{name: "empty", format: `""`},
		{name: "json", format: `"json"`, kind: xgrammar.JSONSchema, source: `{"type":"object"}`, want: true},
		{name: "schema", format: `{"type":"integer"}`, kind: xgrammar.JSONSchema, source: `{"type":"integer"}`, want: true},
		{name: "unsupported string", format: `"xml"`, wantErr: true},
		{name: "whitespace JSON", format: ` "json" `, wantErr: true},
		{name: "whitespace schema", format: ` {"type":"integer"} `, wantErr: true},
		{name: "array", format: `[]`, wantErr: true},
		{name: "invalid json", format: `{`, wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseGrammar(json.RawMessage(tt.format))
			if (err != nil) != tt.wantErr {
				t.Fatalf("parseGrammar error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr {
				return
			}
			if (got != nil) != tt.want {
				t.Fatalf("parseGrammar = %#v, want present %v", got, tt.want)
			}
			if got != nil && (got.kind != tt.kind || got.source != tt.source) {
				t.Errorf("parseGrammar = %#v, want kind %v source %q", got, tt.kind, tt.source)
			}
		})
	}
}

func TestParseGrammarDoesNotEchoOversizedInput(t *testing.T) {
	format := json.RawMessage("{" + strings.Repeat("x", maxGrammarSchemaBytes))
	_, err := parseGrammar(format)
	if err == nil || !strings.Contains(err.Error(), "schema is 1048577 bytes") {
		t.Fatalf("parseGrammar error = %v, want bounded size error", err)
	}
	if len(err.Error()) > 256 {
		t.Fatalf("parseGrammar echoed oversized input in %d-byte error", len(err.Error()))
	}
}

func nestedGrammarSchema(depth int) []byte {
	return []byte(`{"value":` + strings.Repeat("[", depth-1) + `0` + strings.Repeat("]", depth-1) + `}`)
}

func grammarSchemaArray(entries int) []byte {
	var b strings.Builder
	b.WriteString(`{"enum":[`)
	for i := range entries {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteByte('0')
	}
	b.WriteString(`]}`)
	return []byte(b.String())
}

func TestValidateGrammarSchema(t *testing.T) {
	invalidUTF8 := append([]byte(`{"value":"`), 0xff)
	invalidUTF8 = append(invalidUTF8, []byte(`"}`)...)
	tests := []struct {
		name    string
		schema  []byte
		wantErr string
	}{
		{name: "object", schema: []byte(`{"type":"object","properties":{"answer":{"type":"string"}}}`)},
		{name: "maximum depth", schema: nestedGrammarSchema(maxGrammarSchemaDepth)},
		{name: "too deep", schema: nestedGrammarSchema(maxGrammarSchemaDepth + 1), wantErr: "nesting exceeds"},
		{name: "too many tokens", schema: grammarSchemaArray(maxGrammarSchemaTokens), wantErr: "more than 16384 JSON tokens"},
		{name: "too large", schema: []byte(`{"value":"` + strings.Repeat("x", maxGrammarSchemaBytes) + `"}`), wantErr: "limit is 1048576"},
		{name: "invalid UTF-8", schema: invalidUTF8, wantErr: "not valid UTF-8"},
		{name: "trailing value", schema: []byte(`{} {}`), wantErr: "more than one JSON value"},
		{name: "malformed", schema: []byte(`{"type":`), wantErr: "unexpected end"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateGrammarSchema(tt.schema)
			if tt.wantErr == "" {
				if err != nil {
					t.Fatal(err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("validateGrammarSchema error = %v, want containing %q", err, tt.wantErr)
			}
		})
	}
}

func BenchmarkValidateGrammarSchema(b *testing.B) {
	schema := []byte(`{"type":"object","properties":{"answer":{"type":"string","enum":["ok"]}},"required":["answer"],"additionalProperties":false}`)
	b.ReportAllocs()
	for b.Loop() {
		if err := validateGrammarSchema(schema); err != nil {
			b.Fatal(err)
		}
	}
}

func FuzzParseGrammar(f *testing.F) {
	for _, format := range []string{
		"",
		`"json"`,
		`{"type":"integer"}`,
		`{`,
	} {
		f.Add(format)
	}

	f.Fuzz(func(t *testing.T, format string) {
		spec, err := parseGrammar(json.RawMessage(format))
		if err != nil || spec == nil {
			return
		}
		switch spec.kind {
		case xgrammar.JSONSchema:
		default:
			t.Fatalf("parseGrammar kind = %d", spec.kind)
		}
	})
}

func TestValidateGrammarVocab(t *testing.T) {
	for _, tt := range []struct {
		name      string
		logits    int
		tokenizer int
		wantErr   bool
	}{
		{name: "exact fit", logits: 32, tokenizer: 32},
		{name: "padded model head", logits: 40, tokenizer: 32},
		{name: "input-only tokens past the head", logits: 31, tokenizer: 32},
		{name: "invalid tokenizer", logits: 32, tokenizer: -1, wantErr: true},
		{name: "invalid logits width", logits: 0, tokenizer: 32, wantErr: true},
		{name: "allocation bound", logits: maxGrammarVocabSize + 1, tokenizer: 32, wantErr: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			err := validateGrammarVocab(tt.logits, tt.tokenizer)
			if (err != nil) != tt.wantErr {
				t.Fatalf("validateGrammarVocab(%d, %d) error = %v, wantErr %v", tt.logits, tt.tokenizer, err, tt.wantErr)
			}
		})
	}
}

// resolvedGrammarCompilation wraps an already-built matcher as a finished
// compilation.
func resolvedGrammarCompilation(m *xgrammar.Matcher) *grammarCompilation {
	c := &grammarCompilation{done: make(chan struct{}), grammar: &grammar{m: m}}
	close(c.done)
	return c
}

func TestPrepareGrammarUnavailable(t *testing.T) {
	r := &Runner{
		Model:         textOnlyModel{},
		Tokenizer:     newTestTokenizer(t, []int32{7}),
		contextLength: 32,
	}
	request := &Request{CompletionRequest: CompletionRequest{
		Prompt: "0",
		Format: json.RawMessage(`"json"`),
	}}
	err := r.Prepare(request)
	var statusErr api.StatusError
	if !errors.As(err, &statusErr) {
		t.Fatalf("Prepare error = %T %v, want api.StatusError", err, err)
	}
	if statusErr.StatusCode != http.StatusNotImplemented {
		t.Fatalf("status = %d, want %d", statusErr.StatusCode, http.StatusNotImplemented)
	}
}

type grammarTestDrafter struct{}

func (grammarTestDrafter) open([]any) draftSession { return grammarTestDraftSession{} }
func (grammarTestDrafter) draftLimit() int         { return 0 }

type grammarTestDraftSession struct{}

func (grammarTestDraftSession) propose(*mlx.Array, int) *draftCandidates { return nil }
func (grammarTestDraftSession) committed(*mlx.Array, *mlx.Array, int, []batch.MediaItem) {
}
func (grammarTestDraftSession) settle(*mlx.Array) {}
func (grammarTestDraftSession) close()            {}

func TestGrammarParksSpeculation(t *testing.T) {
	s := &speculation{drafter: grammarTestDrafter{}, depth: newDepthController()}
	constrained := s.open(Request{Grammar: resolvedGrammarCompilation(&xgrammar.Matcher{})}, nil)
	defer constrained.close()
	if constrained.enabled {
		t.Fatal("structured output request enabled speculative decoding")
	}

	unconstrained := s.open(Request{}, nil)
	defer unconstrained.close()
	if !unconstrained.enabled {
		t.Fatal("ordinary request unexpectedly disabled speculative decoding")
	}
}
