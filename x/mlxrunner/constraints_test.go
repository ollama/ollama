package mlxrunner

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"sync"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/constraint"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func TestParseConstraint(t *testing.T) {
	tests := []struct {
		name    string
		format  string
		kind    constraint.Kind
		source  string
		want    bool
		wantErr bool
	}{
		{name: "unset"},
		{name: "null", format: `null`},
		{name: "empty", format: `""`},
		{name: "json", format: `"json"`, kind: constraint.JSON, want: true},
		{name: "schema", format: `{"type":"integer"}`, kind: constraint.JSONSchema, source: `{"type":"integer"}`, want: true},
		{name: "unsupported string", format: `"xml"`, wantErr: true},
		{name: "whitespace JSON", format: ` "json" `, wantErr: true},
		{name: "whitespace schema", format: ` {"type":"integer"} `, wantErr: true},
		{name: "array", format: `[]`, wantErr: true},
		{name: "invalid json", format: `{`, wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseConstraint(json.RawMessage(tt.format))
			if (err != nil) != tt.wantErr {
				t.Fatalf("parseConstraint error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr {
				return
			}
			if (got != nil) != tt.want {
				t.Fatalf("parseConstraint = %#v, want present %v", got, tt.want)
			}
			if got != nil && (got.kind != tt.kind || got.source != tt.source) {
				t.Errorf("parseConstraint = %#v, want kind %v source %q", got, tt.kind, tt.source)
			}
		})
	}
}

func TestParseConstraintDoesNotEchoOversizedInput(t *testing.T) {
	format := json.RawMessage(strings.Repeat("x", maxConstraintSchemaBytes+1))
	_, err := parseConstraint(format)
	if err == nil || !strings.Contains(err.Error(), "input is 1048577 bytes") {
		t.Fatalf("parseConstraint error = %v, want bounded size error", err)
	}
	if len(err.Error()) > 256 {
		t.Fatalf("parseConstraint echoed oversized input in %d-byte error", len(err.Error()))
	}
}

func nestedConstraintSchema(depth int) []byte {
	return []byte(`{"value":` + strings.Repeat("[", depth-1) + `0` + strings.Repeat("]", depth-1) + `}`)
}

func constraintSchemaArray(entries int) []byte {
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

func TestValidateConstraintSchema(t *testing.T) {
	invalidUTF8 := append([]byte(`{"value":"`), 0xff)
	invalidUTF8 = append(invalidUTF8, []byte(`"}`)...)
	tests := []struct {
		name    string
		schema  []byte
		wantErr string
	}{
		{name: "object", schema: []byte(`{"type":"object","properties":{"answer":{"type":"string"}}}`)},
		{name: "maximum depth", schema: nestedConstraintSchema(maxConstraintSchemaDepth)},
		{name: "too deep", schema: nestedConstraintSchema(maxConstraintSchemaDepth + 1), wantErr: "nesting exceeds"},
		{name: "too many tokens", schema: constraintSchemaArray(maxConstraintSchemaTokens), wantErr: "more than 4096 JSON tokens"},
		{name: "too large", schema: []byte(`{"value":"` + strings.Repeat("x", maxConstraintSchemaBytes) + `"}`), wantErr: "limit is 1048576"},
		{name: "invalid UTF-8", schema: invalidUTF8, wantErr: "not valid UTF-8"},
		{name: "trailing value", schema: []byte(`{} {}`), wantErr: "more than one JSON value"},
		{name: "malformed", schema: []byte(`{"type":`), wantErr: "unexpected end"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateConstraintSchema(tt.schema)
			if tt.wantErr == "" {
				if err != nil {
					t.Fatal(err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("validateConstraintSchema error = %v, want containing %q", err, tt.wantErr)
			}
		})
	}
}

func BenchmarkValidateConstraintSchema(b *testing.B) {
	schema := []byte(`{"type":"object","properties":{"answer":{"type":"string","enum":["ok"]}},"required":["answer"],"additionalProperties":false}`)
	b.ReportAllocs()
	for b.Loop() {
		if err := validateConstraintSchema(schema); err != nil {
			b.Fatal(err)
		}
	}
}

func FuzzParseConstraint(f *testing.F) {
	for _, format := range []string{
		"",
		`"json"`,
		`{"type":"integer"}`,
		`{`,
	} {
		f.Add(format)
	}

	f.Fuzz(func(t *testing.T, format string) {
		spec, err := parseConstraint(json.RawMessage(format))
		if err != nil || spec == nil {
			return
		}
		switch spec.kind {
		case constraint.JSON, constraint.JSONSchema:
		default:
			t.Fatalf("parseConstraint kind = %d", spec.kind)
		}
	})
}

func TestConstraintVocabSize(t *testing.T) {
	for _, tt := range []struct {
		name       string
		configured int
		tokenizer  int
		want       int
		wantErr    bool
	}{
		{name: "tokenizer fallback", tokenizer: 32, want: 32},
		{name: "padded model vocabulary", configured: 40, tokenizer: 32, want: 40},
		{name: "model smaller than tokenizer", configured: 31, tokenizer: 32, wantErr: true},
		{name: "invalid tokenizer", tokenizer: -1, wantErr: true},
		{name: "allocation bound", configured: maxConstraintVocabSize + 1, tokenizer: 32, wantErr: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := constraintVocabSize(tt.configured, tt.tokenizer)
			if (err != nil) != tt.wantErr {
				t.Fatalf("constraintVocabSize(%d, %d) error = %v, wantErr %v", tt.configured, tt.tokenizer, err, tt.wantErr)
			}
			if got != tt.want {
				t.Errorf("constraintVocabSize(%d, %d) = %d, want %d", tt.configured, tt.tokenizer, got, tt.want)
			}
		})
	}
}

func TestParseModelVocabSize(t *testing.T) {
	for _, tt := range []struct {
		name    string
		config  string
		want    int
		wantErr bool
	}{
		{name: "top level", config: `{"vocab_size":40,"text_config":{"vocab_size":32}}`, want: 40},
		{name: "text config", config: `{"text_config":{"vocab_size":32}}`, want: 32},
		{name: "missing", config: `{}`},
		{name: "negative top level", config: `{"vocab_size":-1}`, wantErr: true},
		{name: "negative text config", config: `{"text_config":{"vocab_size":-1}}`, wantErr: true},
		{name: "invalid JSON", config: `{`, wantErr: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseModelVocabSize([]byte(tt.config))
			if (err != nil) != tt.wantErr {
				t.Fatalf("parseModelVocabSize error = %v, wantErr %v", err, tt.wantErr)
			}
			if got != tt.want {
				t.Errorf("parseModelVocabSize = %d, want %d", got, tt.want)
			}
		})
	}
}

type trackingConstraint struct {
	closed chan struct{}
	once   sync.Once
}

func (*trackingConstraint) VocabSize() int               { return 1 }
func (*trackingConstraint) Fill() ([]int32, bool, error) { return []int32{1}, true, nil }
func (*trackingConstraint) Accept(int32) error           { return nil }
func (c *trackingConstraint) Close()                     { c.once.Do(func() { close(c.closed) }) }

func TestRunRequestRetainsConstraintUntilPipelineReturns(t *testing.T) {
	matcher := &trackingConstraint{closed: make(chan struct{})}
	started := make(chan struct{})
	release := make(chan struct{})
	ctx, cancel := context.WithCancel(t.Context())
	request := Request{
		Ctx:        ctx,
		Constraint: matcher,
		Pipeline: func(context.Context, Request) error {
			close(started)
			<-release
			return nil
		},
	}
	done := make(chan error, 1)
	go func() { done <- (&Runner{}).runRequest(request) }()

	<-started
	cancel()
	select {
	case <-matcher.closed:
		t.Fatal("constraint closed while the pipeline was still running")
	default:
	}
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	select {
	case <-matcher.closed:
	default:
		t.Fatal("constraint was not closed after the pipeline returned")
	}
}

func TestPrepareConstraintUnavailable(t *testing.T) {
	r := &Runner{
		Model:         textOnlyModel{},
		Tokenizer:     newTestTokenizer(t, []int32{7}),
		contextLength: 32,
		constraintErr: errors.New("library missing"),
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

type constraintTestDrafter struct{}

func (constraintTestDrafter) open([]any) draftSession { return constraintTestDraftSession{} }
func (constraintTestDrafter) draftLimit() int         { return 0 }

type constraintTestDraftSession struct{}

func (constraintTestDraftSession) propose(*mlx.Array, int) *draftCandidates { return nil }
func (constraintTestDraftSession) committed(*mlx.Array, *mlx.Array, int, []batch.MediaItem) {
}
func (constraintTestDraftSession) settle(*mlx.Array) {}
func (constraintTestDraftSession) close()            {}

func TestConstraintParksSpeculation(t *testing.T) {
	s := &speculation{drafter: constraintTestDrafter{}, depth: newDepthController()}
	constrained := s.open(Request{Constraint: &constraint.Matcher{}}, nil)
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
