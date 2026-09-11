package mlxrunner

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"net/http"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/x/internal/mlxtest"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	sampler "github.com/ollama/ollama/x/mlxrunner/sample"
	"github.com/ollama/ollama/x/mlxrunner/xgrammar"
)

// schemaTag wraps a JSON Schema into the structural tag the MLX client
// sends for a schema format.
func schemaTag(schema string) string {
	return `{"type":"structural_tag","format":{"type":"json_schema","json_schema":` + schema + `}}`
}

func nestedGrammar(depth int) string {
	return `{"type":"structural_tag","format":` + strings.Repeat("[", depth-1) + `0` + strings.Repeat("]", depth-1) + `}`
}

func TestParseGrammar(t *testing.T) {
	invalidUTF8 := `{"type":"structural_tag","value":"` + "\xff" + `"}`
	tests := []struct {
		name    string
		format  string
		want    string
		wantErr string
	}{
		{name: "unset"},
		{name: "null", format: `null`},
		{name: "empty", format: `""`},
		{name: "structural tag", format: schemaTag(`{"type":"integer"}`), want: schemaTag(`{"type":"integer"}`)},
		{name: "type after nested members", format: `{"format":{"type":"any_text","excludes":["type"]},"type":"structural_tag"}`, want: `{"format":{"type":"any_text","excludes":["type"]},"type":"structural_tag"}`},
		{name: "maximum depth", format: nestedGrammar(maxGrammarDepth), want: nestedGrammar(maxGrammarDepth)},
		{name: "too deep", format: nestedGrammar(maxGrammarDepth + 1), wantErr: "nesting exceeds"},
		{name: "too large", format: `{"value":"` + strings.Repeat("x", maxGrammarBytes) + `"}`, wantErr: "limit is 1048576"},
		{name: "invalid UTF-8", format: invalidUTF8, wantErr: "not valid UTF-8"},
		{name: "json", format: `"json"`, wantErr: "expected a structural tag"},
		{name: "schema", format: `{"type":"integer"}`, wantErr: "expected a structural tag"},
		{name: "untyped object", format: `{"format":{}}`, wantErr: "expected a structural tag"},
		{name: "nested type only", format: `{"format":{"type":"structural_tag"}}`, wantErr: "expected a structural tag"},
		{name: "whitespace", format: ` ` + schemaTag(`{}`) + ` `, wantErr: "expected a structural tag"},
		{name: "array", format: `[]`, wantErr: "expected a structural tag"},
		{name: "trailing value", format: `{"type":"structural_tag"} {}`, wantErr: "more than one JSON value"},
		{name: "malformed", format: `{"type":`, wantErr: "unexpected end"},
		{name: "invalid json", format: `{`, wantErr: "unexpected end"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseGrammar(json.RawMessage(tt.format))
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("parseGrammar error = %v, want containing %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Errorf("parseGrammar = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestParseGrammarDoesNotEchoOversizedInput(t *testing.T) {
	format := json.RawMessage("{" + strings.Repeat("x", maxGrammarBytes))
	_, err := parseGrammar(format)
	if err == nil || !strings.Contains(err.Error(), "grammar is 1048577 bytes") {
		t.Fatalf("parseGrammar error = %v, want bounded size error", err)
	}
	if len(err.Error()) > 256 {
		t.Fatalf("parseGrammar echoed oversized input in %d-byte error", len(err.Error()))
	}
}

func BenchmarkParseGrammar(b *testing.B) {
	grammar := json.RawMessage(schemaTag(`{"type":"object","properties":{"answer":{"type":"string","enum":["ok"]}},"required":["answer"],"additionalProperties":false}`))
	b.ReportAllocs()
	for b.Loop() {
		if _, err := parseGrammar(grammar); err != nil {
			b.Fatal(err)
		}
	}
}

func FuzzParseGrammar(f *testing.F) {
	for _, format := range []string{
		"",
		`"json"`,
		`{"type":"integer"}`,
		schemaTag(`{"type":"integer"}`),
		`{`,
	} {
		f.Add(format)
	}

	f.Fuzz(func(t *testing.T, format string) {
		source, err := parseGrammar(json.RawMessage(format))
		if err == nil && source != "" && source != format {
			t.Fatalf("parseGrammar = %q, want the format verbatim", source)
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
		Format: json.RawMessage(schemaTag(`{"type":"object"}`)),
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

func TestSpeculationGating(t *testing.T) {
	s := &speculation{drafter: grammarTestDrafter{}, depth: newDepthController()}
	constrained := s.open(Request{Grammar: resolvedGrammarCompilation(&xgrammar.Matcher{})}, nil)
	defer constrained.close()
	if !constrained.enabled {
		t.Fatal("structured output request disabled speculative decoding")
	}

	logprobs := s.open(Request{SamplerOpts: sampler.Options{Logprobs: true}}, nil)
	defer logprobs.close()
	if logprobs.enabled {
		t.Fatal("logprobs request enabled speculative decoding")
	}

	unconstrained := s.open(Request{}, nil)
	defer unconstrained.close()
	if !unconstrained.enabled {
		t.Fatal("ordinary request unexpectedly disabled speculative decoding")
	}
}

// testDigitGrammar builds a grammar over the decode fakes' digit
// vocabulary, with token 7 as the stop id.
func testDigitGrammar(t *mlxtest.T, schema string) (*grammarEngine, *grammar) {
	t.Helper()
	path, err := mlx.LoadedLibraryPath()
	if err != nil {
		t.Skipf("native MLX payload is not built: %v", err)
	}
	pieces := make([]string, mtpTestVocab)
	for i := range pieces {
		pieces[i] = fmt.Sprintf("%d", i)
	}
	compiler, err := xgrammar.New(filepath.Dir(path), pieces, mtpTestVocab, []int32{7}, 8, 128<<20)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			t.Skipf("native xgrammar payload is not built: %v", err)
		}
		t.Fatal(err)
	}
	t.Cleanup(compiler.Close)
	m, err := compiler.Compile(schemaTag(schema))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(m.Close)
	e := &grammarEngine{}
	e.initMask(mtpTestVocab)
	t.Cleanup(e.close)
	return e, &grammar{m: m}
}

func TestDecodeGrammarTransitions(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		// Drive every mode change (park, resume, a round with a rejection, park
		// again, EOS) while a shadow matcher replays the emitted tokens:
		// identical masks prove the matcher tracks exactly the emitted stream.
		const eos int32 = 7
		target := map[int32]int32{1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: eos, eos: 0}
		draftPredict := map[int32]int32{2: 3, 3: 4, 4: 5, 5: 0, 6: eos, eos: 0}
		r := mtpTestRunner(t, target, []int32{eos}, sampler.Options{})
		engine, g := testDigitGrammar(t, `{"type":"integer"}`)
		_, shadow := testDigitGrammar(t, `{"type":"integer"}`)
		r.grammarEngine = engine

		draft := &fakeKVDraft{predict: draftPredict}
		caches, _ := newMTPTestCaches(2)
		draft.draftCaches = caches[1:]
		r.cache.caches = caches
		r.spec = newSpeculation(r, draft, caches[:1], caches[1:])
		req := Request{
			Tokens:            []int32{1},
			CompletionRequest: CompletionRequest{Options: api.Options{NumPredict: 20}},
			SamplerOpts:       sampler.Options{},
		}
		spec := r.spec.open(req, nil)
		if spec == nil || !spec.enabled {
			t.Fatalf("want a drafting speculationSession, got %+v", spec)
		}
		pinDraftLimit(spec, 0)
		d := spec.decoder(mlx.FromValues([]int32{1}, 1), 0, g).(*speculativeDecoder)

		check := func(want []int32) {
			t.Helper()
			results, err := d.next(20)
			if err != nil {
				t.Fatalf("next: %v", err)
			}
			if got := resultIDs(results); !slices.Equal(got, want) {
				t.Fatalf("results = %v, want %v", got, want)
			}
			for _, id := range want {
				if err := shadow.m.Accept(id); err != nil {
					t.Fatalf("shadow accept %d: %v", id, err)
				}
			}
			requireSameMask(t, g, shadow, mtpTestVocab)
		}

		check([]int32{2}) // parked
		check([]int32{3}) // parked
		spec.limit = 2
		check([]int32{4})    // resume: the drained sample catches the matcher up
		check([]int32{5, 6}) // round: draft [5 0], rejection at 1, residual 6
		spec.limit = 0
		check([]int32{eos}) // parked again off the round's final token
		if !g.m.Terminated() {
			t.Fatal("matcher not terminated after the emitted EOS")
		}

		d.close()
		spec.close()
	})
}

func TestRunMTPDecodeGrammar(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		// The greedy decode chain under an integer grammar: the round's drafted
		// EOS ends the run through the done path, with every emitted token masked.
		const eos int32 = 7
		predict := map[int32]int32{1: 2, 2: 3, 3: 4, 4: eos, eos: 0}
		r := mtpTestRunner(t, predict, []int32{eos}, sampler.Options{})
		engine, g := testDigitGrammar(t, `{"type":"integer"}`)
		r.grammarEngine = engine
		draft := &fakeMTPDraft{predict: predict}
		caches, _ := newMTPTestCaches(1)
		r.cache.caches = caches
		r.spec = newSpeculation(r, draft, caches[:1], caches[1:])
		session, ch := newMTPTestSession(caches)

		req := Request{
			Responses:         ch,
			Tokens:            []int32{0},
			CompletionRequest: CompletionRequest{Options: api.Options{NumPredict: 20}},
			SamplerOpts:       sampler.Options{},
		}
		spec := r.spec.open(req, nil)
		pinDraftLimit(spec, 4)
		d := spec.decoder(mlx.FromValues([]int32{1}, 1), 1, g)
		if err := r.decode(context.Background(), req, session, d, 0); err != nil {
			t.Fatalf("decode: %v", err)
		}
		d.close()
		spec.close()

		content, final := collectResponses(ch)
		if content != "234" {
			t.Fatalf("content = %q, want %q", content, "234")
		}
		if final.DoneReason != 0 {
			t.Fatalf("DoneReason = %d, want 0 (EOS)", final.DoneReason)
		}
	})
}

func TestRunMTPDecodeGrammarRejectsInvalidDraft(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		// Target and draft predict 3 at every step, but the enum admits only
		// "12": the masks zero each draft's probability, verification rejects
		// them, and every emitted token is a residual from a masked distribution.
		const eos int32 = 7
		predict := map[int32]int32{1: 3, 3: 4, 2: eos, eos: 0}
		r := mtpTestRunner(t, predict, []int32{eos}, sampler.Options{})
		engine, g := testDigitGrammar(t, `{"enum":[12]}`)
		r.grammarEngine = engine
		draft := &fakeMTPDraft{predict: predict}
		caches, _ := newMTPTestCaches(1)
		r.cache.caches = caches
		r.spec = newSpeculation(r, draft, caches[:1], caches[1:])
		session, ch := newMTPTestSession(caches)

		req := Request{
			Responses:         ch,
			Tokens:            []int32{1},
			CompletionRequest: CompletionRequest{Options: api.Options{NumPredict: 20}},
			SamplerOpts:       sampler.Options{},
		}
		spec := r.spec.open(req, nil)
		pinDraftLimit(spec, 2)
		d := spec.decoder(mlx.FromValues([]int32{1}, 1), 1, g)
		if err := r.decode(context.Background(), req, session, d, 0); err != nil {
			t.Fatalf("decode: %v", err)
		}
		d.close()
		spec.close()

		content, final := collectResponses(ch)
		if content != "12" {
			t.Fatalf("content = %q, want %q", content, "12")
		}
		if final.DoneReason != 0 {
			t.Fatalf("DoneReason = %d, want 0 (EOS)", final.DoneReason)
		}
	})
}
