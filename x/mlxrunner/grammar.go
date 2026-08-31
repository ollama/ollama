package mlxrunner

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"path/filepath"
	"slices"
	"sync"
	"unicode/utf8"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/xgrammar"
	"github.com/ollama/ollama/x/tokenizer"
)

const maxGrammarVocabSize = 1 << 20

const (
	maxGrammarSchemaBytes = 1 << 20
	maxGrammarSchemaDepth = 128
	// The token cap bounds grammar compile cost, which the serial runner
	// pays as head-of-line blocking of the request queue.
	maxGrammarSchemaTokens = 1 << 14
)

const (
	grammarCompileThreads    = 8
	grammarCompileCacheBytes = 128 << 20
)

// grammarEngine is the runner's structured-output subsystem: xgrammar bound
// to the model's vocabulary, plus the pinned lookup table for expanding
// packed token masks on the device.
type grammarEngine struct {
	// compileMu is the single compile slot: one native compile at a time
	// bounds the engine's compile threads and memory, and close takes it to
	// order the compiler's release against in-flight compiles.
	compileMu sync.Mutex
	compiler  *xgrammar.Compiler

	// words is the width of one packed mask row, ceil(vocab/32).
	words int

	maskTable  *mlx.Array
	byteShifts *mlx.Array
}

func newGrammarEngine(logitsWidth int, tokenizer *tokenizer.Tokenizer) *grammarEngine {
	library, err := mlx.LoadedLibraryPath()
	if err != nil {
		slog.Warn("Structured output is unavailable", "error", err)
		return nil
	}
	if err := validateGrammarVocab(logitsWidth, tokenizer.VocabSize()); err != nil {
		slog.Warn("Structured output is unavailable", "error", err)
		return nil
	}
	pieces := make([]string, logitsWidth)
	for id := range logitsWidth {
		pieces[id] = tokenizer.Decode([]int32{int32(id)})
	}
	stops := slices.DeleteFunc(slices.Clone(tokenizer.EOSTokens()), func(id int32) bool {
		return id < 0 || int(id) >= logitsWidth
	})
	compiler, err := xgrammar.New(filepath.Dir(library), pieces, logitsWidth, stops, grammarCompileThreads, grammarCompileCacheBytes)
	if err != nil {
		slog.Warn("Structured output is unavailable", "error", err)
		return nil
	}
	e := &grammarEngine{compiler: compiler}
	e.initMask(logitsWidth)
	slog.Info("Structured output initialized", "library", "xgrammar", "version", compiler.Version(), "vocab_size", logitsWidth, "path", compiler.Path())
	return e
}

// The grammar vocabulary is always the logits width. A tokenizer longer
// than the model's head (input-only tokens, e.g. Llama 3.2 Vision's image
// tokens) is fine: those ids can never be sampled, so they stay out of the
// grammar's vocabulary.
func validateGrammarVocab(logitsWidth, tokenizerSize int) error {
	if tokenizerSize <= 0 {
		return fmt.Errorf("invalid tokenizer vocabulary size %d", tokenizerSize)
	}
	if logitsWidth <= 0 {
		return fmt.Errorf("invalid model logits width %d", logitsWidth)
	}
	if logitsWidth > maxGrammarVocabSize {
		return fmt.Errorf("model logits width %d exceeds structured output limit %d", logitsWidth, maxGrammarVocabSize)
	}
	return nil
}

// initMask builds the byte-to-mask lookup table for expanding packed token
// masks on the device, pinned for the runner's lifetime: row v holds, for
// each of the byte value v's eight bits low to high, 0 where the bit is set
// (token allowed) and -inf where it is clear.
func (e *grammarEngine) initMask(vocabSize int) {
	e.words = (vocabSize + 31) / 32
	vals := make([]float32, 256*8)
	for v := range 256 {
		for bit := range 8 {
			if v>>bit&1 == 0 {
				vals[v*8+bit] = float32(math.Inf(-1))
			}
		}
	}
	e.maskTable = mlx.FromValues(vals, 256, 8)
	e.byteShifts = mlx.FromValues([]int32{0, 8, 16, 24}, 4)
	mlx.Pin(e.maskTable, e.byteShifts)
}

func (e *grammarEngine) close() {
	e.compileMu.Lock()
	defer e.compileMu.Unlock()
	if e.compiler != nil {
		e.compiler.Close()
		e.compiler = nil
	}
	mlx.Unpin(e.maskTable, e.byteShifts)
	e.maskTable, e.byteShifts = nil, nil
}

// prepare parses a request format and, when it asks for structured output,
// launches and returns the grammar compilation; a format that asks for none
// returns nil. Safe on a nil subsystem, which reports structured output
// unavailable.
func (e *grammarEngine) prepare(format json.RawMessage) (*grammarCompilation, error) {
	spec, err := parseGrammar(format)
	if err != nil || spec == nil {
		return nil, err
	}
	if e == nil {
		return nil, api.StatusError{StatusCode: http.StatusNotImplemented, ErrorMessage: "structured output is unavailable"}
	}
	return e.compile(spec), nil
}

type grammarSpec struct {
	kind   xgrammar.Kind
	source string
}

func parseGrammar(format json.RawMessage) (*grammarSpec, error) {
	if len(format) > 0 {
		switch string(format) {
		case `null`, `""`:
			return nil, nil
		case `"json"`:
			// The API documents "json" as producing a JSON object; the engine's
			// builtin JSON grammar would also admit arrays and bare values.
			return &grammarSpec{kind: xgrammar.JSONSchema, source: `{"type":"object"}`}, nil
		default:
			if format[0] != '{' {
				return nil, errors.New("invalid format: expected \"json\" or a valid JSON Schema object")
			}
			if err := validateGrammarSchema(format); err != nil {
				return nil, fmt.Errorf("invalid JSON Schema: %w", err)
			}
			return &grammarSpec{kind: xgrammar.JSONSchema, source: string(format)}, nil
		}
	}
	return nil, nil
}

func validateGrammarSchema(schema []byte) error {
	if len(schema) > maxGrammarSchemaBytes {
		return fmt.Errorf("schema is %d bytes; limit is %d", len(schema), maxGrammarSchemaBytes)
	}
	if !utf8.Valid(schema) {
		return errors.New("schema is not valid UTF-8")
	}

	decoder := json.NewDecoder(bytes.NewReader(schema))
	decoder.UseNumber()
	first, err := decoder.Token()
	if err != nil {
		return fmt.Errorf("invalid JSON: %w", err)
	}
	if first != json.Delim('{') {
		return errors.New("schema must be a JSON object")
	}

	tokens, depth := 1, 1
	for depth > 0 {
		token, err := decoder.Token()
		if err != nil {
			if err == io.EOF {
				return errors.New("unexpected end of JSON")
			}
			return fmt.Errorf("invalid JSON: %w", err)
		}
		tokens++
		if tokens > maxGrammarSchemaTokens {
			return fmt.Errorf("schema contains more than %d JSON tokens", maxGrammarSchemaTokens)
		}

		if delim, ok := token.(json.Delim); ok {
			switch delim {
			case '{', '[':
				depth++
				if depth > maxGrammarSchemaDepth {
					return fmt.Errorf("schema nesting exceeds %d levels", maxGrammarSchemaDepth)
				}
			case '}', ']':
				depth--
			}
		}
	}

	if _, err := decoder.Token(); err != io.EOF {
		if err != nil {
			return fmt.Errorf("invalid JSON after schema object: %w", err)
		}
		return errors.New("schema contains more than one JSON value")
	}
	return nil
}

func (e *grammarEngine) compile(spec *grammarSpec) *grammarCompilation {
	c := &grammarCompilation{done: make(chan struct{})}
	go func() {
		defer close(c.done)

		e.compileMu.Lock()
		defer e.compileMu.Unlock()
		// A request cancelled while queued for the slot never compiles.
		c.mu.Lock()
		abandoned := c.abandoned
		c.mu.Unlock()
		if abandoned {
			return
		}
		if e.compiler == nil {
			c.err = errors.New("grammar engine closed")
			return
		}
		matcher, err := e.compiler.Compile(spec.kind, spec.source)
		if err != nil {
			// A schema can pass validateGrammarSchema yet be rejected by
			// the engine (e.g. an empty enum); that is still a request error.
			c.err = api.StatusError{
				StatusCode:   http.StatusBadRequest,
				ErrorMessage: fmt.Sprintf("invalid structured output grammar: %v", err),
			}
			return
		}
		c.mu.Lock()
		defer c.mu.Unlock()
		if c.abandoned {
			matcher.Close()
			return
		}
		c.grammar = &grammar{m: matcher}
	}()
	return c
}

// A grammarCompilation runs concurrently with prompt processing. resolve
// blocks until it finishes; close releases the grammar without waiting, so a
// cancelled request never holds the serial request loop through a compile.
type grammarCompilation struct {
	done chan struct{}
	err  error // written by the compile goroutine, read only after done

	// mu orders the compile's finish against close: whichever runs second
	// frees the matcher.
	mu        sync.Mutex
	abandoned bool
	grammar   *grammar
}

// A nil compilation resolves to no grammar.
func (c *grammarCompilation) resolve(ctx context.Context) (*grammar, error) {
	if c == nil {
		return nil, nil
	}
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-c.done:
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	// A Closed compilation must never resolve to an unconstrained decode.
	if c.abandoned {
		return nil, errors.New("grammar compilation abandoned")
	}
	return c.grammar, c.err
}

func (c *grammarCompilation) close() {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.abandoned = true
	c.grammar.close()
	c.grammar = nil
}

// grammar is a request's compiled grammar, the runner's seam to the engine.
// Methods are safe on a nil grammar, which never constrains.
type grammar struct {
	m *xgrammar.Matcher
}

// constraining reports whether sampling is currently constrained, read from
// the matcher's state: a grammar constrains from its first token until its
// state machine terminates. Kinds that trigger mid-response will decide
// this from richer matcher state.
func (g *grammar) constraining() bool {
	return g != nil && !g.m.Terminated()
}

func (g *grammar) close() {
	if g != nil {
		g.m.Close()
	}
}

// hasGrammar reports whether any row carries a grammar — the read that
// decides whether a step takes the deferred shape: grammar work needs the
// committed token values on the host, so the step's sample cannot fuse
// onto the forward's chain.
func (e *grammarEngine) hasGrammar(grammars []*grammar) bool {
	for _, g := range grammars {
		if g != nil {
			return true
		}
	}
	return false
}

// accept advances a batch's grammars over its newly committed tokens —
// every row's grammar accepts the row's token, constraining or not, with
// grammars[i] and committed[i] aligned. Each token was sampled under its
// own matcher's mask, so a rejection is an engine fault; errs[i] carries
// row i's fault, which ends that request rather than the runner, and errs
// is nil when every row succeeded.
func (e *grammarEngine) accept(grammars []*grammar, committed []int32) []error {
	var errs []error
	for i, g := range grammars {
		if g == nil {
			continue
		}
		if err := g.m.Accept(committed[i]); err != nil {
			if errs == nil {
				errs = make([]error, len(grammars))
			}
			errs[i] = fmt.Errorf("grammar: accept sampled token %d: %w", committed[i], err)
		}
	}
	return errs
}

// mask fills each constraining row's packed token mask and applies them to
// the batch's logits in one device op, logits row i masked under
// grammars[i]. Unconstrained, terminated, and failed rows keep their logits
// (all-ones mask rows add zero). errs as in accept.
func (e *grammarEngine) mask(grammars []*grammar, logits *mlx.Array) (*mlx.Array, []error) {
	var errs []error
	packed := make([]int32, len(grammars)*e.words)
	for i := range packed {
		packed[i] = -1
	}
	apply := false
	for i, g := range grammars {
		if !g.constraining() {
			continue
		}
		row := packed[i*e.words : (i+1)*e.words]
		constrained, err := g.m.Fill(row)
		if err != nil {
			if errs == nil {
				errs = make([]error, len(grammars))
			}
			errs[i] = fmt.Errorf("grammar: fill token mask: %w", err)
			continue
		}
		// An all-zero mask would send every logit to -inf and sampling to NaN.
		if constrained && !slices.ContainsFunc(row, func(w int32) bool { return w != 0 }) {
			if errs == nil {
				errs = make([]error, len(grammars))
			}
			errs[i] = errors.New("grammar: token mask rejects every vocabulary token")
			continue
		}
		apply = apply || constrained
	}
	if !apply {
		return logits, errs
	}
	return e.apply(logits, mlx.FromValues(packed, len(grammars), e.words)), errs
}

// apply masks logits under packed token masks, one row per sequence: logits
// is [B, V], packed is [B, words], and disallowed tokens come back -inf.
// Each mask word is split into bytes, each byte gathers its eight mask
// values from the table, and the flattened rows are added to the logits.
func (e *grammarEngine) apply(logits, packed *mlx.Array) *mlx.Array {
	maskBytes := packed.ExpandDims(-1).RightShift(e.byteShifts).BitwiseAnd(mlx.FromValue(255))
	mask := e.maskTable.TakeAxis(maskBytes, 0).Flatten(1, 3)
	mask = mask.Slice(mlx.Slice(), mlx.Slice(0, logits.Dim(1))).AsType(logits.DType())
	return logits.Add(mask)
}
