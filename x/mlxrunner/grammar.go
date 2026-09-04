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

const (
	maxGrammarBytes = 1 << 20
	maxGrammarDepth = 128
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
	source, err := parseGrammar(format)
	if err != nil || source == "" {
		return nil, err
	}
	if e == nil {
		return nil, api.StatusError{StatusCode: http.StatusNotImplemented, ErrorMessage: "structured output is unavailable"}
	}
	return e.compile(source), nil
}

// parseGrammar returns the format's structural tag, or "" when the format
// asks for no structured output.
func parseGrammar(format json.RawMessage) (string, error) {
	switch string(format) {
	case ``, `null`, `""`:
		return "", nil
	}
	if len(format) > maxGrammarBytes {
		return "", fmt.Errorf("invalid format: grammar is %d bytes; limit is %d", len(format), maxGrammarBytes)
	}
	if !utf8.Valid(format) {
		return "", errors.New("invalid format: grammar is not valid UTF-8")
	}
	if format[0] != '{' {
		return "", errors.New("invalid format: expected a structural tag")
	}

	decoder := json.NewDecoder(bytes.NewReader(format))
	decoder.UseNumber()
	if _, err := decoder.Token(); err != nil {
		return "", fmt.Errorf("invalid format: %w", err)
	}

	depth, wantKey, key, structuralTag := 1, true, "", false
	for depth > 0 {
		token, err := decoder.Token()
		if err != nil {
			if err == io.EOF {
				return "", errors.New("invalid format: unexpected end of JSON")
			}
			return "", fmt.Errorf("invalid format: %w", err)
		}
		delim, isDelim := token.(json.Delim)
		if depth == 1 && !(isDelim && (delim == '}' || delim == ']')) {
			if wantKey {
				key, _ = token.(string)
			} else if key == "type" {
				value, _ := token.(string)
				structuralTag = value == "structural_tag"
			}
			wantKey = !wantKey
		}
		if isDelim {
			switch delim {
			case '{', '[':
				depth++
				if depth > maxGrammarDepth {
					return "", fmt.Errorf("invalid format: grammar nesting exceeds %d levels", maxGrammarDepth)
				}
			case '}', ']':
				depth--
			}
		}
	}

	if _, err := decoder.Token(); err != io.EOF {
		if err != nil {
			return "", fmt.Errorf("invalid format: %w", err)
		}
		return "", errors.New("invalid format: grammar contains more than one JSON value")
	}
	if !structuralTag {
		return "", errors.New("invalid format: expected a structural tag")
	}
	return string(format), nil
}

func (e *grammarEngine) compile(source string) *grammarCompilation {
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
		matcher, err := e.compiler.Compile(source)
		if err != nil {
			// A grammar can pass parseGrammar yet be rejected by the engine
			// (e.g. an empty enum); that is still a request error.
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

// accept advances each constraining row grammar over the row's newly
// committed token, grammars[i] and committed[i] aligned. errs[i] carries a
// row's rejection or fault, which ends that request rather than the runner;
// errs is nil when every row succeeded.
func (e *grammarEngine) accept(grammars []*grammar, committed []int32) []error {
	var errs []error
	for i, g := range grammars {
		if !g.constraining() {
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

// mask fills and applies each constraining row grammar's token masks over
// the batch's [B, L, V] logits: row b's position i constrains the token
// after drafts[b][:i], with nil drafts masking one position per row from
// the current state. Each matcher advances through its drafts as the
// positions fill and is rolled back before returning, so its state is
// unchanged; positions past a rejected draft stay unmasked.
func (e *grammarEngine) mask(grammars []*grammar, logits *mlx.Array, drafts [][]int32) (*mlx.Array, []error) {
	positions := 1
	for _, ids := range drafts {
		positions = max(positions, len(ids)+1)
	}
	packed := make([]int32, len(grammars)*positions*e.words)
	for i := range packed {
		packed[i] = -1
	}
	var errs []error
	apply := false
	for b, g := range grammars {
		if !g.constraining() {
			continue
		}
		var ids []int32
		if drafts != nil {
			ids = drafts[b]
		}
		block := packed[b*positions*e.words : (b+1)*positions*e.words]
		constrains, err := e.fill(g, block, ids)
		if err != nil {
			if errs == nil {
				errs = make([]error, len(grammars))
			}
			errs[b] = err
			// Nothing from a faulted row may constrain.
			for j := range block {
				block[j] = -1
			}
			continue
		}
		apply = apply || constrains
	}
	if !apply {
		return logits, errs
	}
	rows := len(grammars) * positions
	masked := e.apply(logits.Reshape(rows, logits.Dim(2)), mlx.FromValues(packed, rows, e.words))
	return masked.Reshape(len(grammars), positions, logits.Dim(2)), errs
}

// fill fills one row's per-position masks, advancing the matcher through
// the drafts and rolling the whole advance back before returning.
func (e *grammarEngine) fill(g *grammar, packed []int32, ids []int32) (bool, error) {
	constrains := false
	advanced := 0
	var walkErr error
	for i := range len(ids) + 1 {
		row := packed[i*e.words : (i+1)*e.words]
		constrained, err := g.m.Fill(row)
		if err != nil {
			walkErr = fmt.Errorf("grammar: fill token mask: %w", err)
			break
		}
		if constrained && !slices.ContainsFunc(row, func(w int32) bool { return w != 0 }) {
			// No token continues this state: fatal at position 0, whose state
			// is committed; a later position is left unmasked, and a run kept
			// into it fails at its accept.
			if i == 0 {
				walkErr = errors.New("grammar: token mask rejects every vocabulary token")
			} else {
				for j := range row {
					row[j] = -1
				}
			}
			break
		}
		constrains = constrains || constrained
		if i == len(ids) {
			break
		}
		if g.m.Accept(ids[i]) != nil {
			// A rejected draft ends the walk; a fault resurfaces at the
			// committed run's accept.
			break
		}
		advanced++
	}
	if err := g.m.Rollback(advanced); err != nil && walkErr == nil {
		walkErr = fmt.Errorf("grammar: rollback draft tokens: %w", err)
	}
	return constrains, walkErr
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
