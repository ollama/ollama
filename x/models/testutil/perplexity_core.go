package testutil

import (
	"fmt"
	"io"
	"math"

	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

// Mode selects the perplexity scoring methodology. Different communities use
// different conventions; this enum makes the choice explicit and lets a single
// implementation reproduce numbers from each ecosystem.
type Mode int

const (
	// ModeHarness reproduces EleutherAI lm-evaluation-harness' wikitext task:
	// document-level rolling loglikelihood with context_len=1 (each window
	// shares one token of context with the prior window). The prefix token
	// (BOS or EOS, whichever the model has) is prepended once at document
	// start, and every prediction position in each window is scored.
	ModeHarness Mode = iota

	// ModeLlamaCpp reproduces llama.cpp's llama-perplexity tool: non-overlapping
	// chunks of n_ctx tokens fed as a flat stream, the first token of each
	// chunk is replaced with BOS for the forward pass, and only the second
	// half of each chunk is scored.
	ModeLlamaCpp
)

func (m Mode) String() string {
	switch m {
	case ModeHarness:
		return "harness"
	case ModeLlamaCpp:
		return "llamacpp"
	default:
		return fmt.Sprintf("Mode(%d)", int(m))
	}
}

// ParseMode converts a CLI string into a Mode.
func ParseMode(s string) (Mode, error) {
	switch s {
	case "harness", "":
		return ModeHarness, nil
	case "llamacpp":
		return ModeLlamaCpp, nil
	default:
		return 0, fmt.Errorf("unknown perplexity mode %q (valid: harness, llamacpp)", s)
	}
}

// Logger is the minimal logging surface used by the perplexity core. *log.Logger,
// *testing.T (via an adapter), and any io.Writer-backed type can satisfy this.
type Logger interface {
	Logf(format string, args ...any)
}

// noopLogger discards everything.
type noopLogger struct{}

func (noopLogger) Logf(string, ...any) {}

// stderrLogger writes to an io.Writer.
type writerLogger struct {
	w io.Writer
}

func (l writerLogger) Logf(format string, args ...any) {
	fmt.Fprintf(l.w, format+"\n", args...)
}

// NewWriterLogger wraps an io.Writer (e.g. os.Stderr) as a Logger.
func NewWriterLogger(w io.Writer) Logger {
	if w == nil {
		return noopLogger{}
	}
	return writerLogger{w: w}
}

// Document is a unit of corpus text to be evaluated. In ModeHarness each
// document is processed independently with its own rolling-window pass; in
// ModeLlamaCpp all documents are concatenated into a single token stream.
//
// Some corpora (notably wikitext-2-raw-v1) require dataset-specific
// preprocessing of the model input that does NOT change the word/byte
// counts used as denominators for word_perplexity and byte_perplexity.
// To support that, Document distinguishes between Text (original text,
// used for byte/word counting and as the default model input) and
// ScoringText (optional alternative passed to the tokenizer). When
// ScoringText is empty, Text is used unchanged.
type Document struct {
	// Text is the original document text, used for word and byte
	// statistics. If TokenIDs is set, Text is only used for these
	// denominators.
	Text string

	// ScoringText, if non-empty, is the text actually tokenized and fed
	// to the model. Used by the wikitext task loader to apply the
	// EleutherAI detokenizer transform without disturbing the
	// word/byte denominators.
	ScoringText string

	// TokenIDs is the pre-tokenized form. If set it overrides Text /
	// ScoringText for the forward-pass input. Word/byte counts still
	// come from Text.
	TokenIDs []int32
}

// PPLOptions configures a perplexity run.
type PPLOptions struct {
	// Mode selects the scoring methodology.
	Mode Mode

	// MaxLength is the per-window context length passed to the model
	// (analogous to lm-eval-harness max_length or llama.cpp n_ctx).
	// Defaults: 2048 for harness mode, 512 for llamacpp mode.
	MaxLength int

	// MaxDocs caps the number of documents processed (harness mode) or
	// chunks evaluated (llamacpp mode). Zero or negative = process all.
	MaxDocs int

	// PrefixTokenID is the token prepended at document start in harness
	// mode (typically BOS or EOS). If zero, the model tokenizer's BOS is
	// used. Set to -1 to disable prefixing.
	PrefixTokenID int32

	// BOSSwapLlamaCpp enables the llama.cpp BOS substitution trick at the
	// start of each chunk. Only meaningful in ModeLlamaCpp. Defaults to true
	// when in that mode.
	BOSSwapLlamaCpp bool

	// LogEvery prints a running PPL message every N documents/chunks.
	// Zero = use a sensible default.
	LogEvery int
}

func (o *PPLOptions) defaultsFor() {
	if o.MaxLength <= 0 {
		switch o.Mode {
		case ModeLlamaCpp:
			o.MaxLength = 512
		default:
			o.MaxLength = 2048
		}
	}
	if o.LogEvery == 0 {
		o.LogEvery = 10
	}
}

// PPLResult holds the aggregate output of a perplexity run. Per-token NLL and
// NLL-squared sums are tracked so callers can compute confidence intervals
// via the delta method.
type PPLResult struct {
	Mode Mode

	TotalNLL    float64 // sum of -log P over all scored tokens
	TotalNLL2   float64 // sum of (-log P)^2; for stderr via delta method
	TotalTokens int     // number of scored tokens (denominator for token PPL)
	TotalChars  int     // number of *original* characters across scored docs (for byte PPL / bpb)
	TotalWords  int     // number of *original* words across scored docs (for word PPL)

	// TokenPerplexity is exp(TotalNLL/TotalTokens). This is the
	// "perplexity" most people mean when they say "PPL" without
	// qualification.
	TokenPerplexity float64

	// StderrTokenPPL is the standard error of TokenPerplexity, propagated
	// from per-token NLL variance via the delta method.
	StderrTokenPPL float64

	// WordPerplexity, BytePerplexity, and BitsPerByte are tokenizer-
	// independent metrics computed from TotalNLL and the original
	// (pre-tokenization) document text. Only populated when input
	// documents include Text (or DocumentText is provided to score).
	WordPerplexity float64
	BytePerplexity float64
	BitsPerByte    float64
}

// RunPerplexity is the pure entry point. It takes a loaded model, a sequence
// of documents, and options, and returns a PPLResult plus an error. No
// dependency on the testing package.
func RunPerplexity(m base.Model, docs []Document, opts PPLOptions, log Logger) (PPLResult, error) {
	if log == nil {
		log = noopLogger{}
	}
	opts.defaultsFor()

	// Resolve the prefix token from the tokenizer if not explicitly set.
	if opts.PrefixTokenID == 0 {
		if tok := m.Tokenizer(); tok != nil {
			if bos := tok.BOS(); bos >= 0 {
				opts.PrefixTokenID = bos
			}
		}
	}

	// Tokenize any documents that arrive as raw text. Prefer ScoringText
	// (the model-facing variant) when set; fall back to Text otherwise.
	tok := m.Tokenizer()
	for i := range docs {
		if len(docs[i].TokenIDs) > 0 {
			continue
		}
		modelText := docs[i].ScoringText
		if modelText == "" {
			modelText = docs[i].Text
		}
		if modelText == "" {
			continue
		}
		if tok == nil {
			return PPLResult{}, fmt.Errorf("document %d has Text but model has no tokenizer", i)
		}
		docs[i].TokenIDs = tok.Encode(modelText, false)
	}

	switch opts.Mode {
	case ModeHarness:
		return runHarness(m, docs, opts, log)
	case ModeLlamaCpp:
		return runLlamaCpp(m, docs, opts, log)
	default:
		return PPLResult{}, fmt.Errorf("unknown perplexity mode: %v", opts.Mode)
	}
}

// runHarness reproduces lm-evaluation-harness' loglikelihood_rolling: each
// document is tokenized independently, prefixed with the model's BOS/EOS,
// and split into a series of windows of length MaxLength. Consecutive
// windows share `context_len=1` token of overlap. Every prediction position
// in each window is scored. Results are aggregated across documents.
func runHarness(m base.Model, docs []Document, opts PPLOptions, log Logger) (PPLResult, error) {
	result := PPLResult{Mode: opts.Mode}

	maxDocs := len(docs)
	if opts.MaxDocs > 0 && opts.MaxDocs < maxDocs {
		maxDocs = opts.MaxDocs
	}

	for i := range maxDocs {
		doc := docs[i]
		if len(doc.TokenIDs) == 0 {
			continue
		}

		nll, nll2, n, err := scoreDocumentHarness(m, doc.TokenIDs, opts)
		if err != nil {
			return result, fmt.Errorf("doc %d: %w", i, err)
		}

		result.TotalNLL += nll
		result.TotalNLL2 += nll2
		result.TotalTokens += n
		result.TotalChars += len(doc.Text)
		result.TotalWords += countWords(doc.Text)

		if (i+1)%opts.LogEvery == 0 || i == maxDocs-1 || i < 3 {
			running := math.Exp(result.TotalNLL / float64(result.TotalTokens))
			log.Logf("  [%d/%d] running token PPL=%.4f", i+1, maxDocs, running)
		}

		// Free intermediate tensors between docs to keep memory bounded.
		mlx.Sweep()
		mlx.ClearCache()
	}

	finalize(&result)
	return result, nil
}

// scoreDocumentHarness implements lm-evaluation-harness'
// `loglikelihood_rolling` for context_len=1, which is what the wikitext task
// uses. The contract is the one in `lm_eval/utils.get_rolling_token_windows`:
//
//   - The first window's input is `[prefix, t_0, ..., t_{L-2}]` of length L
//     (where L = min(maxLen, doc_len)). All L logits are scored against
//     targets `[t_0, ..., t_{L-1}]`. The last logit predicts a token that is
//     NOT in the input — that's the model's "next token after the input"
//     distribution, evaluated against the actual next token from the doc.
//
//   - Each subsequent window slides forward by `pred_len = maxLen` newly
//     scored tokens. Its input is `tokens[window_end - maxLen - 1 : window_end - 1]`
//     (length maxLen, including 1 token of carry-over context from the prior
//     window) and its targets are `tokens[window_end - pred_len : window_end]`.
//     All maxLen logits are scored — every prediction has up to maxLen tokens
//     of preceding context.
//
//   - The very last window of a doc has fewer new targets than `pred_len`. In
//     that case the input is *padded backwards* into earlier doc tokens to
//     keep the input length at maxLen, but only the last `windowPredLen` logits
//     are scored. This is the critical detail: the prior algorithm in this
//     file truncated the last-window input instead of padding back, which
//     starved the early predictions of context and inflated NLL by ~13%.
//
// Returns (nll_sum, nll2_sum, scored_tokens).
func scoreDocumentHarness(m base.Model, tokens []int32, opts PPLOptions) (float64, float64, int, error) {
	if len(tokens) == 0 {
		return 0, 0, 0, nil
	}

	maxLen := opts.MaxLength
	predLen := maxLen // context_len=1 → pred_len = maxLen - context_len + 1 = maxLen

	var (
		totalNLL  float64
		totalNLL2 float64
		totalTok  int
	)

	// First window: input = [prefix] + tokens[:firstSeqLen-1], targets = tokens[:firstSeqLen]
	firstSeqLen := min(maxLen, len(tokens))
	var firstInput []int32
	if opts.PrefixTokenID >= 0 {
		firstInput = make([]int32, 0, firstSeqLen)
		firstInput = append(firstInput, opts.PrefixTokenID)
		firstInput = append(firstInput, tokens[:firstSeqLen-1]...)
	} else {
		// No prefix — can only score firstSeqLen-1 targets (positions 1..firstSeqLen-1
		// of the input predict tokens 1..firstSeqLen-1; logit[firstSeqLen-1] would
		// predict tokens[firstSeqLen] which may or may not exist).
		firstInput = tokens[:firstSeqLen]
	}
	firstTargets := tokens[:firstSeqLen]
	nll, nll2, scored, err := scoreLastN(m, firstInput, firstTargets, len(firstTargets))
	if err != nil {
		return 0, 0, 0, err
	}
	totalNLL += nll
	totalNLL2 += nll2
	totalTok += scored
	predicted := firstSeqLen
	mlx.Sweep()
	mlx.ClearCache()

	// Subsequent windows.
	for predicted < len(tokens) {
		windowPredLen := min(len(tokens)-predicted, predLen)
		windowEnd := predicted + windowPredLen

		inputStart := max(windowEnd-maxLen-1, 0)
		input := tokens[inputStart : windowEnd-1]
		targets := tokens[windowEnd-windowPredLen : windowEnd]

		nll, nll2, scored, err := scoreLastN(m, input, targets, len(targets))
		if err != nil {
			return 0, 0, 0, err
		}
		totalNLL += nll
		totalNLL2 += nll2
		totalTok += scored
		predicted += windowPredLen
		mlx.Sweep()
		mlx.ClearCache()
	}

	return totalNLL, totalNLL2, totalTok, nil
}

// scoreLastN runs one forward pass on `input` and computes the negative
// log-likelihood for the LAST `numTargets` positions. The model produces
// logits at every input position; logit[i] is interpreted as the model's
// distribution for the token that follows input[i] in the source sequence.
// `targets` must contain exactly numTargets tokens, aligned so that
// targets[k] is the prediction target for logit[inplen - numTargets + k].
//
// This shape (rather than "score positions [scoreFrom..N-1] of an
// in-window slice") is what lets the caller use a back-padded window for
// the last-window-of-doc case: input can be longer than the targets so the
// model gets full preceding context, while only the last numTargets logits
// contribute to the NLL.
func scoreLastN(m base.Model, input, targets []int32, numTargets int) (float64, float64, int, error) {
	inplen := len(input)
	if inplen < 1 || numTargets < 1 || numTargets > inplen {
		return 0, 0, 0, nil
	}
	if len(targets) != numTargets {
		return 0, 0, 0, fmt.Errorf("scoreLastN: targets length %d != numTargets %d", len(targets), numTargets)
	}

	tokens := mlx.FromValues(input, 1, inplen)

	caches := m.(interface{ NewCaches() []cache.Cache }).NewCaches()
	// IMPORTANT: free per-window KV caches before returning. Several model
	// implementations pin their cache k/v tensors internally so the global
	// Sweep won't release them — without an explicit Free() they accumulate
	// across windows and run the system out of memory long before the doc
	// stream finishes.
	defer func() {
		for _, c := range caches {
			if c != nil {
				c.Free()
			}
		}
	}()

	h := m.Forward(tokens, caches)

	// Slice the hidden states down to the positions we'll actually score, so
	// the unembed step only produces logits for those rows. For typical
	// "score every position" windows this is a no-op; for the last window of
	// a doc with windowPredLen << inplen it's a real saving.
	hScored := h.Slice(mlx.Slice(), mlx.Slice(inplen-numTargets, inplen), mlx.Slice())
	mlx.Eval(hScored)
	mlx.Pin(hScored)
	defer mlx.Unpin(hScored)

	return scoreHiddenChunked(m, hScored, targets, numTargets)
}

// scoreHiddenChunked runs the unembed + log-softmax + gather pipeline over a
// sliced hidden-state tensor in chunks along the sequence axis, so the peak
// per-window allocation never exceeds chunkSeqLen * vocab * 4 bytes (the
// transient f32 logits chunk). This matters at large vocab (~250k) and full
// 2048-token windows where the un-chunked approach materializes a ~2 GB f32
// tensor and pushes the working set close to OOM on heavily-loaded systems.
func scoreHiddenChunked(m base.Model, hScored *mlx.Array, targets []int32, numTargets int) (float64, float64, int, error) {
	// Picked to keep one chunk's f32 logits well under 512 MB even at
	// vocab=250k (256 * 250k * 4 = 256 MB). Larger chunks are faster but
	// raise peak memory; 256 is a comfortable balance on 128 GB systems.
	const chunkSeqLen = 256

	var totalNLL, totalNLL2 float64

	for k := 0; k < numTargets; k += chunkSeqLen {
		end := min(k+chunkSeqLen, numTargets)
		chunkLen := end - k

		hChunk := hScored.Slice(mlx.Slice(), mlx.Slice(k, end), mlx.Slice())
		logitsChunk := m.Unembed(hChunk)
		logitsChunkF32 := logitsChunk.AsType(mlx.DTypeFloat32)

		lseChunk := mlx.LogSumexpAxis(logitsChunkF32, 2, true) // [1, chunkLen, 1]

		targetIDs := make([]int32, chunkLen)
		copy(targetIDs, targets[k:end])
		targetsArr := mlx.FromValues(targetIDs, 1, chunkLen, 1)

		targetLogits := logitsChunkF32.TakeAlongAxis(targetsArr, 2) // [1, chunkLen, 1]
		targetLogprobs := targetLogits.Subtract(lseChunk)           // [1, chunkLen, 1]

		flat := targetLogprobs.Squeeze(2).Squeeze(0).Negative() // [chunkLen]
		nllSum := flat.SumAxis(0, false)
		nll2Sum := flat.Multiply(flat).SumAxis(0, false)

		nllF32 := nllSum.AsType(mlx.DTypeFloat32)
		nll2F32 := nll2Sum.AsType(mlx.DTypeFloat32)
		mlx.Eval(nllF32, nll2F32)

		totalNLL += float64(nllF32.Floats()[0])
		totalNLL2 += float64(nll2F32.Floats()[0])

		// Free the chunk's transient f32 logits + LSE before the next chunk.
		mlx.Sweep()
		mlx.ClearCache()
	}

	return totalNLL, totalNLL2, numTargets, nil
}

// runLlamaCpp implements llama-perplexity's algorithm: concatenate all docs
// into one flat token stream, split into non-overlapping chunks of MaxLength,
// substitute BOS at the start of each chunk for the forward pass, and score
// only the second half of each chunk.
func runLlamaCpp(m base.Model, docs []Document, opts PPLOptions, log Logger) (PPLResult, error) {
	result := PPLResult{Mode: opts.Mode}

	// Concatenate.
	var stream []int32
	for _, d := range docs {
		stream = append(stream, d.TokenIDs...)
		result.TotalChars += len(d.Text)
		result.TotalWords += countWords(d.Text)
	}

	chunkSize := opts.MaxLength
	maxChunks := len(stream) / chunkSize
	if opts.MaxDocs > 0 && opts.MaxDocs < maxChunks {
		maxChunks = opts.MaxDocs
	}
	if maxChunks == 0 {
		return result, fmt.Errorf("not enough tokens for one chunk (need %d, have %d)", chunkSize, len(stream))
	}

	bosID := opts.PrefixTokenID
	if !opts.BOSSwapLlamaCpp {
		bosID = -1
	}

	for i := range maxChunks {
		chunk := stream[i*chunkSize : (i+1)*chunkSize]

		// Build forward-pass input with BOS at position 0 if requested.
		// Original chunk is preserved for the scoring targets.
		var window []int32
		if bosID >= 0 {
			window = make([]int32, chunkSize)
			copy(window, chunk)
			window[0] = bosID
		} else {
			window = chunk
		}

		// Score the second half: positions [chunkSize/2..chunkSize-1]
		// using ORIGINAL chunk as the target source.
		first := chunkSize / 2
		nll, nll2, scored, err := scoreSecondHalf(m, window, chunk, first)
		if err != nil {
			return result, fmt.Errorf("chunk %d: %w", i, err)
		}
		result.TotalNLL += nll
		result.TotalNLL2 += nll2
		result.TotalTokens += scored

		if (i+1)%opts.LogEvery == 0 || i == maxChunks-1 || i < 3 {
			running := math.Exp(result.TotalNLL / float64(result.TotalTokens))
			log.Logf("  [%d/%d] running token PPL=%.4f", i+1, maxChunks, running)
		}
		mlx.Sweep()
		mlx.ClearCache()
	}

	finalize(&result)
	return result, nil
}

// scoreSecondHalf is the llama-perplexity scoring kernel: predict
// chunk[first..n-1] using logits at window[first-1..n-2]. The window is the
// (possibly BOS-substituted) input to the forward pass; chunk is the original
// untouched targets.
func scoreSecondHalf(m base.Model, window, chunk []int32, first int) (float64, float64, int, error) {
	n := len(window)
	if first < 1 || first >= n {
		return 0, 0, 0, fmt.Errorf("invalid first=%d for window of length %d", first, n)
	}

	tokens := mlx.FromValues(window, 1, n)
	caches := m.(interface{ NewCaches() []cache.Cache }).NewCaches()
	defer func() {
		for _, c := range caches {
			if c != nil {
				c.Free()
			}
		}
	}()
	h := m.Forward(tokens, caches)

	// Slice hidden states to the scored positions, then chunk the
	// unembed/log-softmax/gather pipeline. See scoreHiddenChunked for the
	// memory rationale.
	numScored := n - first
	hScored := h.Slice(mlx.Slice(), mlx.Slice(first-1, n-1), mlx.Slice())
	mlx.Eval(hScored)
	mlx.Pin(hScored)
	defer mlx.Unpin(hScored)

	return scoreHiddenChunked(m, hScored, chunk[first:n], numScored)
}

// finalize computes the aggregate metrics from the running totals.
//
//	token PPL = exp(NLL_total / N_tokens)
//	stderr_PPL = stderr_NLL * PPL  via delta method
//	word PPL = exp(NLL_total / N_words)
//	byte PPL = exp(NLL_total / N_bytes)
//	bits/byte = (NLL_total / N_bytes) / ln(2)
func finalize(r *PPLResult) {
	if r.TotalTokens > 1 {
		meanNLL := r.TotalNLL / float64(r.TotalTokens)
		r.TokenPerplexity = math.Exp(meanNLL)
		varNLL := r.TotalNLL2/float64(r.TotalTokens) - meanNLL*meanNLL
		if varNLL > 0 {
			stderrNLL := math.Sqrt(varNLL / float64(r.TotalTokens-1))
			r.StderrTokenPPL = stderrNLL * r.TokenPerplexity
		}
	}
	if r.TotalWords > 0 {
		r.WordPerplexity = math.Exp(r.TotalNLL / float64(r.TotalWords))
	}
	if r.TotalChars > 0 {
		r.BytePerplexity = math.Exp(r.TotalNLL / float64(r.TotalChars))
		r.BitsPerByte = (r.TotalNLL / float64(r.TotalChars)) / math.Ln2
	}
}

// countWords counts whitespace-separated chunks in s, matching Python's
// `re.split(r"\s+", s)` semantics. This is what
// lm-evaluation-harness/preprocess_wikitext uses for word_perplexity:
//
//	_words = len(re.split(r"\s+", doc["page"]))
//
// Note: Python's re.split with a pattern includes an empty string at the
// start of the result if the input begins with whitespace, and at the end
// if it ends with whitespace. We replicate that exactly so word counts match.
func countWords(s string) int {
	// re.split with `\s+` produces (number of `\s+` runs) + 1 chunks,
	// including empty strings at the start/end when s begins or ends
	// with whitespace. We replicate that exactly: start at 1, increment
	// once per whitespace run.
	count := 1
	inWS := false
	for _, r := range s {
		ws := r == ' ' || r == '\t' || r == '\n' || r == '\r' || r == '\v' || r == '\f'
		if ws && !inWS {
			count++
		}
		inWS = ws
	}
	return count
}
