package mlxrunner

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"time"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	sampler "github.com/ollama/ollama/x/mlxrunner/sample"
	"github.com/ollama/ollama/x/tokenizer"
)

func prefillChunkSize() int {
	return 2 << 10
}

// Prepare tokenizes the prompt and validates it against the model's
// context length. It is safe to call from any goroutine. On success it
// populates request.Tokens and adjusts request.Options.NumPredict.
func (r *Runner) Prepare(request *Request) error {
	if r.Model == nil {
		return errors.New("model not loaded")
	}

	tokens := r.Tokenizer.Encode(request.Prompt, r.Tokenizer.AddBOS())
	if len(tokens) == 0 {
		return errors.New("empty prompt")
	}

	if len(tokens) >= r.contextLength {
		return fmt.Errorf("input length (%d tokens) exceeds the model's maximum context length (%d tokens)", len(tokens), r.contextLength)
	}

	// Cap generation to stay within the model's context length
	maxGenerate := r.contextLength - len(tokens)
	if request.Options.NumPredict <= 0 {
		request.Options.NumPredict = maxGenerate
	} else {
		request.Options.NumPredict = min(request.Options.NumPredict, maxGenerate)
	}

	request.Tokens = tokens
	return nil
}

// The runner serializes requests today so we just use a fixed slot ID.
const pipelineSlot = 0

func (r *Runner) TextGenerationPipeline(ctx context.Context, request Request) error {
	mlx.ResetPeakMemory()

	defer func() {
		r.Sampler.Remove(pipelineSlot)
		mlx.Sweep()
		mlx.ClearCache()

		if slog.Default().Enabled(context.TODO(), logutil.LevelTrace) {
			mlx.LogArrays()
			r.cache.dumpTree()
		}
		slog.Info("peak memory", "size", mlx.PrettyBytes(mlx.PeakMemory()))
	}()

	inputs := request.Tokens

	session := r.cache.begin(r.Model, inputs)
	defer session.close()
	caches := session.caches

	// Built before prefill so a drafter with draft caches follows the prompt
	// through prefill alongside the target.
	spec := r.spec.open(request, caches)
	defer spec.close()

	seed, position, promptEval, err := r.prefill(ctx, session, spec)
	if err != nil {
		return err
	}

	// Register the sampler after prefill completes.
	r.Sampler.Add(pipelineSlot, request.SamplerOpts, inputs)

	var d decoder
	if spec != nil {
		d = spec.decoder(seed, position)
	} else {
		d = r.pipelinedDecoder(nil, caches, mlx.FromValues(seed, 1, len(seed)), position)
	}
	defer d.close()
	return r.decode(ctx, request, session, d, promptEval)
}

// prefill evaluates the prompt in chunks, leaving one token for decode to
// seed from, and schedules the prompt's periodic snapshots. It returns the
// seed tokens, the resume position, and the prompt-evaluation duration.
func (r *Runner) prefill(ctx context.Context, session *cacheSession, spec *speculationSession) ([]int32, int, time.Duration, error) {
	start := time.Now()
	inputs := session.inputs
	tokens := session.remaining
	caches := session.caches
	prefillChunk := prefillChunkSize()

	// Request periodic snapshots during prefill and near the end of the
	// prompt so that long prompts can be partially restored and
	// thinking/generation can be retried without full reprocessing.
	const snapshotInterval = 8192
	var snapshotOffsets []int
	for offset := snapshotInterval; offset < len(inputs); offset += snapshotInterval {
		snapshotOffsets = append(snapshotOffsets, offset)
	}

	const preThinking = 4
	if end := len(inputs) - preThinking; end > 0 {
		snapshotOffsets = append(snapshotOffsets, end)
	}

	materializeCaches := func() {
		state := make([]*mlx.Array, 0, 2*len(caches))
		for _, c := range caches {
			state = append(state, c.State()...)
		}
		if len(state) == 0 {
			return
		}
		mlx.Eval(state...)
	}

	session.schedulePrefillSnapshots(snapshotOffsets)

	total, processed := len(tokens), 0
	position := len(inputs) - len(tokens)
	for total-processed > 1 {
		if err := ctx.Err(); err != nil {
			return nil, 0, 0, err
		}

		n := min(prefillChunk, total-processed-1)

		chunkIDs := mlx.FromValues(tokens[processed:processed+n], 1, n)
		hidden := r.Model.Forward(&batch.Batch{
			InputIDs:     chunkIDs,
			SeqOffsets:   []int32{int32(position)},
			SeqQueryLens: []int32{int32(n)},
		}, caches)
		spec.committed(chunkIDs, hidden, position)
		mlx.Sweep()
		materializeCaches()
		processed += n
		position += n
		slog.Info("Prompt processing progress", "processed", processed, "total", total)
		logutil.TraceContext(ctx, "mlx prompt forward", "processed", processed, "total", total, "tokens", n, "memory", mlx.Memory{})

		mlx.ClearCache()
	}

	// Flush before attaching: snapshots attach only at offsets every cache
	// has crossed, and a drafter with draft caches keeps buffered pairs that
	// would otherwise hold those caches short of the scheduled offsets.
	spec.flush()
	session.attachPrefillSnapshots()

	return tokens[processed:], position, time.Since(start), nil
}

// A decoder produces each run of tokens to emit, owning its own dispatch and
// synchronization; the decode loop owns the budget, emission, and
// cancellation. next may return none while its first tokens are in flight.
type decoder interface {
	next(remaining int) ([]sampler.Result, error)
	close()
}

// decode drives either decoder and owns where generation stops — at an EOS
// or the NumPredict budget. Every produced token is recorded so the caches
// never rest ahead of session.outputs; tokens past the stop are recorded but
// not streamed or counted.
func (r *Runner) decode(ctx context.Context, request Request, session *cacheSession, d decoder, promptEval time.Duration) error {
	detok := detokenizer{
		tokenizer:       r.Tokenizer,
		wantLogprobs:    request.SamplerOpts.Logprobs,
		wantTopLogprobs: request.SamplerOpts.TopLogprobs,
	}

	final := CompletionResponse{
		Done:                  true,
		PromptEvalCount:       len(request.Tokens),
		PromptEvalCachedCount: len(session.inputs) - len(session.remaining),
		DoneReason:            1,
	}
	final.PromptEvalDuration = promptEval
	now := time.Now()

	// Release MLX's cached free buffers every clearCacheInterval tokens so the
	// allocator's pool does not grow unbounded over a long generation.
	const clearCacheInterval = 256

	generated := 0
	for generated < request.Options.NumPredict {
		if err := ctx.Err(); err != nil {
			return err
		}

		results, err := d.next(request.Options.NumPredict - generated)
		if err != nil {
			return err
		}

		// Record the whole run before streaming any of it: a cancelled
		// stream returns early and must not leave the caches ahead of
		// session.outputs.
		done := false
		stream := len(results)
		for i, res := range results {
			// Int evaluates the array before reading it; a raw data read
			// on a lazy array races its evaluation and returns garbage.
			id := int32(res.Token.Int())
			session.outputs = append(session.outputs, id)
			if done {
				continue
			}
			if r.Tokenizer.IsEOS(id) {
				final.DoneReason = 0
				done = true
				stream = i
				continue
			}
			generated++
			if generated >= request.Options.NumPredict {
				done = true
				stream = i + 1
			}
		}

		for _, res := range results[:stream] {
			resp, ok := detok.detokenize(res)
			if !ok {
				continue
			}
			select {
			case <-ctx.Done():
				return ctx.Err()
			case request.Responses <- resp:
			}
		}

		if done {
			break
		}

		if generated%clearCacheInterval == 0 {
			mlx.ClearCache()
		}
	}

	final.EvalCount = generated
	final.EvalDuration = time.Since(now)
	select {
	case <-ctx.Done():
		return ctx.Err()
	case request.Responses <- final:
		return nil
	}
}

// pipelinedDecoder decodes one token per call, one call ahead of emission:
// the next token's chain is dispatched before the returned one is
// synchronized, so the device runs ahead of host emission.
type pipelinedDecoder struct {
	r *Runner
	// spec, when non-nil, receives every forwarded token and settles its
	// drafter at close, keeping a non-drafting session's draft KV level.
	spec     *speculationSession
	caches   []cache.Cache
	position int
	sample   sampler.Result // in flight: sampled, not yet forwarded
	emitted  sampler.Result // last call's result, pinned until the next call
}

func (r *Runner) pipelinedDecoder(spec *speculationSession, caches []cache.Cache, seed *mlx.Array, position int) *pipelinedDecoder {
	t := &pipelinedDecoder{r: r, spec: spec, caches: caches, position: position}
	t.sample = t.dispatch(seed)
	return t
}

// dispatch builds one forward-and-sample chain without reading the token's
// value, so it is in flight before the previous token is synchronized.
func (t *pipelinedDecoder) dispatch(token *mlx.Array) sampler.Result {
	r := t.r
	hidden := r.Model.Forward(&batch.Batch{
		InputIDs:     token,
		SeqOffsets:   []int32{int32(t.position)},
		SeqQueryLens: []int32{int32(token.Dim(1))},
	}, t.caches)
	t.spec.committed(token, hidden, t.position)
	t.position += token.Dim(1)
	logits := r.Model.Unembed(hidden)
	next := r.Sampler.Sample([]int{pipelineSlot}, logits.Slice(mlx.Slice(), mlx.Slice(logits.Dim(1)-1), mlx.Slice()).Squeeze(1))
	mlx.Pin(next.Arrays()...)
	mlx.Sweep()
	mlx.AsyncEval(next.Arrays()...)
	return next
}

func (t *pipelinedDecoder) next(int) ([]sampler.Result, error) {
	mlx.Unpin(t.emitted.Arrays()...)
	t.emitted, t.sample = t.sample, t.dispatch(t.sample.Token.ExpandDims(-1))
	return []sampler.Result{t.emitted}, nil
}

// detach ends a parked stretch: it hands the in-flight sample (sampled but
// never forwarded) and the resume position to the caller, releasing the
// decoder's emitted pin but not settling the drafter. The caller drafts from
// the sample next, and that round completes the still-open frontier pair.
func (t *pipelinedDecoder) detach() (sampler.Result, int) {
	mlx.Unpin(t.emitted.Arrays()...)
	return t.sample, t.position
}

func (t *pipelinedDecoder) close() {
	// The in-flight sample's forward was never dispatched; its report settles
	// the drafter level with the caches' resting offset.
	t.spec.finish(t.sample.Token)
	mlx.Unpin(t.emitted.Arrays()...)
	mlx.Unpin(t.sample.Arrays()...)
}

// detokenizer serializes sampled tokens into response chunks, holding bytes
// whose UTF-8 sequence hasn't completed yet and the logprobs that belong
// with those bytes so Content and Logprobs stay aligned when a chunk does
// flush.
type detokenizer struct {
	tokenizer       *tokenizer.Tokenizer
	buf             bytes.Buffer
	logprobs        []llm.Logprob
	wantLogprobs    bool
	wantTopLogprobs int
}

func (d *detokenizer) detokenize(res sampler.Result) (CompletionResponse, bool) {
	output := int32(res.Token.Int())
	d.buf.WriteString(d.tokenizer.Decode([]int32{output}))
	d.logprobs = append(d.logprobs, buildLogprob(res, d.wantLogprobs, d.wantTopLogprobs, d.tokenizer.Decode)...)

	content := flushValidUTF8Prefix(&d.buf)
	if content == "" {
		return CompletionResponse{}, false
	}
	resp := CompletionResponse{Content: content, Logprobs: d.logprobs}
	d.logprobs = nil
	return resp, true
}

// buildLogprob converts the sampler's logprob tensors into the wire-format
// llm.Logprob entries the caller wants. The sampler populates its logprob
// tensors whenever any registered slot requested them, so the caller must
// gate emission on its own request config (wantLogprobs / wantTopLogprobs)
// rather than on whether the tensors happen to be non-nil.
func buildLogprob(sample sampler.Result, wantLogprobs bool, wantTopLogprobs int, decode func([]int32) string) []llm.Logprob {
	if !wantLogprobs || sample.Logprob == nil {
		return nil
	}
	tok := func(id int32) string { return decode([]int32{id}) }

	out := llm.Logprob{
		TokenLogprob: llm.TokenLogprob{
			Token:   tok(int32(sample.Token.Int())),
			Logprob: float64(sample.Logprob.Floats()[0]),
		},
	}

	if wantTopLogprobs > 0 && sample.TopTokens != nil {
		ids := sample.TopTokens.Ints()
		vals := sample.TopLogprobs.Floats()
		pairs := make([]llm.TokenLogprob, len(ids))
		for i, id := range ids {
			pairs[i] = llm.TokenLogprob{
				Token:   tok(int32(id)),
				Logprob: float64(vals[i]),
			}
		}
		// The sampler emits the top maxK across registered slots via
		// Argpartition, which leaves entries unsorted.
		sort.Slice(pairs, func(i, j int) bool {
			return pairs[i].Logprob > pairs[j].Logprob
		})
		if wantTopLogprobs < len(pairs) {
			pairs = pairs[:wantTopLogprobs]
		}
		out.TopLogprobs = pairs
	}
	return []llm.Logprob{out}
}
