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
	"github.com/ollama/ollama/x/mlxrunner/model/base"
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

	var tokens []int32
	var items []mediaItem
	if len(request.Media) == 0 {
		tokens = r.Tokenizer.Encode(request.Prompt, r.Tokenizer.AddBOS())
	} else {
		mm, ok := r.Model.(base.MediaModel)
		if !ok {
			kind := string(request.Media[0].Kind)
			if kind == "" {
				kind = "media"
			}
			return fmt.Errorf("this model does not support %s input", kind)
		}
		prepared, bound, err := r.expandMedia(mm, request.Prompt, request.Media)
		if err != nil {
			return err
		}
		tokens, items = prepared.Tokens, bound
		request.Layout = prepared.Layout
	}

	if len(tokens) == 0 {
		return errors.New("empty prompt")
	}

	if len(tokens) >= r.contextLength {
		return fmt.Errorf("input length (%d tokens) exceeds the model's maximum context length (%d tokens)", len(tokens), r.contextLength)
	}

	request.Options.NumPredict = generationBudget(request.Options.NumPredict, request.Options.NumCtx, r.contextLength, len(tokens))

	request.Tokens = tokens
	request.MediaItems = items
	return nil
}

// generationBudget resolves how many tokens a request may generate. The
// model's context length bounds every request, and an open-ended num_predict
// is additionally bounded by the context window the request was scheduled
// with, the same way llama-server's runner bounds it.
//
// contextLength is the checkpoint's max_position_embeddings, which on a large
// model is 128k tokens or more. Without the num_ctx bound an open-ended
// request inherits that whole span as its budget, so a checkpoint that talks
// past its stop token runs for hours and grows the KV cache well past the
// context window the server advertises for the runner — the request reads as
// an indefinite hang rather than as a completion.
func generationBudget(numPredict, numCtx, contextLength, promptLen int) int {
	maxGenerate := contextLength - promptLen
	numPredict = llm.BoundedNumPredict(numPredict, numCtx)
	if numPredict <= 0 {
		return maxGenerate
	}
	return min(numPredict, maxGenerate)
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

	session := r.cache.begin(inputs, request.MediaItems)
	defer session.close()
	caches := session.caches

	media := r.openMedia(request)
	defer media.close()

	// Built before prefill so a drafter with draft caches follows the prompt
	// through prefill alongside the target.
	spec := r.spec.open(request, media.rowLayout())
	defer spec.close()

	seed, position, promptEval, err := r.prefill(ctx, session, spec, media)
	if err != nil {
		return err
	}

	// Register the sampler after prefill completes.
	r.Sampler.Add(pipelineSlot, request.SamplerOpts, inputs)

	var d decoder
	if spec != nil {
		d = spec.decoder(seed, position)
	} else {
		d = r.pipelinedDecoder(nil, caches, seed.ExpandDims(-1), position, media.rowLayout())
	}
	defer d.close()
	return r.decode(ctx, request, session, d, promptEval)
}

// prefill evaluates the prompt in chunks, leaving one token for decode to
// seed from, and schedules the prompt's periodic snapshots. It returns the
// seed token, the resume position, and the prompt-evaluation duration.
func (r *Runner) prefill(ctx context.Context, session *cacheSession, spec *speculationSession, media *requestMedia) (*mlx.Array, int, time.Duration, error) {
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
			if c == nil {
				continue
			}
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
	// Free restored items' buffers now: on a full cache hit the loop never runs.
	media.release(position)
	for total-processed > 1 {
		if err := ctx.Err(); err != nil {
			return nil, 0, 0, err
		}

		n := min(prefillChunk, total-processed-1)
		n = media.extendChunk(position, n)

		chunkIDs := mlx.FromValues(tokens[processed:processed+n], 1, n)
		manifest := media.batchMedia(position, n)
		_, auxHidden := r.Model.Forward(&batch.Batch{
			InputIDs:     chunkIDs,
			SeqOffsets:   []int32{int32(position)},
			SeqQueryLens: []int32{int32(n)},
			Media:        manifest,
			Layout:       media.rowLayout(),
		}, caches)
		// Report to the drafter only after the chunk's eval: a draft flush
		// evaluates, and an eval before the sweep cannot free any buffer the
		// chunk's live handles retain — on media chunks, the whole vision tower.
		mlx.Pin(chunkIDs, auxHidden)
		mlx.Sweep()
		materializeCaches()
		spec.committed(chunkIDs, auxHidden, position, manifest)
		mlx.Unpin(chunkIDs, auxHidden)
		// Released after committed so the drafter can capture rows its
		// deferred flush still embeds.
		media.release(position + n)
		processed += n
		position += n
		slog.Info("Prompt processing progress", "processed", processed, "total", total)
		logutil.TraceContext(ctx, "mlx prompt forward", "processed", processed, "total", total, "tokens", n, "memory", mlx.Memory{})

		mlx.ClearCache()
	}

	// Settle before attaching: snapshots attach only at offsets every cache
	// has crossed, and the draft caches stay a pair short of the target
	// until the seed completes the frontier pair.
	seed := mlx.FromValues(tokens[processed:], 1)
	spec.settle(seed)
	session.attachPrefillSnapshots()

	return seed, position, time.Since(start), nil
}

// A decoder produces each run of tokens to emit, owning its own dispatch and
// synchronization; the decode loop owns the budget, emission, and
// cancellation. next may return none while its first tokens are in flight.
type decoder interface {
	next(remaining int) ([]sampler.Result, error)

	// drain ends production, returning any results sampled but never
	// delivered through next and the position the next forward would have
	// taken; the decoder remains closeable.
	drain() ([]sampler.Result, int)

	close()
}

// decode drives either decoder and owns where generation stops — at an EOS
// or the NumPredict budget. Every produced token is recorded so the caches
// never rest ahead of session.outputs; tokens past the stop are recorded but
// not streamed or counted.
func (r *Runner) decode(ctx context.Context, request Request, session *cacheSession, d decoder, promptEval time.Duration) error {
	// A sampled-but-undelivered result is still a produced token; record it.
	defer func() {
		results, _ := d.drain()
		for _, res := range results {
			session.outputs = append(session.outputs, int32(res.Token.Int()))
		}
	}()

	detok := detokenizer{
		tokenizer:       r.Tokenizer,
		wantLogprobs:    request.SamplerOpts.Logprobs,
		wantTopLogprobs: request.SamplerOpts.TopLogprobs,
	}

	final := CompletionResponse{Done: true, PromptEvalCount: len(request.Tokens), DoneReason: 1}
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
	layout   []any // the request's per-row layout state, stamped on every forward
	position int
	sample   sampler.Result // in flight: sampled, not yet forwarded
}

func (r *Runner) pipelinedDecoder(spec *speculationSession, caches []cache.Cache, seed *mlx.Array, position int, layout []any) *pipelinedDecoder {
	t := &pipelinedDecoder{r: r, spec: spec, caches: caches, layout: layout, position: position}
	t.sample = t.dispatch(seed)
	return t
}

// dispatch builds one forward-and-sample chain without reading the token's
// value, so it is in flight before the previous token is synchronized.
func (t *pipelinedDecoder) dispatch(token *mlx.Array) sampler.Result {
	r := t.r
	hidden, auxHidden := r.Model.Forward(&batch.Batch{
		InputIDs:     token,
		SeqOffsets:   []int32{int32(t.position)},
		SeqQueryLens: []int32{int32(token.Dim(1))},
		Layout:       t.layout,
	}, t.caches)
	t.spec.committed(token, auxHidden, t.position, nil)
	t.position += token.Dim(1)
	logits := r.Model.Unembed(hidden)
	next := r.Sampler.Sample([]int{pipelineSlot}, logits.Slice(mlx.Slice(), mlx.Slice(logits.Dim(1)-1), mlx.Slice()).Squeeze(1))
	mlx.Pin(next.Arrays()...)
	mlx.Sweep()
	mlx.AsyncEval(next.Arrays()...)
	return next
}

func (t *pipelinedDecoder) next(int) ([]sampler.Result, error) {
	out := t.sample
	t.sample = t.dispatch(out.Token.ExpandDims(-1))
	mlx.Unpin(out.Arrays()...)
	return []sampler.Result{out}, nil
}

// drain ends production: it returns the in-flight sample (sampled but never
// forwarded) and the position its forward would have taken. The decoder
// keeps the sample for close.
func (t *pipelinedDecoder) drain() ([]sampler.Result, int) {
	return []sampler.Result{t.sample}, t.position
}

func (t *pipelinedDecoder) close() {
	// The in-flight sample's forward was never dispatched; its report settles
	// the drafter level with the caches' resting offset.
	t.spec.settle(t.sample.Token)
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
