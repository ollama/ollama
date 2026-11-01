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
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	sampler "github.com/ollama/ollama/mlxrunner/sample"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

func prefillChunkSize() int {
	return 2 << 10
}

// Prepare tokenizes the prompt and validates it against the model's
// context length. It is safe to call from any goroutine. On success it
// populates request.Tokens and adjusts request.Options.NumPredict.
func (r *Runner) Prepare(request *Request) (err error) {
	if r.Model == nil {
		return errors.New("model not loaded")
	}

	// Launched first so the compile overlaps tokenization and media
	// preparation as well as prefill.
	grammar, err := r.grammarEngine.prepare(request.Format)
	if err != nil {
		return err
	}
	request.Grammar = grammar
	defer func() {
		if err != nil {
			request.Grammar.close()
			request.Grammar = nil
		}
	}()

	var tokens []int32
	var items []mediaItem
	if len(request.Media) == 0 {
		tokens = r.Tokenizer.Encode(request.Prompt, r.Tokenizer.AddBOS())
	} else {
		mm, ok := r.Model.(model.MediaModel)
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

	// Cap generation to stay within the model's context length
	maxGenerate := r.contextLength - len(tokens)
	if request.Options.NumPredict <= 0 {
		request.Options.NumPredict = maxGenerate
	} else {
		request.Options.NumPredict = min(request.Options.NumPredict, maxGenerate)
	}

	request.Tokens = tokens
	request.MediaItems = items
	return nil
}

// The serial pipeline uses one stable sampling slot.
const pipelineSlot = 0

// skipCleanupOnPanic keeps a backend failure from being replaced by cleanup
// that evaluates or saves partially updated cache state.
func skipCleanupOnPanic(cleanup func()) {
	if failure := recover(); failure != nil {
		panic(failure)
	}
	cleanup()
}

func (r *Runner) TextGenerationPipeline(ctx context.Context, request Request) (err error) {
	mlx.ResetPeakMemory()
	mlx.Scoped(func() { err = r.generate(ctx, request) })
	mlx.ClearCache()

	if slog.Default().Enabled(context.TODO(), logutil.LevelTrace) {
		r.cache.dumpTree()
	}
	slog.Info("memory", "peak", mlx.PrettyBytes(mlx.PeakMemory()), "held", mlx.PrettyBytes(mlx.ActiveMemory()))
	return err
}

func (r *Runner) generate(ctx context.Context, request Request) error {
	inputs := request.Tokens

	session := r.cache.begin(inputs, request.MediaItems)
	defer skipCleanupOnPanic(session.close)
	caches := session.caches

	media := r.openMedia(request)
	defer skipCleanupOnPanic(media.close)

	// Built before prefill so a drafter with draft caches follows the prompt
	// through prefill alongside the target.
	spec := r.spec.open(request, media.rowLayout())
	defer skipCleanupOnPanic(spec.close)

	seed, position, promptEval, err := r.prefill(ctx, session, spec, media)
	if err != nil {
		return err
	}

	// Register the sampler after prefill completes.
	r.Sampler.Add(pipelineSlot, request.SamplerOpts, inputs)
	defer r.Sampler.Remove(pipelineSlot)

	grammar, err := request.Grammar.resolve(ctx)
	if err != nil {
		return err
	}

	var d decoder
	if spec != nil {
		d = spec.decoder(seed, position, grammar)
	} else {
		d = r.pipelinedDecoder(pipelineSlot, nil, caches, seed.ExpandDims(-1), position, media.rowLayout(), grammar)
	}
	defer skipCleanupOnPanic(d.close)
	return r.decode(ctx, request, session, d, promptEval)
}

// prefillCursor advances one prompt chunk at a time so decoding can run between
// chunks. Both the serial pipeline and the batch scheduler use this path.
type prefillCursor struct {
	r                   *Runner
	session             *cacheSession
	spec                *speculationSession
	media               *requestMedia
	processed, position int
	started             time.Time
}

func (r *Runner) newPrefill(session *cacheSession, spec *speculationSession, media *requestMedia) *prefillCursor {
	p := &prefillCursor{
		r: r, session: session, spec: spec, media: media,
		position: len(session.inputs) - len(session.remaining), started: time.Now(),
	}
	var offsets []int
	for offset := prefillSnapshotInterval; offset < len(session.inputs); offset += prefillSnapshotInterval {
		offsets = append(offsets, offset)
	}
	const preThinking = 4
	if end := len(session.inputs) - preThinking; end > 0 {
		offsets = append(offsets, end)
	}
	session.schedulePrefillSnapshots(offsets)
	media.free(p.position)
	return p
}

func (p *prefillCursor) step(ctx context.Context) (bool, error) {
	tokens := p.session.remaining
	if err := ctx.Err(); err != nil {
		p.spec.settle(mlx.FromValues(tokens[p.processed:p.processed+1], 1))
		return false, err
	}
	if len(tokens)-p.processed <= 1 {
		return true, nil
	}
	n := p.media.extendChunk(p.position, min(prefillChunkSize(), len(tokens)-p.processed-1))
	mlx.Scoped(func() {
		ids := mlx.FromValues(tokens[p.processed:p.processed+n], 1, n)
		media := p.media.batchMedia(p.position, n)
		aux := mlx.ScopedArrays(func() []*mlx.Array {
			_, aux := p.r.Model.Forward(&batch.Batch{
				InputIDs:   ids,
				SeqOffsets: []int32{int32(p.position)}, SeqQueryLens: []int32{int32(n)},
				Media: media, Layout: p.media.rowLayout(),
			}, cache.Layers(p.session.caches))
			return []*mlx.Array{aux}
		})[0]
		var state []*mlx.Array
		for _, c := range p.session.caches {
			if c != nil {
				state = append(state, c.State()...)
			}
		}
		mlx.Eval(state...)
		p.spec.committed(ids, aux, p.position, media)
		p.media.free(p.position + n)
	})
	p.processed += n
	p.position += n
	slog.Info("Prompt processing progress", "processed", p.processed, "total", len(tokens))
	logutil.TraceContext(ctx, "mlx prompt forward", "processed", p.processed, "total", len(tokens), "tokens", n, "memory", mlx.Memory{})
	mlx.ClearCache()
	return len(tokens)-p.processed <= 1, nil
}

func (p *prefillCursor) seed() *mlx.Array {
	seed := mlx.FromValues(p.session.remaining[p.processed:], 1)
	p.spec.settle(seed)
	p.session.attachPrefillSnapshots()
	return seed
}

func (r *Runner) prefill(ctx context.Context, session *cacheSession, spec *speculationSession, media *requestMedia) (*mlx.Array, int, time.Duration, error) {
	p := r.newPrefill(session, spec, media)
	for {
		done, err := p.step(ctx)
		if err != nil {
			return nil, 0, 0, err
		}
		if done {
			return p.seed(), p.position, time.Since(p.started), nil
		}
	}
}

// A decoder produces each run of tokens to emit, owning its own dispatch and
// synchronization; the decode loop owns the budget, emission, and
// cancellation. next may return none while its first tokens are in flight.
type decoder interface {
	next(remaining int) ([]sampler.Result, error)

	// drain ends production, returning any results sampled but never
	// delivered through next and the position the next forward would have
	// taken, and surfaces the decoder's deferred fault; the decoder remains
	// closeable.
	drain() ([]sampler.Result, int, error)

	close()
}

// decode drives either decoder and owns where generation stops — at an EOS
// or the NumPredict budget. Every produced token is recorded so the caches
// never rest ahead of session.outputs; tokens past the stop are recorded but
// not streamed or counted.
func (r *Runner) decode(ctx context.Context, request Request, session *cacheSession, d decoder, promptEval time.Duration) error {
	// A sampled-but-undelivered result is still a produced token; record it.
	defer skipCleanupOnPanic(func() {
		results, _, _ := d.drain()
		for _, res := range results {
			session.outputs = append(session.outputs, res.Token.Int())
		}
	})

	output := newGenerationOutput(r, request, session, promptEval)
	for output.generated < request.Options.NumPredict {
		if err := ctx.Err(); err != nil {
			return err
		}
		var done bool
		var err error
		before := output.generated
		mlx.Scoped(func() {
			var results []sampler.Result
			results, err = d.next(request.Options.NumPredict - output.generated)
			if err != nil {
				return
			}
			var responses []CompletionResponse
			responses, done = output.consume(results)
			for _, response := range responses {
				select {
				case <-ctx.Done():
					err = ctx.Err()
					return
				case request.Responses <- response:
				}
			}
		})
		if err != nil {
			return err
		}
		if done {
			break
		}
		if output.generated/256 != before/256 {
			mlx.ClearCache()
		}
	}
	select {
	case <-ctx.Done():
		return ctx.Err()
	case request.Responses <- output.finish():
		return nil
	}
}

type generationOutput struct {
	r                *Runner
	session          *cacheSession
	detok            detokenizer
	limit, generated int
	started          time.Time
	final            CompletionResponse
}

func newGenerationOutput(r *Runner, request Request, session *cacheSession, promptEval time.Duration) *generationOutput {
	cached := len(session.inputs) - len(session.remaining)
	return &generationOutput{
		r: r, session: session, limit: request.Options.NumPredict, started: time.Now(),
		detok: detokenizer{tokenizer: r.Tokenizer, wantLogprobs: request.SamplerOpts.Logprobs, wantTopLogprobs: request.SamplerOpts.TopLogprobs},
		final: CompletionResponse{Done: true, DoneReason: 1, PromptEvalCount: len(request.Tokens), PromptEvalCachedCount: &cached, PromptEvalDuration: promptEval},
	}
}

func (o *generationOutput) consume(results []sampler.Result) ([]CompletionResponse, bool) {
	done, stream := false, len(results)
	// Record all produced tokens before emission, including speculative overshoot.
	for i, res := range results {
		id := res.Token.Int()
		o.session.outputs = append(o.session.outputs, id)
		if done {
			continue
		}
		if o.r.Tokenizer.IsEOS(id) {
			o.final.DoneReason = 0
			done = true
			stream = i
			continue
		}
		o.generated++
		if o.generated >= o.limit {
			done = true
			stream = i + 1
		}
	}
	var responses []CompletionResponse
	for _, res := range results[:stream] {
		if response, ok := o.detok.detokenize(res); ok {
			responses = append(responses, response)
		}
	}
	return responses, done
}

func (o *generationOutput) finish() CompletionResponse {
	o.final.EvalCount = o.generated
	o.final.EvalDuration = time.Since(o.started)
	return o.final
}

// pipelinedDecoder decodes one token per row per call, one call ahead of
// emission: the forward for the next tokens is dispatched before the
// returned ones are synchronized, so the device runs ahead of host
// emission. While no grammar constrains, the next sample is fused onto the
// forward's chain. A constraining grammar's sample needs a token mask that
// depends on the returned token's value, so only the forward runs ahead
// and the host's grammar work overlaps it.
type pipelinedDecoder struct {
	r      *Runner
	slotID int
	// spec, when non-nil, receives every forwarded token and settles its
	// drafter at close, keeping a non-drafting session's draft KV level.
	spec     *speculationSession
	caches   []cache.Cache
	layout   []any      // the request's per-row layout state, stamped on every forward
	grammars []*grammar // row i's grammar; nil rows are unconstrained
	position int
	pending  sampler.Result // in flight: sampled, not yet forwarded
	scope    *mlx.Scope     // holds pending across steps
	// Steps run ahead asynchronously: when one faults, its token is already
	// forwarded and still has to be returned, so err waits for the next call.
	err error
}

func (r *Runner) pipelinedDecoder(slotID int, spec *speculationSession, caches []cache.Cache, seed *mlx.Array, position int, layout []any, g *grammar) *pipelinedDecoder {
	t := &pipelinedDecoder{
		r: r, slotID: slotID, spec: spec, caches: caches, layout: layout, position: position,
		grammars: []*grammar{g}, scope: mlx.NewScope(),
	}
	logits := t.constrain(t.forward(seed), nil)
	t.pending = t.sample(logits)
	return t
}

// constrain overlaps the forward with host grammar work. A failure belongs to
// the next sample: the previous token has already been forwarded and is still
// returned to the caller.
func (t *pipelinedDecoder) constrain(logits, previous *mlx.Array) *mlx.Array {
	if !t.r.grammarEngine.hasGrammar(t.grammars) {
		return logits
	}
	mlx.AsyncEval(logits)
	var err error
	if previous != nil {
		err = t.failRows(t.r.grammarEngine.accept(t.grammars, previous.Ints()))
	}
	logits, errs := t.r.grammarEngine.mask(t.grammars, logits, nil)
	t.err = errors.Join(err, t.failRows(errs))
	return logits
}

// failRows clears the grammars of rows whose grammar work faulted — a dead
// row does no further grammar work — and joins their errors.
func (t *pipelinedDecoder) failRows(errs []error) error {
	for i, err := range errs {
		if err != nil {
			t.grammars[i] = nil
		}
	}
	return errors.Join(errs...)
}

func (t *pipelinedDecoder) next(int) ([]sampler.Result, error) {
	if t.err != nil {
		return nil, t.err
	}
	out := t.pending
	logits := t.constrain(t.forward(out.Token.ExpandDims(-1)), out.Token)

	t.pending = t.sample(logits)

	t.scope.Detach(out.Arrays()...)
	return []sampler.Result{out}, nil
}

// forward runs the model one step over token, shaped [B, L], and returns the
// final position's [B, 1, V] logits, still lazy.
func (t *pipelinedDecoder) forward(token *mlx.Array) *mlx.Array {
	return mlx.ScopedArrays(func() []*mlx.Array {
		hidden, auxHidden := t.r.Model.Forward(&batch.Batch{
			InputIDs:     token,
			SeqOffsets:   []int32{int32(t.position)},
			SeqQueryLens: []int32{int32(token.Dim(1))},
			Layout:       t.layout,
		}, cache.Layers(t.caches))
		t.spec.committed(token, auxHidden, t.position, nil)
		t.position += token.Dim(1)
		logits := t.r.Model.Unembed(hidden)
		return []*mlx.Array{logits.Slice(mlx.Slice(), mlx.Slice(logits.Dim(1)-1), mlx.Slice())}
	})[0]
}

// sample dispatches the batched sample over the decoder's rows. On an
// unconstrained step it fuses onto the forward's chain, so the whole step
// is in flight before the previous tokens are synchronized.
func (t *pipelinedDecoder) sample(logits *mlx.Array) sampler.Result {
	next := t.r.Sampler.Sample([]int{t.slotID}, logits.Squeeze(1))
	mlx.AsyncEval(next.Arrays()...)
	t.scope.Attach(next.Arrays()...)
	return next
}

// drain ends production: it returns the in-flight sample (sampled but never
// forwarded) and the position its forward would have taken, surfacing the
// decoder's deferred fault. The decoder keeps the sample for close.
func (t *pipelinedDecoder) drain() ([]sampler.Result, int, error) {
	err := t.err
	if err == nil && t.r.grammarEngine.hasGrammar(t.grammars) {
		// The sample leaves without its forward, so its accept runs here.
		err = t.failRows(t.r.grammarEngine.accept(t.grammars, t.pending.Token.Ints()))
	}
	if err == nil {
		t.scope.Detach(t.pending.Arrays()...)
	}
	return []sampler.Result{t.pending}, t.position, err
}

func (t *pipelinedDecoder) close() {
	// The in-flight sample's forward was never dispatched; its report settles
	// the drafter level with the caches' resting offset.
	t.spec.settle(t.pending.Token)
	t.scope.Close()
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

// clampLogprobs floors logprobs at -9999, the OpenAI-compatible bound;
// tokens a grammar masked out would otherwise report -Inf, which JSON
// cannot encode.
func clampLogprobs(logprobs []llm.Logprob) {
	for i := range logprobs {
		logprobs[i].Logprob = max(logprobs[i].Logprob, -9999)
		for j := range logprobs[i].TopLogprobs {
			logprobs[i].TopLogprobs[j].Logprob = max(logprobs[i].TopLogprobs[j].Logprob, -9999)
		}
	}
}

func (d *detokenizer) detokenize(res sampler.Result) (CompletionResponse, bool) {
	output := res.Token.Int()
	d.buf.WriteString(d.tokenizer.Decode([]int32{output}))
	logprobs := buildLogprob(res, d.wantLogprobs, d.wantTopLogprobs, d.tokenizer.Decode)
	clampLogprobs(logprobs)
	d.logprobs = append(d.logprobs, logprobs...)

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
			Token:   tok(sample.Token.Int()),
			Logprob: float64(sample.Logprob.Floats()[0]),
		},
	}

	if wantTopLogprobs > 0 && sample.TopTokens != nil {
		ids := sample.TopTokens.Ints()
		vals := sample.TopLogprobs.Floats()
		pairs := make([]llm.TokenLogprob, len(ids))
		for i, id := range ids {
			pairs[i] = llm.TokenLogprob{
				Token:   tok(id),
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
