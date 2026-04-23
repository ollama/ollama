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

func (r *Runner) TextGenerationPipeline(ctx context.Context, request Request) error {
	mlx.ResetPeakMemory()
	var sample, nextSample sampler.Result

	defer func() {
		if request.Sampler != nil {
			request.Sampler.Free()
		}
		mlx.Unpin(sample.Arrays()...)
		mlx.Unpin(nextSample.Arrays()...)
		mlx.Sweep()
		mlx.ClearCache()

		if slog.Default().Enabled(context.TODO(), logutil.LevelTrace) {
			mlx.LogArrays()
			r.cache.dumpTree()
		}
		slog.Info("peak memory", "size", mlx.PrettyBytes(mlx.PeakMemory()))
	}()

	inputs := request.Tokens
	request.Sampler.ResetHistory(inputs)

	session := r.cache.begin(r.Model, inputs)
	defer session.close()

	caches := session.caches
	tokens := session.remaining
	prefillChunk := prefillChunkSize()

	// Request periodic snapshots during prefill and near the end of the
	// prompt so that long prompts can be partially restored and
	// thinking/generation can be retried without full reprocessing.
	const snapshotInterval = 8192
	for offset := snapshotInterval; offset < len(inputs); offset += snapshotInterval {
		session.requestSnapshot(offset)
	}

	const preThinking = 4
	if end := len(inputs) - preThinking; end > 0 {
		session.requestSnapshot(end)
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

	now := time.Now()
	total, processed := len(tokens), 0
	for total-processed > 1 {
		if err := ctx.Err(); err != nil {
			return err
		}

		n := min(prefillChunk, total-processed-1)

		// If there's a pending snapshot, split the batch so we can
		// capture it at the exact offset.
		if snapOffset := session.nextPendingSnapshot(); snapOffset > 0 {
			baseOffset := len(session.inputs) - len(tokens)
			tokensUntilSnapshot := snapOffset - (baseOffset + processed)
			if tokensUntilSnapshot > 0 && tokensUntilSnapshot < n {
				n = tokensUntilSnapshot
			}
		}

		r.Model.Forward(mlx.FromValues(tokens[processed:processed+n], n).ExpandDims(0), caches)
		mlx.Sweep()
		materializeCaches()
		processed += n
		slog.Info("Prompt processing progress", "processed", processed, "total", total)

		// Create snapshot if we've reached a pending offset.
		if snapOffset := session.nextPendingSnapshot(); snapOffset > 0 {
			baseOffset := len(session.inputs) - len(tokens)
			if baseOffset+processed >= snapOffset {
				session.snapshot()
			}
		}

		mlx.ClearCache()
	}

	step := func(token *mlx.Array) sampler.Result {
		fwd := r.Model.Forward(token.ExpandDims(0), caches)
		logits := r.Model.Unembed(fwd)
		logits = logits.Slice(mlx.Slice(), mlx.Slice(logits.Dim(1)-1), mlx.Slice()).Squeeze(1)

		sample := request.Sampler.Sample(logits)
		mlx.Pin(sample.Arrays()...)
		mlx.Sweep()
		mlx.AsyncEval(sample.Arrays()...)
		return sample
	}

	sample = step(mlx.FromValues(tokens[processed:], total-processed))

	dec := decoder{tokenizer: r.Tokenizer}

	final := CompletionResponse{Done: true, PromptEvalCount: len(inputs), EvalCount: request.Options.NumPredict, DoneReason: 1}
	for i := range request.Options.NumPredict {
		if err := ctx.Err(); err != nil {
			return err
		}

		request.Sampler.AppendToken(sample.Token)
		nextSample = step(sample.Token)

		if i == 0 {
			mlx.Eval(sample.Arrays()...)
			final.PromptEvalDuration = time.Since(now)
			now = time.Now()
		}

		output := int32(sample.Token.Int())
		session.outputs = append(session.outputs, output)

		if r.Tokenizer.IsEOS(output) {
			final.DoneReason = 0
			final.EvalCount = i
			break
		}

		if resp, ok := dec.decode(sample); ok {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case request.Responses <- resp:
			}
		}

		mlx.Unpin(sample.Arrays()...)
		sample, nextSample = nextSample, sampler.Result{}

		if i%256 == 0 {
			mlx.ClearCache()
		}
	}

	final.EvalDuration = time.Since(now)
	select {
	case <-ctx.Done():
		return ctx.Err()
	case request.Responses <- final:
		return nil
	}
}

// decoder serializes sampled tokens into response chunks, holding bytes
// whose UTF-8 sequence hasn't completed yet and the logprobs that belong
// with those bytes so Content and Logprobs stay aligned when a chunk does
// flush.
type decoder struct {
	tokenizer *tokenizer.Tokenizer
	buf       bytes.Buffer
	logprobs  []llm.Logprob
}

func (d *decoder) decode(res sampler.Result) (CompletionResponse, bool) {
	output := int32(res.Token.Int())
	d.buf.WriteString(d.tokenizer.Decode([]int32{output}))
	d.logprobs = append(d.logprobs, buildLogprob(res, d.tokenizer.Decode)...)

	content := flushValidUTF8Prefix(&d.buf)
	if content == "" {
		return CompletionResponse{}, false
	}
	resp := CompletionResponse{Content: content, Logprobs: d.logprobs}
	d.logprobs = nil
	return resp, true
}

func buildLogprob(sample sampler.Result, decode func([]int32) string) []llm.Logprob {
	if sample.Logprob == nil {
		return nil
	}
	tok := func(id int32) string { return decode([]int32{id}) }

	out := llm.Logprob{
		TokenLogprob: llm.TokenLogprob{
			Token:   tok(int32(sample.Token.Int())),
			Logprob: float64(sample.Logprob.Floats()[0]),
		},
	}

	if sample.TopTokens != nil {
		ids := sample.TopTokens.Ints()
		vals := sample.TopLogprobs.Floats()
		pairs := make([]llm.TokenLogprob, len(ids))
		for i, id := range ids {
			pairs[i] = llm.TokenLogprob{
				Token:   tok(int32(id)),
				Logprob: float64(vals[i]),
			}
		}
		sort.Slice(pairs, func(i, j int) bool {
			return pairs[i].Logprob > pairs[j].Logprob
		})
		out.TopLogprobs = pairs
	}
	return []llm.Logprob{out}
}
