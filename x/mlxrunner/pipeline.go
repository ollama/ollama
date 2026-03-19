package mlxrunner

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func prefillChunkSize() int {
	return 2 << 10
}

func (r *Runner) TextGenerationPipeline(request Request) error {
	if r.Model == nil {
		return errors.New("model not loaded")
	}

	enableCompile := true
	if modelCompile, ok := r.Model.(interface{ EnableCompile() bool }); ok {
		enableCompile = modelCompile.EnableCompile()
	}
	if enableCompile {
		mlx.EnableCompile()
	} else {
		mlx.DisableCompile()
	}
	mlx.ResetPeakMemory()
	ctx := request.Ctx
	var (
		sample, logprobs         *mlx.Array
		nextSample, nextLogprobs *mlx.Array
	)

	defer func() {
		if request.Sampler != nil {
			request.Sampler.Free()
		}
		mlx.Unpin(sample, logprobs)
		mlx.Unpin(nextSample, nextLogprobs)
		mlx.Sweep()
		mlx.ClearCache()

		if slog.Default().Enabled(context.TODO(), logutil.LevelTrace) {
			mlx.LogArrays()
			r.cache.dumpTree()
		}
		slog.Info("peak memory", "size", mlx.PrettyBytes(mlx.PeakMemory()))
	}()

	inputs := r.Tokenizer.Encode(request.Prompt, true)
	if len(inputs) == 0 {
		return errors.New("empty prompt")
	}

	if len(inputs) >= r.contextLength {
		return api.StatusError{
			StatusCode:   http.StatusBadRequest,
			ErrorMessage: fmt.Sprintf("input length (%d tokens) exceeds the model's maximum context length (%d tokens)", len(inputs), r.contextLength),
		}
	}

	// Cap generation to stay within the model's context length
	maxGenerate := r.contextLength - len(inputs)
	if request.Options.MaxTokens <= 0 {
		request.Options.MaxTokens = maxGenerate
	} else {
		request.Options.MaxTokens = min(request.Options.MaxTokens, maxGenerate)
	}

	request.Sampler.ResetHistory(inputs)

	session := r.cache.begin(r.Model, inputs)
	defer session.close()
	caches := session.caches
	tokens := session.remaining
	prefillChunk := prefillChunkSize()

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

		// If there's a pending intermediate snapshot, split the batch
		// so we can capture it at the exact offset. The cache offset
		// after this batch will be: baseOffset + processed + n.
		if session.snapshotOffset > 0 {
			baseOffset := len(session.inputs) - len(tokens)
			tokensUntilSnapshot := session.snapshotOffset - (baseOffset + processed)
			if tokensUntilSnapshot > 0 && tokensUntilSnapshot < n {
				n = tokensUntilSnapshot
			}
		}

		r.Model.Forward(mlx.FromValues(tokens[processed:processed+n], n).ExpandDims(0), caches)
		mlx.Sweep()
		materializeCaches()
		processed += n
		slog.Info("Prompt processing progress", "processed", processed, "total", total)

		// Create snapshot at branch point for future diverging requests.
		if session.snapshotOffset > 0 {
			baseOffset := len(session.inputs) - len(tokens)
			if baseOffset+processed >= session.snapshotOffset {
				session.snapshot(false)
			}
		}

		mlx.ClearCache()
	}

	step := func(token *mlx.Array) (*mlx.Array, *mlx.Array) {
		fwd := r.Model.Forward(token.ExpandDims(0), caches)
		logits := r.Model.Unembed(fwd)
		logits = logits.Slice(mlx.Slice(), mlx.Slice(logits.Dim(1)-1), mlx.Slice()).Squeeze(1)

		logprobs := logits.Subtract(logits.Logsumexp(true))
		sample := request.Sampler.Sample(logprobs)

		mlx.Pin(sample, logprobs)
		mlx.Sweep()
		mlx.AsyncEval(sample, logprobs)

		return sample, logprobs
	}

	sample, logprobs = step(mlx.FromValues(tokens[processed:], total-processed))

	var b bytes.Buffer

	final := CompletionResponse{Done: true, PromptEvalCount: len(inputs), EvalCount: request.Options.MaxTokens, DoneReason: 1}
	for i := range request.Options.MaxTokens {
		if err := ctx.Err(); err != nil {
			return err
		}

		request.Sampler.AppendToken(sample)
		nextSample, nextLogprobs = step(sample)

		if i == 0 {
			mlx.Eval(sample)
			final.PromptEvalDuration = time.Since(now)
			now = time.Now()
		}

		output := int32(sample.Int())
		session.outputs = append(session.outputs, output)

		if r.Tokenizer.IsEOS(output) {
			final.DoneReason = 0
			final.EvalCount = i
			break
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case request.Responses <- CompletionResponse{
			Content: r.Decode(output, &b),
		}:
		}

		mlx.Unpin(sample, logprobs)
		sample, logprobs = nextSample, nextLogprobs
		nextSample, nextLogprobs = nil, nil

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

func (r Runner) Decode(sample int32, b *bytes.Buffer) string {
	token := r.Tokenizer.Decode([]int32{sample})

	if _, err := b.WriteString(token); err != nil {
		slog.Error("Failed to write token to buffer", "error", err)
		return ""
	}

	return flushValidUTF8Prefix(b)
}
