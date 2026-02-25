//go:build mlx

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
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func prefillChunkSize(lowMemoryDecode bool) int {
	if lowMemoryDecode {
		return 32
	}
	return 2 << 10
}

func hasRecurrentCaches(caches []cache.Cache) bool {
	for _, c := range caches {
		if c == nil {
			continue
		}
		if _, ok := c.(*cache.RecurrentCache); ok {
			return true
		}
	}
	return false
}

func (r *Runner) TextGenerationPipeline(request Request) error {
	if r.Model == nil {
		return errors.New("model not loaded")
	}

	ctx := request.Ctx
	if ctx == nil {
		ctx = context.Background()
	}

	var (
		sample, logprobs         *mlx.Array
		nextSample, nextLogprobs *mlx.Array
	)

	defer func() {
		mlx.Unpin(sample, logprobs)
		mlx.Unpin(nextSample, nextLogprobs)
		mlx.Sweep()
		mlx.ClearCache()

		if slog.Default().Enabled(context.TODO(), logutil.LevelTrace) {
			mlx.LogArrays()
			r.cache.log()
		}
	}()

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

	session := r.cache.begin(r.Model, inputs)
	defer session.close()
	caches := session.caches
	tokens := session.remaining
	lowMemoryDecode := false
	if m, ok := r.Model.(interface{ LowMemoryDecode() bool }); ok {
		lowMemoryDecode = m.LowMemoryDecode()
	}
	// Keep recurrent models on the conservative decode path to avoid async graph
	// growth until the faster recurrent async path is reintroduced separately.
	if hasRecurrentCaches(caches) {
		lowMemoryDecode = true
	}
	prefillChunk := prefillChunkSize(lowMemoryDecode)

	materializeCaches := func() {
		state := make([]*mlx.Array, 0, 2*len(caches))
		for _, c := range caches {
			if c == nil {
				continue
			}
			state = append(state, c.Materialize()...)
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
		r.Model.Forward(mlx.FromValues(tokens[processed:processed+n], n).ExpandDims(0), caches)
		mlx.Sweep()
		materializeCaches()
		processed += n
		slog.Info("Prompt processing progress", "processed", processed, "total", total)
		mlx.ClearCache()
	}

	step := func(token *mlx.Array, async bool) (*mlx.Array, *mlx.Array) {
		fwd := r.Model.Forward(token.ExpandDims(0), caches)
		logits := r.Model.Unembed(fwd)
		logits = logits.Slice(mlx.Slice(), mlx.Slice(logits.Dim(1)-1), mlx.Slice()).Squeeze(1)

		logprobs := logits.Subtract(logits.Logsumexp(true))
		sample := request.Sample(logprobs)

		mlx.Pin(sample, logprobs)
		mlx.Sweep()
		if async {
			mlx.AsyncEval(sample, logprobs)
		}

		return sample, logprobs
	}

	sample, logprobs = step(mlx.FromValues(tokens[processed:], total-processed), !lowMemoryDecode)
	if lowMemoryDecode {
		materializeCaches()
	}

	var b bytes.Buffer

	final := CompletionResponse{Done: true, PromptEvalCount: len(inputs), EvalCount: request.Options.MaxTokens, DoneReason: 1}
	for i := range request.Options.MaxTokens {
		if err := ctx.Err(); err != nil {
			return err
		}

		nextSample, nextLogprobs = nil, nil
		if !lowMemoryDecode {
			nextSample, nextLogprobs = step(sample, true)
		}

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
		case <-request.Ctx.Done():
			return request.Ctx.Err()
		case request.Responses <- CompletionResponse{
			Content: r.Decode(output, &b),
		}:
		}

		if lowMemoryDecode {
			mlx.Unpin(sample, logprobs)
			sample, logprobs = nil, nil
			mlx.Sweep()
			if i+1 >= request.Options.MaxTokens {
				break
			}
			mlx.ClearCache()
			sample, logprobs = step(mlx.FromValues([]int32{output}, 1), false)
			materializeCaches()
			continue
		}

		mlx.Unpin(sample, logprobs)
		sample, logprobs = nextSample, nextLogprobs
		nextSample, nextLogprobs = nil, nil

		if i%256 == 0 {
			mlx.ClearCache()
		}
	}

	final.EvalDuration = time.Since(now)
	final.PeakMemory = uint64(mlx.PeakMemory())
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
