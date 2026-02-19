//go:build mlx

package mlxrunner

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"time"

	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

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

	inputs := r.Tokenizer.Encode(request.Prompt, true)

	caches, tokens := r.FindNearestCache(inputs)
	if len(caches) == 0 {
		if cacheFactory, ok := r.Model.(interface{ NewCaches() []cache.Cache }); ok {
			caches = cacheFactory.NewCaches()
		} else {
			caches = make([]cache.Cache, r.Model.NumLayers())
			for i := range caches {
				caches[i] = cache.NewKVCache()
			}
		}
	}

	total, processed := len(tokens), 0
	slog.Info("Prompt processing progress", "processed", processed, "total", total)
	for total-processed > 1 {
		n := min(2<<10, total-processed-1)
		r.Model.Forward(mlx.FromValues(tokens[processed:processed+n], n).ExpandDims(0), caches)
		mlx.Sweep()
		mlx.Eval(func() []*mlx.Array {
			s := make([]*mlx.Array, 2*len(caches))
			for i, c := range caches {
				s[2*i], s[2*i+1] = c.State()
			}
			return s
		}()...)
		processed += n
		slog.Info("Prompt processing progress", "processed", processed, "total", total)
		mlx.ClearCache()
	}

	step := func(token *mlx.Array) (*mlx.Array, *mlx.Array) {
		fwd := r.Model.Forward(token.ExpandDims(0), caches)
		logits := r.Model.Unembed(fwd)
		logits = logits.Slice(mlx.Slice(), mlx.Slice(logits.Dim(1)-1), mlx.Slice()).Squeeze(1)

		logprobs := logits.Subtract(logits.Logsumexp(true))
		sample := request.Sample(logprobs)

		mlx.Pin(sample, logprobs)
		mlx.Sweep()
		mlx.AsyncEval(sample, logprobs)

		return sample, logprobs
	}

	sample, logprobs := step(mlx.FromValues(tokens[processed:], total-processed))

	var b bytes.Buffer

	now := time.Now()
	final := Response{Done: true, PromptTokens: total, CompletionTokens: request.Options.MaxTokens, DoneReason: 1}
	outputs := make([]int32, 0, request.Options.MaxTokens)
	for i := range request.Options.MaxTokens {
		nextSample, nextLogprobs := step(sample)

		if i == 0 {
			slog.Info("Prompt processing progress", "processed", total, "total", total)
			mlx.Eval(sample)
			final.PromptTokensDuration = time.Since(now)
			now = time.Now()
		}

		output := int32(sample.Int())
		outputs = append(outputs, output)

		if r.Tokenizer.IsEOS(output) {
			mlx.Unpin(nextSample, nextLogprobs)
			final.Token = int(output)
			final.DoneReason = 0
			final.CompletionTokens = i
			break
		}

		request.Responses <- Response{
			Text:  r.Decode(output, &b),
			Token: int(output),
		}

		mlx.Unpin(sample, logprobs)
		if i%256 == 0 {
			mlx.ClearCache()
		}

		sample, logprobs = nextSample, nextLogprobs
	}

	mlx.Unpin(sample, logprobs)
	final.CompletionTokensDuration = time.Since(now)
	request.Responses <- final
	r.InsertCache(append(inputs, outputs...), caches)
	mlx.Sweep()

	if slog.Default().Enabled(context.TODO(), logutil.LevelTrace) {
		mlx.LogArrays()
	}

	return nil
}

func (r Runner) Decode(sample int32, b *bytes.Buffer) string {
	token := r.Tokenizer.Decode([]int32{sample})

	if _, err := b.WriteString(token); err != nil {
		slog.Error("Failed to write token to buffer", "error", err)
		return ""
	}

	return flushValidUTF8Prefix(b)
}
