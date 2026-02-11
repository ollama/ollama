//go:build mlx

package mlxrunner

import (
	"bytes"
	"errors"
	"log/slog"
	"time"
	"unicode/utf8"

	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func (r *Runner) TextGenerationPipeline(request Request) error {
	if r.Model == nil {
		return errors.New("model not loaded")
	}

	inputs := r.Tokenizer.Encode(request.Prompt, true)

	caches, tokens := r.FindNearestCache(inputs)
	if len(caches) == 0 {
		caches = make([]cache.Cache, r.Model.NumLayers())
		for i := range caches {
			caches[i] = cache.NewKVCache()
		}
	}

	total, processed := len(tokens), 0
	slog.Info("Prompt processing progress", "processed", processed, "total", total)
	for total-processed > 1 {
		n := min(2<<10, total-processed-1)
		temp := r.Model.Forward(mlx.FromValues(tokens[processed:processed+n], n).ExpandDims(0), caches)
		defer mlx.Free(temp)
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
		logits := r.Model.Unembed(r.Model.Forward(token.ExpandDims(0), caches))
		logits = logits.Slice(mlx.Slice(), mlx.Slice(logits.Dim(1)-1), mlx.Slice()).Squeeze(1)

		logprobs := logits.Subtract(logits.Logsumexp(true))
		return request.Sample(logprobs), logprobs
	}

	sample, logprobs := step(mlx.FromValues(tokens[processed:], total-processed))
	mlx.AsyncEval(sample, logprobs)

	var b bytes.Buffer

	now := time.Now()
	final := Response{PromptTokens: total, CompletionTokens: request.Options.MaxTokens, DoneReason: 1}
	outputs := make([]int32, 0, request.Options.MaxTokens)
	for i := range request.Options.MaxTokens {
		nextSample, nextLogprobs := step(sample)
		mlx.AsyncEval(nextSample, nextLogprobs)

		if i == 0 {
			slog.Info("Prompt processing progress", "processed", total, "total", total)
			mlx.Eval(sample)
			final.PromptTokensDuration = time.Since(now)
			now = time.Now()
		}

		output := int32(sample.Int())
		outputs = append(outputs, output)

		if r.Tokenizer.IsEOS(output) {
			final.Token = int(output)
			final.DoneReason = 0
			final.CompletionTokens = i
			break
		}

		request.Responses <- Response{
			Text:  r.Decode(output, &b),
			Token: int(output),
		}

		mlx.Free(sample, logprobs)
		if i%256 == 0 {
			mlx.ClearCache()
		}

		sample, logprobs = nextSample, nextLogprobs
	}

	mlx.Free(sample, logprobs)
	final.CompletionTokensDuration = time.Since(now)
	request.Responses <- final
	r.InsertCache(append(inputs, outputs...), caches)
	return nil
}

func (r Runner) Decode(sample int32, b *bytes.Buffer) string {
	token := r.Tokenizer.Decode([]int32{sample})

	if _, err := b.WriteString(token); err != nil {
		slog.Error("Failed to write token to buffer", "error", err)
		return ""
	}

	if text := b.String(); utf8.ValidString(text) {
		b.Reset()
		return text
	} else if b.Len() >= utf8.UTFMax {
		b.Reset()
		return text
	}

	return ""
}
